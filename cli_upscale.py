#!/usr/bin/env python3
"""
GeoTIFF & Sentinel-2 Command Line AI Upscaler.
Direct-to-disk streaming upscaler supporting gigapixel satellite imagery,
preserving CRS projections, affine geotransforms, and geospatial metadata.

Usage:
  python cli_upscale.py --input path/to/image.tif
  python cli_upscale.py --input path/to/mosaic.zip --model RealESRGAN_x4plus.pth --tile-size 512
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

try:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.enums import Resampling, ColorInterp
    import rasterio.windows
except ImportError:
    print("[ERROR] rasterio is required. Please run inside Docker or install rasterio.")
    sys.exit(1)


def load_model(model_name="RealESRGAN_x4plus.pth", models_dir="/app/models/upscale_models"):
    if not os.path.exists(models_dir):
        # Check local paths
        for p in ["storage/models/upscale_models", "models/upscale_models"]:
            if os.path.exists(p):
                models_dir = p
                break

    model_path = os.path.join(models_dir, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Upscale model not found: {model_path}")

    import comfy.utils
    from spandrel import ModelLoader
    try:
        import spandrel_extra_arches
        spandrel_extra_arches.install()
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Model] Loading '{model_name}' on {device}...")
    sd = comfy.utils.load_torch_file(model_path, safe_load=True)
    model = ModelLoader().load_from_state_dict(sd).eval()
    model.to(device)
    return model, device


def apply_satellite_stretch(data, nodata_val=None):
    valid_mask = np.ones(data.shape[:2], dtype=bool)
    if nodata_val is not None and nodata_val != 0:
        for c in range(data.shape[2]):
            valid_mask = valid_mask & (data[:, :, c] != nodata_val) & (~np.isnan(data[:, :, c]))

    for c in range(data.shape[2]):
        valid_mask = valid_mask & (data[:, :, c] > -1000)

    if not np.any(valid_mask):
        return np.zeros_like(data, dtype=np.float32), valid_mask

    max_val = float(np.max(data[valid_mask]))
    min_val = float(np.min(data[valid_mask]))

    if max_val <= 1.0 and min_val >= 0.0:
        out = np.clip(data, 0.0, 1.0)
    elif max_val <= 255.0 and min_val >= 0.0:
        out = np.clip(data / 255.0, 0.0, 1.0)
    elif max_val <= 12000.0:
        p98 = float(np.percentile(data[valid_mask], 98))
        scale_target = max(min(p98 * 1.1, 4000.0), 1000.0)
        out = np.clip(data / scale_target, 0.0, 1.0) ** (1.0 / 1.15)
    else:
        p_low = float(np.percentile(data[valid_mask], 1))
        p_high = float(np.percentile(data[valid_mask], 99))
        rng = max(p_high - p_low, 1.0)
        out = np.clip((data - p_low) / rng, 0.0, 1.0)

    return out.astype(np.float32), valid_mask


def upscale_geotiff(input_path, output_path, model_name="RealESRGAN_x4plus.pth",
                    tile_size=512, overlap=32, compression="deflate"):
    start_time = time.time()
    
    # Check if input is zip
    if input_path.lower().endswith('.zip'):
        print(f"[Input] Detected Sentinel-2 zip archive: {input_path}")
        from convert_sentinel2 import convert_sentinel2_zip
        temp_rgb = input_path.replace('.zip', '_RGB_temp.tif')
        convert_sentinel2_zip(input_path, temp_rgb, preset="natural_color")
        input_path = temp_rgb

    with rasterio.open(input_path) as src:
        w_in, h_in = src.width, src.height
        crs = src.crs
        transform = src.transform
        count = src.count
        nodata = src.nodata
        dtype = src.dtypes[0]

        model, device = load_model(model_name)
        scale = getattr(model, 'scale', 4)

        w_out, h_out = w_in * scale, h_in * scale
        new_transform = Affine(transform.a / scale, transform.b / scale, transform.c,
                               transform.d / scale, transform.e / scale, transform.f)

        gpix = (w_out * h_out) / 1e9
        print(f"[Raster] Input: {w_in}x{h_in} ({count} bands, {dtype}) | CRS: {crs}")
        print(f"[Raster] Output: {w_out}x{h_out} ({scale}x Upscale, {gpix:.2f} Gpix)")

        stride = tile_size - overlap
        y_steps = list(range(0, h_in, stride))
        x_steps = list(range(0, w_in, stride))
        total_tiles = len(y_steps) * len(x_steps)
        print(f"[Tiler] Processing {total_tiles} tiles ({tile_size}x{tile_size}, overlap {overlap}px)...")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        profile = {
            'driver': 'GTiff',
            'height': h_out,
            'width': w_out,
            'count': 3,
            'dtype': 'uint8',
            'crs': crs,
            'transform': new_transform,
            'compress': compression.lower() if compression != "none" else None,
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
            'bigtiff': 'YES',
            'predictor': 2
        }

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]

            tile_count = 0
            for y in y_steps:
                for x in x_steps:
                    y_end = min(y + tile_size, h_in)
                    x_end = min(x + tile_size, w_in)
                    y_start = max(0, y_end - tile_size)
                    x_start = max(0, x_end - tile_size)

                    cur_th, cur_tw = y_end - y_start, x_end - x_start
                    win_in = rasterio.windows.Window(x_start, y_start, cur_tw, cur_th)

                    if count == 1:
                        raw_b = src.read(1, window=win_in)
                        raw_tile = np.repeat(raw_b[np.newaxis, :, :], 3, axis=0)
                    elif count >= 3:
                        raw_tile = src.read([1, 2, 3], window=win_in)
                    else:
                        raw_tile = np.zeros((3, cur_th, cur_tw), dtype=np.float32)

                    raw_tile = np.transpose(raw_tile.astype(np.float32), (1, 2, 0))
                    norm_tile, _ = apply_satellite_stretch(raw_tile, nodata_val=nodata)

                    t_in = torch.from_numpy(norm_tile).permute(2, 0, 1).unsqueeze(0).to(device)
                    with torch.no_grad():
                        t_out = model(t_in).squeeze(0).permute(1, 2, 0).cpu().numpy()

                    out_np = np.clip(t_out * 255.0, 0, 255).astype(np.uint8)

                    pad_left = (x - x_start) * scale
                    pad_top = (y - y_start) * scale
                    write_w = min(stride, w_in - x) * scale
                    write_h = min(stride, h_in - y) * scale

                    sub_out = out_np[pad_top:pad_top+write_h, pad_left:pad_left+write_w, :]
                    sub_out = np.transpose(sub_out, (2, 0, 1))

                    win_out = rasterio.windows.Window(x * scale, y * scale, write_w, write_h)
                    dst.write(sub_out, window=win_out)

                    tile_count += 1
                    if tile_count % 20 == 0 or tile_count == total_tiles:
                        pct = (tile_count / total_tiles) * 100
                        elapsed = time.time() - start_time
                        fps = tile_count / max(elapsed, 0.1)
                        print(f"  Progress: {tile_count}/{total_tiles} ({pct:.1f}%) - {fps:.1f} tiles/s", end="\r")

    elapsed = time.time() - start_time
    file_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n[DONE] Upscaled GeoTIFF saved: {output_path} ({file_mb:.1f} MB) in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="GeoTIFF & Sentinel-2 Command Line AI Upscaler")
    parser.add_argument("--input", "-i", required=True, help="Path to input GeoTIFF (.tif) or Sentinel-2 (.zip)")
    parser.add_argument("--output", "-o", default=None, help="Path to output upscaled GeoTIFF (.tif)")
    parser.add_argument("--model", "-m", default="RealESRGAN_x4plus.pth", help="Upscale model name (default: RealESRGAN_x4plus.pth)")
    parser.add_argument("--tile-size", "-t", type=int, default=512, help="Tile size in pixels (default: 512)")
    parser.add_argument("--overlap", type=int, default=32, help="Tile overlap in pixels (default: 32)")
    parser.add_argument("--compression", default="deflate", choices=["deflate", "lzw", "zstd", "none"], help="TIFF compression")

    args = parser.parse_args()

    if args.output is None:
        base, _ = os.path.splitext(args.input)
        args.output = f"{base}_Upscaled_4x.tif"

    upscale_geotiff(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        tile_size=args.tile_size,
        overlap=args.overlap,
        compression=args.compression
    )


if __name__ == "__main__":
    main()
