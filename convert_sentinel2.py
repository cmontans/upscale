#!/usr/bin/env python3
"""
Sentinel-2 Mosaic Zip to Georeferenced GeoTIFF Converter.
Converts Sentinel-2 quarterly/seasonal mosaic zip archives (containing B02, B03, B04, B08)
into seamless True-Color RGB, False-Color NIR, or 4-band GeoTIFF rasters ready for AI upscaling and GIS.
"""

import os
import sys
import glob
import json
import zipfile
import argparse
import numpy as np

try:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.enums import Resampling, ColorInterp
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def find_band_in_zip(zip_path, band_name):
    """Finds the relative path of a specific band (e.g. 'B04', 'B03', 'B02', 'B08') inside a zip file."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        for name in z.namelist():
            base = os.path.basename(name).upper()
            if base == f"{band_name.upper()}.TIF" or base.startswith(f"{band_name.upper()}_"):
                return name
    return None


def get_zip_metadata(zip_path):
    """Extracts userdata.json or product metadata from zip if available."""
    metadata = {}
    with zipfile.ZipFile(zip_path, 'r') as z:
        for name in z.namelist():
            if name.endswith('.json'):
                try:
                    data = json.loads(z.read(name).decode('utf-8'))
                    metadata[name] = data
                except Exception:
                    pass
    return metadata


def tone_map_sentinel2(red, green, blue, nodata_mask=None, preset="natural_color"):
    """
    Applies radiometric tone mapping and contrast enhancement for Sentinel-2 surface reflectance.
    Standard Sentinel-2 BOA reflectance ranges from 0 to ~10,000 (100% reflectance).
    """
    if preset == "raw_16bit":
        # Keep raw uint16 values clipped to valid range
        r = np.clip(red, 0, 10000).astype(np.uint16)
        g = np.clip(green, 0, 10000).astype(np.uint16)
        b = np.clip(blue, 0, 10000).astype(np.uint16)
        return r, g, b, 'uint16'

    # Convert to float for processing
    r = red.astype(np.float32)
    g = green.astype(np.float32)
    b = blue.astype(np.float32)

    # Clean invalid/NoData values (<0)
    valid_mask = (r > 0) & (g > 0) & (b > 0)
    if nodata_mask is not None:
        valid_mask = valid_mask & (~nodata_mask)

    if preset == "percentile":
        if np.any(valid_mask):
            p_low = np.percentile(np.concatenate([r[valid_mask], g[valid_mask], b[valid_mask]]), 1)
            p_high = np.percentile(np.concatenate([r[valid_mask], g[valid_mask], b[valid_mask]]), 99)
        else:
            p_low, p_high = 0, 3000
        rng = max(p_high - p_low, 100)
        r = np.clip((r - p_low) / rng, 0.0, 1.0)
        g = np.clip((g - p_low) / rng, 0.0, 1.0)
        b = np.clip((b - p_low) / rng, 0.0, 1.0)
    elif preset == "natural_color":
        # Realistic atmospheric & reflectance compensation for Sentinel-2
        # Max reflectance clip around 3200-3500, slight gamma boost (1.15) for shadow clarity
        r_scale = 3200.0
        g_scale = 3000.0
        b_scale = 2600.0

        r = np.clip(r / r_scale, 0.0, 1.0) ** (1.0 / 1.15)
        g = np.clip(g / g_scale, 0.0, 1.0) ** (1.0 / 1.15)
        b = np.clip(b / b_scale, 0.0, 1.0) ** (1.0 / 1.15)
    elif preset == "linear_reflectance":
        r = np.clip(r / 10000.0, 0.0, 1.0)
        g = np.clip(g / 10000.0, 0.0, 1.0)
        b = np.clip(b / 10000.0, 0.0, 1.0)

    # Convert to 8-bit RGB [0..255]
    r_out = (r * 255.0).astype(np.uint8)
    g_out = (g * 255.0).astype(np.uint8)
    b_out = (b * 255.0).astype(np.uint8)

    # Set NoData pixels to 0
    r_out[~valid_mask] = 0
    g_out[~valid_mask] = 0
    b_out[~valid_mask] = 0

    return r_out, g_out, b_out, 'uint8'


def convert_sentinel2_zip(zip_path, output_path=None, mode="rgb", preset="natural_color",
                          tile_size=2048, build_pyramids=True):
    """
    Converts a Sentinel-2 mosaic zip archive into a single georeferenced GeoTIFF.
    Streams in blocks to minimize memory consumption on large (10,000+ px) rasters.
    """
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio is required to run convert_sentinel2.py. Please install rasterio.")

    zip_path = os.path.abspath(zip_path)
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Sentinel-2 zip file not found: {zip_path}")

    # Identify bands
    b04_subpath = find_band_in_zip(zip_path, "B04")  # Red
    b03_subpath = find_band_in_zip(zip_path, "B03")  # Green
    b02_subpath = find_band_in_zip(zip_path, "B02")  # Blue
    b08_subpath = find_band_in_zip(zip_path, "B08")  # NIR

    if not (b04_subpath and b03_subpath and b02_subpath):
        raise ValueError(f"Could not find required RGB bands (B04, B03, B02) in zip: {zip_path}")

    zip_vsi_prefix = f"/vsizip/{zip_path.replace(os.sep, '/')}"
    b04_vsi = f"{zip_vsi_prefix}/{b04_subpath}"
    b03_vsi = f"{zip_vsi_prefix}/{b03_subpath}"
    b02_vsi = f"{zip_vsi_prefix}/{b02_subpath}"
    b08_vsi = f"{zip_vsi_prefix}/{b08_subpath}" if b08_subpath else None

    # Derive output filename if not specified
    if not output_path:
        base_name = os.path.splitext(os.path.basename(zip_path))[0]
        mode_tag = mode.upper()
        output_path = os.path.join("storage", "input", f"{base_name}_{mode_tag}.tif")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"[*] Opening Sentinel-2 archive: {os.path.basename(zip_path)}")
    with rasterio.open(b04_vsi) as src_ref:
        profile = src_ref.profile.copy()
        crs = src_ref.crs
        transform = src_ref.transform
        width = src_ref.width
        height = src_ref.height

        print(f"    Dimensions: {width} x {height} px")
        print(f"    CRS:        {crs}")
        print(f"    Resolution: dx={transform.a:.2f}m, dy={transform.e:.2f}m")

        # Determine band count & output profile
        band_count = 4 if mode == "4band" and b08_vsi else 3
        out_dtype = 'uint16' if preset == "raw_16bit" else 'uint8'
        nodata_val = 0 if out_dtype == 'uint8' else (src_ref.nodata if src_ref.nodata is not None else 0)

        profile.update({
            'driver': 'GTiff',
            'count': band_count,
            'dtype': out_dtype,
            'nodata': nodata_val,
            'crs': crs,
            'transform': transform,
            'compress': 'deflate',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
            'bigtiff': 'IF_SAFER',
            'predictor': 2
        })

        print(f"[*] Processing and writing to: {output_path}")

        # Open band sources
        with rasterio.open(b04_vsi) as src_r, \
             rasterio.open(b03_vsi) as src_g, \
             rasterio.open(b02_vsi) as src_b:

            src_nir = rasterio.open(b08_vsi) if (mode in ["cir", "4band"] and b08_vsi) else None

            with rasterio.open(output_path, 'w', **profile) as dst:
                # Color interpretation
                if band_count == 3:
                    dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]
                elif band_count == 4:
                    dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]

                # Process in chunked windows for optimal memory usage
                for y in range(0, height, tile_size):
                    win_h = min(tile_size, height - y)
                    for x in range(0, width, tile_size):
                        win_w = min(tile_size, width - x)
                        window = rasterio.windows.Window(x, y, win_w, win_h)

                        # Read bands for this block
                        r_block = src_r.read(1, window=window)
                        g_block = src_g.read(1, window=window)
                        b_block = src_b.read(1, window=window)

                        if mode == "cir" and src_nir:
                            # False Color NIR: Red = NIR (B08), Green = Red (B04), Blue = Green (B03)
                            nir_block = src_nir.read(1, window=window)
                            r_out, g_out, b_out, _ = tone_map_sentinel2(nir_block, r_block, g_block, preset=preset)
                            dst.write(r_out, 1, window=window)
                            dst.write(g_out, 2, window=window)
                            dst.write(b_out, 3, window=window)
                        elif mode == "4band" and src_nir:
                            nir_block = src_nir.read(1, window=window)
                            r_out, g_out, b_out, _ = tone_map_sentinel2(r_block, g_block, b_block, preset=preset)
                            nir_out, _, _, _ = tone_map_sentinel2(nir_block, nir_block, nir_block, preset=preset)
                            dst.write(r_out, 1, window=window)
                            dst.write(g_out, 2, window=window)
                            dst.write(b_out, 3, window=window)
                            dst.write(nir_out, 4, window=window)
                        else:
                            # Standard True Color RGB
                            r_out, g_out, b_out, _ = tone_map_sentinel2(r_block, g_block, b_block, preset=preset)
                            dst.write(r_out, 1, window=window)
                            dst.write(g_out, 2, window=window)
                            dst.write(b_out, 3, window=window)

                if src_nir:
                    src_nir.close()

                # Build overviews (pyramids) for fast GIS rendering
                if build_pyramids and (width > 1024 or height > 1024):
                    print("[*] Generating pyramid overviews for GIS visualization...")
                    factors = [2, 4, 8, 16, 32]
                    try:
                        dst.build_overviews(factors, Resampling.average)
                        dst.update_tags(ns='rio_overview', resampling='average')
                    except Exception as e:
                        print(f"[Warning] Could not build overviews: {e}")

    print(f"[SUCCESS] GeoTIFF created: {output_path}\n")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert Sentinel-2 mosaic zip archives to GeoTIFF rasters.")
    parser.add_argument("input", nargs="?", default=None, help="Path to Sentinel-2 .zip file or directory containing zips.")
    parser.add_argument("-o", "--output", default=None, help="Output GeoTIFF path or directory.")
    parser.add_argument("-m", "--mode", choices=["rgb", "cir", "4band"], default="rgb",
                        help="Band combination mode: rgb (True Color), cir (Color Infrared / NIR), 4band (RGB+NIR). Default: rgb")
    parser.add_argument("-p", "--preset", choices=["natural_color", "percentile", "linear_reflectance", "raw_16bit"],
                        default="natural_color", help="Tone mapping / stretch preset. Default: natural_color")
    parser.add_argument("--downloads", action="store_true", help="Automatically search for Sentinel-2 zips in C:\\Users\\carlo\\Downloads")

    args = parser.parse_args()

    # Determine files to process
    zip_files = []
    if args.downloads:
        user_downloads = os.path.expanduser("~/Downloads")
        zip_files.extend(glob.glob(os.path.join(user_downloads, "*Sentinel*2*.zip")))
        zip_files.extend(glob.glob(os.path.join(user_downloads, "Sentinel-2*.zip")))
    elif args.input:
        if os.path.isdir(args.input):
            zip_files.extend(glob.glob(os.path.join(args.input, "*.zip")))
        elif os.path.isfile(args.input):
            zip_files.append(args.input)
    else:
        # Default: check user downloads and storage/input
        user_downloads = os.path.expanduser("~/Downloads")
        found_in_downloads = glob.glob(os.path.join(user_downloads, "*Sentinel*2*.zip"))
        found_in_input = glob.glob(os.path.join("storage", "input", "*.zip"))
        zip_files = found_in_downloads + found_in_input

    zip_files = list(set(zip_files))

    if not zip_files:
        print("[!] No Sentinel-2 zip files found.")
        print("Usage examples:")
        print("  python convert_sentinel2.py \"C:\\Users\\carlo\\Downloads\\Sentinel-2_mosaic_2025_Q3_30STG_0_0.zip\"")
        print("  python convert_sentinel2.py --downloads")
        sys.exit(1)

    print(f"[*] Found {len(zip_files)} Sentinel-2 archive(s) to process.\n")
    for zf in zip_files:
        try:
            out_file = None
            if args.output:
                if os.path.isdir(args.output):
                    out_file = os.path.join(args.output, f"{os.path.splitext(os.path.basename(zf))[0]}_{args.mode.upper()}.tif")
                else:
                    out_file = args.output
            convert_sentinel2_zip(zf, output_path=out_file, mode=args.mode, preset=args.preset)
        except Exception as e:
            print(f"[ERROR] Failed to convert {zf}: {e}\n")


if __name__ == "__main__":
    main()
