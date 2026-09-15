# ComfyUI GeoTIFF AI Upscaler 🛰️🔍

A Dockerized ComfyUI environment tailored for **AI-powered upscaling of GeoTIFF geospatial raster imagery** (satellite photos, drone orthomosaics, multispectral imagery, and DEMs) while **strictly preserving CRS projections, affine geotransforms, and bounding box coordinates**.

---

## 🌟 Key Features

- **Geospatial Integrity**: Unlike standard image nodes that discard GIS metadata, our `ComfyUI-GeoTIFF-Upscaler` custom node recalculates affine matrices for upscaled pixel grids ($dx_{new} = dx \cdot \frac{W}{W_{new}}$), preserving exact geographic footprints and EPSG coordinate systems.
- **Multi-Band & Bit-Depth Support**: Reads and normalizes 8-bit, 16-bit uint, 32-bit float, RGB, RGBA (with alpha mask), and 1-band rasters (grayscale/DEM).
- **Gigapixel Tiled Upscaling**: Split massive orthomosaics into overlapping tiles with Hann/Cosine window feathering to eliminate seams and prevent GPU VRAM out-of-memory errors.
- **🛰️ Sentinel-2 Mosaic Zip Support**: Directly load, extract, and convert raw Copernicus/Sinergise Sentinel-2 mosaic archives (`Sentinel-2_mosaic_YYYY_QX_...zip`) containing individual B02, B03, B04, and B08 bands into radiometrically enhanced True Color RGB or False Color NIR GeoTIFFs.
- **GIS-Ready Export**: Saves compressed (Deflate/LZW/ZSTD), tiled BigTIFFs with internal pyramid overviews (2x, 4x, 8x, 16x) for fast rendering in **QGIS**, **ArcGIS**, and web map tile servers.
- **Modern Hardware Accelerated**: CUDA 12.8 + PyTorch inside Docker, fully compatible with NVIDIA RTX 30, 40, and 50-series GPUs.

---

## 🛰️ Sentinel-2 Mosaic Conversion

Sentinel-2 seasonal/quarterly mosaics downloaded from Copernicus Browser / Sentinel Hub come as zip files containing separate spectral bands (`B02.tif`, `B03.tif`, `B04.tif`, `B08.tif`). You have two ways to process them:

### Option A: Standalone Windows Drag & Drop / Batch Converter
- **Drag & Drop**: Drag any `Sentinel-2_mosaic_*.zip` file directly onto [`convert_sentinel2.bat`](file:///c:/Users/carlo/Documents/GitHub/ideogram/convert_sentinel2.bat).
- **Auto Batch**: Double-click [`convert_sentinel2.bat`](file:///c:/Users/carlo/Documents/GitHub/ideogram/convert_sentinel2.bat) to scan and convert all mosaics in `C:\Users\carlo\Downloads\` automatically into `storage/input/`.
- **CLI Usage**:
  ```bash
  python convert_sentinel2.py "C:\Users\carlo\Downloads\Sentinel-2_mosaic_2025_Q3_30STG_0_0.zip"
  # Or convert in NIR false-color:
  python convert_sentinel2.py "C:\Users\carlo\Downloads\Sentinel-2_mosaic_2025_Q3_30STG_0_0.zip" -m cir
  ```

### Option B: Native ComfyUI Node (`LoadSentinel2Zip`)
- Load [`storage/workflows/sentinel2_mosaic_upscale.json`](file:///c:/Users/carlo/Documents/GitHub/ideogram/storage/workflows/sentinel2_mosaic_upscale.json) directly in ComfyUI.
- Select your `.zip` archive from the dropdown.
- It will stream the bands in memory, apply atmospheric reflectance contrast stretching, and feed directly into the upscaling model!

---

## 🚀 Quick Start (Windows)

### 1. Launch with One Click
Double-click `setup_geotiff_upscaler.bat` or run:
```bat
setup_geotiff_upscaler.bat
```
The script will:
1. Verify Docker and NVIDIA GPU driver status.
2. Initialize the `storage/` directory structure.
3. Prompt to download pre-trained upscaling weights (`RealESRGAN_x4plus.pth`, `4x-UltraSharp.pth`).
4. Build and start the container in the background.

### 2. Manual Docker Launch
If you prefer running via CLI:
```bash
docker compose up --build -d
```
Then open your browser to **[http://localhost:8188](http://localhost:8188)**.

---

## 💻 Command Line Interface (CLI)

You can upscale GeoTIFFs and Sentinel-2 mosaics directly from the command line without opening a browser:

### Option 1: 1-Click Windows Batch Runner (`run_upscale_cli.bat`)
- **Drag & Drop**: Drag any `.tif` or Sentinel-2 `.zip` directly onto [`run_upscale_cli.bat`](file:///c:/Users/carlo/Documents/GitHub/ideogram/run_upscale_cli.bat).
- **Run in PowerShell / CMD**:
  ```cmd
  run_upscale_cli.bat "storage\input\Sentinel-2_mosaic_2026_Q2_30STG_0_0.0 (1).tif"
  run_upscale_cli.bat "storage\input\Sentinel-2_mosaic_2025_Q3_30STG_0_0.zip"
  ```

### Option 2: Direct Docker Execution (`docker exec`)
Run the streaming GPU upscaler directly inside the running Docker container:

#### Upscale a GeoTIFF Image (`.tif`):
```bash
docker exec -it comfyui-geotiff-upscaler python /app/custom_nodes/ComfyUI-GeoTIFF-Upscaler/cli_upscale.py \
  --input "/app/input/Sentinel-2_mosaic_2026_Q2_30STG_0_0.0 (1).tif" \
  --output "/app/output/Upscaled_2026_Q2.tif" \
  --model RealESRGAN_x4plus.pth \
  --tile-size 512
```

#### Upscale a Sentinel-2 Zip Archive (`.zip`):
```bash
docker exec -it comfyui-geotiff-upscaler python /app/custom_nodes/ComfyUI-GeoTIFF-Upscaler/cli_upscale.py \
  --input "/app/input/Sentinel-2_mosaic_2025_Q3_30STG_0_0.zip" \
  --output "/app/output/Sentinel2_Zip_Upscaled_4x.tif" \
  --model RealESRGAN_x4plus.pth \
  --tile-size 512
```

### ⚙️ CLI Parameter Reference

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--input`, `-i` | *(Required)* | Path to input GeoTIFF (`.tif`) or Sentinel-2 archive (`.zip`) |
| `--output`, `-o` | Auto | Output file path (defaults to `*_Upscaled_4x.tif`) |
| `--model`, `-m` | `RealESRGAN_x4plus.pth` | Model filename in `storage/models/upscale_models/` |
| `--tile-size`, `-t` | `512` | Window tile dimension in pixels ($256$ to $2048$) |
| `--overlap` | `32` | Boundary overlap padding in pixels |
| `--compression` | `deflate` | TIFF compression (`deflate`, `lzw`, `zstd`, `none`) |

---

## 📁 Directory Structure

```
ideogram/
├── Dockerfile                  # GDAL, PROJ, PyTorch CUDA 12.8 & ComfyUI
├── docker-compose.yml          # Container configuration & GPU pass-through
├── setup_geotiff_upscaler.bat  # Automated setup and model download script
├── storage/
│   ├── input/                  # 📥 Place your input .tif / .geotiff files here
│   ├── output/                 # 💾 Upscaled GeoTIFFs saved here
│   ├── workflows/              # 📋 Preconfigured ComfyUI workflow JSONs
│   │   ├── geotiff_model_upscale.json  # Standard 4x model upscaler
│   │   └── geotiff_tiled_upscale.json  # Gigapixel seamless tiled upscaler
│   ├── models/
│   │   └── upscale_models/     # 🧠 Place .pth / .safetensors upscaler models here
│   └── custom_nodes/
│       └── ComfyUI-GeoTIFF-Upscaler/ # Custom nodes with Rasterio & GDAL
```

---

## 📋 Ready-to-Use Workflows

In ComfyUI, click **Load** or drag-and-drop one of the workflow JSON files from `storage/workflows/`:

### 1. `geotiff_model_upscale.json` (Standard Upscale)
- **Use Case**: Medium-to-high resolution GeoTIFF images.
- **Flow**: `LoadGeoTIFF` ➔ `UpscaleModelLoader` ➔ `ImageUpscaleWithModel` ➔ `SaveGeoTIFF`.

### 2. `geotiff_tiled_upscale.json` (Gigapixel Tiled Upscale)
- **Use Case**: Massive drone orthophotos or large satellite scenes (10,000+ px).
- **Flow**: `LoadGeoTIFF` ➔ `GeoTIFFTileUpscaleWithModel` (Hann feathering) ➔ `SaveGeoTIFF`.

---

## 🎛️ Custom Nodes Reference

### `Load GeoTIFF Raster` (`LoadGeoTIFF`)
- **Inputs**: GeoTIFF filename (from `storage/input/`), channel mode (`Auto`, `RGB`, `RGBA`), 16-bit normalization mode (`Normalize to 0-1`, `Min-Max Stretch`, `Percentile 2%-98%`).
- **Outputs**:
  - `IMAGE`: Standard PyTorch tensor `[B, H, W, 3]`
  - `MASK`: Alpha / nodata channel `[B, H, W]`
  - `GEODATA`: Internal dictionary carrying CRS, Affine transform matrix, bounds, NoData, and raster tags.

### `Save GeoTIFF Raster` (`SaveGeoTIFF`)
- **Inputs**: `IMAGE`, `GEODATA`, `alpha_mask` (optional).
- **Parameters**:
  - `compression`: `DEFLATE`, `LZW`, `ZSTD`, `JPEG`, `PACKBITS`, `NONE`.
  - `predictor`: `2 (Horizontal / Integer)` for high compression ratios.
  - `tiled`: `true` (256x256 tiles for Cloud Optimized GeoTIFF performance).
  - `bigtiff`: `IF_SAFER` (supports files > 4 GB).
  - `generate_overviews`: `true` (builds pyramid levels for instant zoom in GIS).

### `GeoTIFF Tiled Upscaler (Model)` (`GeoTIFFTileUpscaleWithModel`)
- **Inputs**: `upscale_model`, `image`, `geodata`, `tile_size` (e.g., 1024), `overlap` (e.g., 64), `feather_mode` (`Cosine / Hann` or `Linear`).

### `GeoTIFF Metadata Info` (`GeoTIFFInfo`)
- Displays CRS projection, pixel size ($dx, dy$), bounding box coordinates, and band count in the ComfyUI UI.

---

## 🗺️ GIS Verification in QGIS

To verify the upscaled results:
1. Open **QGIS** or **ArcGIS Pro**.
2. Drag and drop your original GeoTIFF and the upscaled GeoTIFF from `storage/output/`.
3. Check **Layer Properties ➔ Information**:
   - **CRS / Projection**: Exactly identical to original.
   - **Extent / Bounding Box**: Exactly identical.
   - **Pixel Size**: Scaled down proportionally ($0.5\text{m} \rightarrow 0.125\text{m}$ for $4\times$ upscale).
   - **Alignment**: Both layers overlay with sub-pixel alignment!

---

## 🛠️ Recommended Upscaling Models

Place `.pth` or `.safetensors` models into `storage/models/upscale_models/`:
- **RealESRGAN_x4plus**: Excellent for satellite orthophotos, roads, buildings, and natural textures.
- **4x-UltraSharp**: General purpose high-contrast upscaling.
- **4x_NMKD-Superscale-SP_178000_G**: Clean upscaler with minimal hallucination.
- **4x-Nomos8k_DAT**: State-of-the-art transformer-based upscaling for aerial drone mapping.
