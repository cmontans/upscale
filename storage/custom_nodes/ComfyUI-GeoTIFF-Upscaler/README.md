# ComfyUI-GeoTIFF-Upscaler 🌍🔍

A specialized ComfyUI node suite designed for loading, tile-processing, upscaling, and exporting **GeoTIFF** satellite imagery, aerial drone orthophotos, and digital elevation models (DEM) while preserving full geospatial metadata.

## Key Features

1. **🌍 Load GeoTIFF Raster**:
   - Supports 8-bit, 16-bit uint, 32-bit float, RGB, RGBA, and single-band rasters.
   - Extracts CRS (EPSG/WKT), Affine transform matrix, bounding box, NoData values, and band metadata.
   - Normalization modes: 0-1, Min-Max stretch, and 2%-98% percentile stretch.

2. **💾 Save GeoTIFF Raster**:
   - Automatically recalculates affine pixel resolution ($dx_{new} = dx_{orig} \times \frac{W_{orig}}{W_{new}}$) to ensure coordinates, bounding boxes, and projections match perfectly in GIS.
   - Supports **DEFLATE**, **LZW**, **ZSTD**, and **JPEG** compression with BigTIFF support.
   - Automatically builds internal pyramid overviews (factors 2, 4, 8, 16) for high-performance rendering in QGIS and ArcGIS.

3. **🔍 GeoTIFF Tiled Upscaler (Model)**:
   - Enables upscaling of massive gigapixel rasters without CUDA out-of-memory errors.
   - Smooth Cosine/Hann window and Linear feathering blends overlapping seams seamlessly.

4. **ℹ️ GeoTIFF Metadata Info**:
   - Displays real-time CRS, resolution, bounds, and band summary in the ComfyUI interface.
