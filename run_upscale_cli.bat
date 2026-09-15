@echo off
setlocal enabledelayedexpansion

echo ======================================================================
echo           GeoTIFF & Sentinel-2 AI Upscaler (CLI Runner)
echo ======================================================================
echo.

if "%~1"=="" (
    echo Usage:
    echo   run_upscale_cli.bat path\to\raster.tif [output.tif]
    echo   run_upscale_cli.bat path\to\Sentinel-2_mosaic.zip [output.tif]
    echo   Or drag and drop any .tif / .zip file onto this batch file.
    echo.
    set /p INPUT_FILE="Enter path to GeoTIFF or Sentinel-2 Zip: "
) else (
    set INPUT_FILE=%~1
)

if not exist "!INPUT_FILE!" (
    echo [ERROR] File not found: !INPUT_FILE!
    pause
    exit /b 1
)

set OUTPUT_FILE=%~2
if "!OUTPUT_FILE!"=="" (
    set OUTPUT_FILE=!INPUT_FILE:.tif=_Upscaled_4x.tif!
    set OUTPUT_FILE=!OUTPUT_FILE:.zip=_Upscaled_4x.tif!
)

echo [1/2] Checking Docker container 'comfyui-geotiff-upscaler'...
docker ps --filter "name=comfyui-geotiff-upscaler" --format "{{.Names}}" | findstr "comfyui-geotiff-upscaler" >nul
if errorlevel 1 (
    echo Starting Docker container...
    docker-compose up -d
)

echo.
echo [2/2] Running GPU AI Upscaler on: !INPUT_FILE!
echo Destination: !OUTPUT_FILE!
echo.

REM Copy file into storage/input if not already in storage
copy /y "!INPUT_FILE!" "storage\input\" >nul 2>&1
for %%F in ("!INPUT_FILE!") do set FNAME=%%~nxF
for %%F in ("!OUTPUT_FILE!") do set OUT_FNAME=%%~nxF

docker exec -it comfyui-geotiff-upscaler python /app/custom_nodes/ComfyUI-GeoTIFF-Upscaler/cli_upscale.py --input "/app/input/!FNAME!" --output "/app/output/!OUT_FNAME!" --model RealESRGAN_x4plus.pth --tile-size 512

echo.
echo [COMPLETE] Upscaled GeoTIFF saved to: storage\output\!OUT_FNAME!
echo ======================================================================
pause
