@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo     Sentinel-2 Mosaic Zip to GeoTIFF Converter
echo ===================================================
echo.

:: Check if an argument (drag & drop) was provided
if not "%~1"=="" (
    echo [*] Converting dragged file: "%~1"
    python convert_sentinel2.py "%~1"
    if %errorlevel% equ 0 (
        echo.
        echo [SUCCESS] GeoTIFF exported to storage\input\
    ) else (
        echo.
        echo [ERROR] Failed to convert mosaic.
    )
    pause
    exit /b 0
)

:: Interactive mode
echo Options:
echo   [1] Convert all Sentinel-2 zips in C:\Users\%USERNAME%\Downloads
echo   [2] Convert all Sentinel-2 zips in .\storage\input\
echo   [3] Convert specific file (manual input)
echo.
set /p CHOICE="Select an option (1/2/3) [default: 1]: "
if "%CHOICE%"=="" set CHOICE=1

if "%CHOICE%"=="1" (
    echo.
    echo [*] Scanning C:\Users\%USERNAME%\Downloads for Sentinel-2 mosaics...
    python convert_sentinel2.py --downloads
) else if "%CHOICE%"=="2" (
    echo.
    echo [*] Scanning storage\input for Sentinel-2 zips...
    python convert_sentinel2.py "storage\input"
) else (
    echo.
    set /p FILE_PATH="Enter full path to .zip file: "
    python convert_sentinel2.py "!FILE_PATH!"
)

echo.
echo ===================================================
echo Conversion complete! Check .\storage\input\ for GeoTIFFs.
echo ===================================================
pause
