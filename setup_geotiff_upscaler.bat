@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo   ComfyUI + GeoTIFF AI Upscaler Setup (Windows)
echo ===================================================
echo.

:: 1. Check if Docker is installed and running
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed or not in PATH.
    echo Please install Docker Desktop for Windows: https://www.docker.com/products/docker-desktop/
    pause
    exit /b 1
)

docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker daemon is not running.
    echo Please start Docker Desktop and ensure WSL2 integration is enabled.
    pause
    exit /b 1
)

echo [OK] Docker is running.
echo.

:: 2. Create Storage Directory Structure
echo [*] Initializing storage directories...
mkdir storage 2>nul
mkdir storage\custom_nodes 2>nul
mkdir storage\output 2>nul
mkdir storage\input 2>nul
mkdir storage\workflows 2>nul
mkdir storage\models\upscale_models 2>nul
mkdir storage\models\checkpoints 2>nul
mkdir storage\models\vae 2>nul

echo [OK] Directory tree initialized under .\storage\
echo.

:: 3. Optional Model Downloads
echo ===================================================
echo           Pre-trained Upscaler Models
echo ===================================================
echo Recommended models for satellite and aerial orthophotos:
echo   [1] RealESRGAN_x4plus (Realistic texture, sharp satellite details)
echo   [2] 4x-UltraSharp (General high-clarity upscaler)
echo   [3] Skip downloads (bring your own weights)
echo.

if not exist "storage\models\upscale_models\RealESRGAN_x4plus.pth" (
    set /p DL_REAL="Download RealESRGAN_x4plus.pth (~64MB)? (y/n): "
    if /i "!DL_REAL!"=="y" (
        echo [*] Downloading RealESRGAN_x4plus.pth...
        curl -L -o "storage\models\upscale_models\RealESRGAN_x4plus.pth" "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        if exist "storage\models\upscale_models\RealESRGAN_x4plus.pth" (
            echo [OK] RealESRGAN_x4plus.pth downloaded successfully.
        ) else (
            echo [WARN] Download failed. You can manually place weights in storage\models\upscale_models\
        )
    )
) else (
    echo [OK] RealESRGAN_x4plus.pth already present.
)

if not exist "storage\models\upscale_models\4x-UltraSharp.pth" (
    set /p DL_SHARP="Download 4x-UltraSharp.pth (~64MB)? (y/n): "
    if /i "!DL_SHARP!"=="y" (
        echo [*] Downloading 4x-UltraSharp.pth...
        curl -L -o "storage\models\upscale_models\4x-UltraSharp.pth" "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x-UltraSharp.pth"
        if exist "storage\models\upscale_models\4x-UltraSharp.pth" (
            echo [OK] 4x-UltraSharp.pth downloaded successfully.
        ) else (
            echo [WARN] Download failed. You can manually place weights in storage\models\upscale_models\
        )
    )
) else (
    echo [OK] 4x-UltraSharp.pth already present.
)
echo.

:: 4. Build and Run Container
echo ===================================================
echo              Docker Container Launch
echo ===================================================
set /p START_CONTAINER="Do you want to build and start the GeoTIFF ComfyUI container now? (y/n): "
if /i "%START_CONTAINER%"=="y" (
    echo.
    echo [*] Building and launching comfyui-geotiff-upscaler container...
    docker compose up --build -d
    if %errorlevel% equ 0 (
        echo.
        echo ===================================================
        echo [SUCCESS] ComfyUI GeoTIFF Upscaler is running!
        echo.
        echo 🌐 Open UI:          http://localhost:8188
        echo 📁 Put Input Tifs:   .\storage\input\
        echo 💾 Exported Tifs:    .\storage\output\
        echo 📋 Workflows:        .\storage\workflows\
        echo.
        echo 📜 View Live Logs:   docker logs -f comfyui-geotiff-upscaler
        echo ⏹️ Stop Container:   docker compose down
        echo ===================================================
    ) else (
        echo.
        echo [ERROR] Failed to start container. Check Docker Desktop settings and GPU driver.
    )
) else (
    echo.
    echo Setup complete. Run "docker compose up -d" whenever you are ready.
)

pause
