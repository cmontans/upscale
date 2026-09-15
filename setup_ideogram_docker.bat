@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo   ComfyUI + Ideogram Local Docker Setup (Windows)
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

:: 2. Create ComfyUI Model and Storage Directories
echo [*] Creating directory structure...
mkdir storage 2>nul
mkdir storage\custom_nodes 2>nul
mkdir storage\output 2>nul
mkdir storage\input 2>nul
mkdir storage\models\diffusion_models 2>nul
mkdir storage\models\text_encoders 2>nul
mkdir storage\models\vae 2>nul
mkdir storage\models\checkpoints 2>nul
mkdir storage\models\clip 2>nul

echo [OK] Directories created under .\storage\
echo.

:: 3. Generate Dockerfile
echo [*] Generating Dockerfile...
(
echo FROM python:3.11-slim
echo.
echo ENV DEBIAN_FRONTEND=noninteractive
echo ENV PYTHONUNBUFFERED=1
echo.
echo RUN apt-get update ^&^& apt-get install -y --no-install-recommends \
echo     git \
echo     build-essential \
echo     libgl1 \
echo     libglib2.0-0 \
echo     curl \
echo     ^&^& rm -rf /var/lib/apt/lists/*
echo.
echo WORKDIR /app
echo.
echo RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
echo.
echo RUN git clone https://github.com/comfyanonymous/ComfyUI.git .
echo RUN pip install --no-cache-dir -r requirements.txt
echo.
echo EXPOSE 8188
echo.
echo CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
) > Dockerfile

:: 4. Generate docker-compose.yml
echo [*] Generating docker-compose.yml...
(
echo services:
echo   comfyui-ideogram:
echo     build: .
echo     container_name: comfyui-ideogram
echo     ports:
echo       - "8188:8188"
echo     environment:
echo       - NVIDIA_VISIBLE_DEVICES=all
echo     deploy:
echo       resources:
echo         reservations:
echo           devices:
echo             - driver: nvidia
echo               count: all
echo               capabilities: [gpu]
echo     volumes:
echo       - ./storage/models:/app/models
echo       - ./storage/custom_nodes:/app/custom_nodes
echo       - ./storage/output:/app/output
echo       - ./storage/input:/app/input
echo     restart: unless-stopped
) > docker-compose.yml

echo [OK] Configuration files generated.
echo.

:: 5. Remind user where to place weights
echo ===================================================
echo [!] Place your Ideogram weights in:
echo     - Diffusion:   .\storage\models\diffusion_models\
echo     - Text Encoders: .\storage\models\text_encoders\
echo     - VAE:         .\storage\models\vae\
echo ===================================================
echo.

:: 6. Build and Run Container
set /p START_CONTAINER="Do you want to build and start the container now? (y/n): "
if /i "%START_CONTAINER%"=="y" (
    echo.
    echo [*] Building and starting container...
    docker compose up --build -d
    if %errorlevel% equ 0 (
        echo.
        echo ===================================================
        echo [SUCCESS] ComfyUI is starting!
        echo Open your browser and navigate to: http://localhost:8188
        echo To view logs: docker logs -f comfyui-ideogram
        echo To stop:      docker compose down
        echo ===================================================
    ) else (
        echo.
        echo [ERROR] Failed to start Docker container. Check GPU driver settings in Docker Desktop.
    )
) else (
    echo.
    echo Setup complete. Run "docker compose up -d" whenever you are ready.
)

pause