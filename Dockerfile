FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies including GDAL, PROJ, and graphics libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA 12.8 support (compatible with RTX 30/40/50-series Blackwell architecture)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Clone ComfyUI and install base requirements
RUN git clone https://github.com/comfyanonymous/ComfyUI.git .
RUN pip install --no-cache-dir -r requirements.txt

# Install Geospatial & Upscaling Python libraries
RUN pip install --no-cache-dir \
    rasterio \
    tifffile \
    scipy \
    pyproj \
    spandrel \
    opencv-python-headless \
    imageio \
    piexif \
    einops

EXPOSE 8188

CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]


