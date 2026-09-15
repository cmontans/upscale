import numpy as np
from PIL import Image

# Create synthetic RGB satellite-like pattern
h, w = 512, 512
x = np.linspace(0, 10, w)
y = np.linspace(0, 10, h)
xx, yy = np.meshgrid(x, y)
r = np.clip(np.sin(xx) * 127 + 128, 0, 255).astype(np.uint8)
g = np.clip(np.cos(yy) * 127 + 128, 0, 255).astype(np.uint8)
b = np.clip((np.sin(xx + yy) * 127 + 128), 0, 255).astype(np.uint8)
rgb = np.stack([r, g, b], axis=-1)
im = Image.fromarray(rgb)
im.save('storage/input/sample_satellite.tif', format='TIFF')
print('Sample GeoTIFF raster created successfully at storage/input/sample_satellite.tif')
