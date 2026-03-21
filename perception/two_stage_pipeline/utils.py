from PIL import Image
import os

def crop_box(img: Image.Image, xyxy):
    """Crop PIL image by xyxy = [x1,y1,x2,y2] and return cropped PIL image."""
    x1, y1, x2, y2 = map(int, xyxy)
    return img.crop((x1, y1, x2, y2))

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
