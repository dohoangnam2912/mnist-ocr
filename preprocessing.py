import numpy as np
from PIL import Image

def preprocess_image(image_path, threshold=128):
    # Load image and convert to grayscale
    img = Image.open(image_path).convert("L")  # "L" mode is for grayscale
    img_np = np.array(img)
    
    # Apply binary threshold
    binary_img = (img_np > threshold).astype(int)  # Binarize: 1 for text, 0 for background
    return binary_img
