import numpy as np
import matplotlib.pyplot as plt
from preprocessing import preprocess_image

def detect_connected_components(binary_img):
    h, w = binary_img.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []
    
    def bfs(x, y):
        queue = [(x, y)]
        visited[x, y] = True
        pixels = [(x, y)]
        min_x, min_y, max_x, max_y = x, y, x, y
        
        while queue:
            cx, cy = queue.pop(0)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < h and 0 <= ny < w and binary_img[nx, ny] == 1 and not visited[nx, ny]:
                    visited[nx, ny] = True
                    queue.append((nx, ny))
                    pixels.append((nx, ny))
                    min_x, min_y = min(min_x, nx), min(min_y, ny)
                    max_x, max_y = max(max_x, nx), max(max_y, ny)
        
        return min_x, min_y, max_x, max_y, pixels

    for i in range(h):
        for j in range(w):
            if binary_img[i, j] == 1 and not visited[i, j]:
                min_x, min_y, max_x, max_y, pixels = bfs(i, j)
                components.append((min_x, min_y, max_x, max_y))
                
    return components

def plot_detected_components(image_path):
    binary_img = preprocess_image(image_path)
    components = detect_connected_components(binary_img)
    
    plt.imshow(binary_img, cmap='gray')
    plt.axis("off")
    
    for (min_x, min_y, max_x, max_y) in components:
        rect = plt.Rectangle((min_y, min_x), max_y - min_y, max_x - min_x, edgecolor="red", linewidth=1, fill=False)
        plt.gca().add_patch(rect)
    
    plt.show()

# main.py

# Run the OCR detection pipeline
if __name__ == "__main__":
    image_path = "./Data/Pictures/OCR.jpg"
    plot_detected_components(image_path)
