import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('example.png', cv2.IMREAD_GRAYSCALE)

# Invert colors
inverted_image = cv2.bitwise_not(image)

# Display using matplotlib
plt.imshow(inverted_image, cmap='gray')
plt.title("Inverted Image")
plt.axis('off')
plt.show()
