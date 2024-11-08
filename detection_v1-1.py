import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from model import ANN, CNNModel

# Define character labels for prediction
characters = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Digits
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',  # Uppercase Letters
        'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'  # Selected lowercase letters
        ]

def load_model(model_path):
    """
    Load the trained model from a .pth file.
    """
    model = ANN()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode
    return model

import cv2
import numpy as np
import torch

import cv2
import numpy as np
import torch

def preprocess_roi(roi):
    """
    Preprocess the ROI with additional blur and fine-tuning of padding and thresholding to closely match EMNIST.
    """
    # Convert to grayscale if needed
    if len(roi.shape) == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian Blur to soften edges further
    roi = cv2.GaussianBlur(roi, (5, 5), 1)

    # Invert colors to ensure the character is white on a black background
    roi = cv2.bitwise_not(roi)

    # Apply binary thresholding with a slightly higher threshold for better contrast
    _, roi = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)

    # Find contours to determine the bounding box
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the bounding box for the largest contour
        x, y, w, h = cv2.boundingRect(contours[0])
        roi = roi[y:y+h, x:x+w]  # Crop the ROI to the bounding box

    # Resize to 20x20 while preserving aspect ratio, with high-quality resampling
    h, w = roi.shape
    if w > h:
        new_w = 20
        new_h = int((20 / w) * h)
    else:
        new_h = 20
        new_w = int((20 / h) * w)
    roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Center the character in a 28x28 frame with a 3-pixel border if needed
    top = (28 - new_h) // 2
    bottom = 28 - new_h - top
    left = (28 - new_w) // 2
    right = 28 - new_w - left
    roi = cv2.copyMakeBorder(roi, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    # Apply a final Gaussian Blur with a slightly larger kernel for a softer look
    roi = cv2.GaussianBlur(roi, (3, 3), 0.8)

    # Normalize to [0, 1] and add batch/channel dimensions for the model
    roi = torch.tensor(roi, dtype=torch.float32) / 255.0  # Normalize pixel values
    roi = roi.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions: (1, 1, 28, 28)
    
    return roi




def detect_and_predict_characters(image_path, model):
    """
    Detect and predict characters from an image.
    """
    # Load the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Resize, grayscale, and binarize the image
    image = cv2.resize(image, dsize=(width * 5, height * 4), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Invert the entire image to make the background black and characters white
    gray = cv2.bitwise_not(gray)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Apply dilation and GaussianBlur
    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    gsblur = cv2.GaussianBlur(img_dilation, (5, 5), 0)

    # Find contours and sort them
    ctrs, _ = cv2.findContours(gsblur.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # Prepare to store predictions
    predicted_characters = []
    dp = image.copy()  # Copy image for drawing

    for ctr in sorted_ctrs:
        x, y, w, h = cv2.boundingRect(ctr)
        roi = image[y-10:y+h+10, x-10:x+w+10]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if needed

        # Display the original ROI before preprocessing
        plt.figure()
        plt.imshow(roi, cmap='gray')
        plt.title("Original ROI (Unprocessed)")
        plt.axis('off')
        plt.show()

        # Preprocess the ROI to match EMNIST format
        roi = preprocess_roi(roi)

        # Display the preprocessed ROI
        plt.figure()
        plt.imshow(roi.squeeze(), cmap='gray')
        plt.title("Preprocessed ROI")
        plt.axis('off')
        plt.show()

        # Make prediction
        with torch.no_grad():
            output = model(roi)
            _, pred = torch.max(output, 1)
            predicted_character = characters[pred.item()]
            predicted_characters.append(predicted_character)

        # Draw bounding box and label on the image
        cv2.rectangle(dp, (x-10, y-10), (x + w + 10, y + h + 10), (90, 0, 255), 2)
        cv2.putText(dp, predicted_character, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 0, 255), 2)

    # Display the result with bounding boxes and predictions
    plt.imshow(dp)
    plt.title("Detected Characters with Predictions")
    plt.axis('off')
    plt.show()

    # Join predicted characters to form the final string
    predicted_string = ''.join(predicted_characters)

    # Print and return the predicted string
    print("Predicted String:", predicted_string)
    return predicted_string


# Example usage:
if __name__ == "__main__":
    # Specify the path to your trained model and input image
    model_path = "./saved_model/emnist/ANN/best_model_epoch_5.pth"  # Path to your saved model file
    image_path = "example2.png"  # Path to your input image
    
    # Load the model
    model = load_model(model_path)
    
    # Detect and predict characters
    predicted_text = detect_and_predict_characters(image_path, model)

