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

def preprocess_roi(roi):
    """
    Improved preprocessing for consistency in character centering, padding, and uniformity.
    """
    if len(roi.shape) == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    roi = cv2.GaussianBlur(roi, (5, 5), 1)
    roi = cv2.bitwise_not(roi)
    _, roi = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        roi = roi[y:y+h, x:x+w]

    h, w = roi.shape
    if w > h:
        new_w = 20
        new_h = int((20 / w) * h)
    else:
        new_h = 20
        new_w = int((20 / h) * w)
    roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    top = (28 - new_h) // 2
    bottom = 28 - new_h - top
    left = (28 - new_w) // 2
    right = 28 - new_w - left
    roi = cv2.copyMakeBorder(roi, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    roi = cv2.GaussianBlur(roi, (3, 3), 0.5)
    roi = torch.tensor(roi, dtype=torch.float32) / 255.0
    roi = roi.unsqueeze(0).unsqueeze(0)
    
    return roi

def segment_lines(image):
    profile = np.sum(image, axis=1)
    threshold = np.max(profile) * 0.1
    line_start = None
    lines = []

    for i, val in enumerate(profile):
        if val < threshold and line_start is not None:
            lines.append((line_start, i))
            line_start = None
        elif val >= threshold and line_start is None:
            line_start = i

    return lines

def segment_words(line_image):
    profile = np.sum(line_image, axis=0)
    threshold = np.max(profile) * 0.1
    word_start = None
    words = []

    for i, val in enumerate(profile):
        if val < threshold and word_start is not None:
            words.append((word_start, i))
            word_start = None
        elif val >= threshold and word_start is None:
            word_start = i

    return words

def segment_characters(word_image):
    _, thresh = cv2.threshold(word_image, 127, 255, cv2.THRESH_BINARY_INV)
    ctrs, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    character_bboxes = [cv2.boundingRect(ctr) for ctr in sorted_ctrs]
    return character_bboxes

def detect_and_predict_text(image_path, model):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.bitwise_not(gray)

    # Prepare to plot results
    plot_image = image.copy()

    # Segment lines and add blue boxes for each line
    lines = segment_lines(gray)
    predicted_text = []

    for line_start, line_end in lines:
        # Draw line bounding box in blue
        cv2.rectangle(plot_image, (0, line_start), (gray.shape[1], line_end), (255, 0, 0), 2)
        line_image = gray[line_start:line_end]
        words = segment_words(line_image)

        line_text = []
        for word_start, word_end in words:
            # Draw word bounding box in green
            cv2.rectangle(plot_image, (word_start, line_start), (word_end, line_end), (0, 255, 0), 2)
            word_image = line_image[:, word_start:word_end]

            # Process each character within the green word box
            character_bboxes = segment_characters(word_image)
            word_text = []
            for (x, y, w, h) in character_bboxes:
                # Preprocess character for prediction
                char_image = word_image[y:y+h, x:x+w]
                roi = preprocess_roi(char_image)

                # Make prediction
                with torch.no_grad():
                    output = model(roi)
                    _, pred = torch.max(output, 1)
                    pred_index = pred.item()

                    # Check if the prediction index is within bounds
                    if pred_index < len(characters):
                        predicted_character = characters[pred_index]
                    else:
                        predicted_character = "?"  # Placeholder for out-of-range predictions

                    word_text.append(predicted_character)

                # Draw character bounding box in red for visualization
                cv2.rectangle(plot_image, (word_start + x, line_start + y), (word_start + x + w, line_start + y + h), (0, 0, 255), 1)

            line_text.append(''.join(word_text))

        predicted_text.append(' '.join(line_text))

    final_text = '\n'.join(predicted_text)
    print("Predicted Text:\n", final_text)

    # Plot the image with boxes
    plt.figure(figsize=(12, 12))
    plt.imshow(cv2.cvtColor(plot_image, cv2.COLOR_BGR2RGB))
    plt.title("Detected Lines, Words, and Characters")
    plt.axis('off')
    plt.show()

    return final_text



# Usage example:
if __name__ == "__main__":
    model_path = "./saved_model/emnist/ANN/best_model_epoch_5.pth"
    image_path = "example4.png"
    
    model = load_model(model_path)
    detect_and_predict_text(image_path, model)
