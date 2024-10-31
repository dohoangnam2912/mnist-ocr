import cv2
import numpy as np

def detect_text_with_east(image_path, east_model_path, min_confidence=0.5, width=320, height=320):
    # Load the image
    image = cv2.imread(image_path)
    orig = image.copy()
    (H, W) = image.shape[:2]

    # Calculate the ratio of original dimensions to the new dimensions
    rW = W / float(width)
    rH = H / float(height)

    # Resize the image to the specified dimensions
    image = cv2.resize(image, (width, height))
    (H, W) = image.shape[:2]

    # Load the pretrained EAST text detector
    net = cv2.dnn.readNet(east_model_path)

    # Construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)

    # Define the layer names for output layers
    layer_names = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ]

    # Forward pass to get the score and geometry map
    (scores, geometry) = net.forward(layer_names)

    # Decode the predictions
    rectangles = []
    confidences = []

    # Loop over rows and columns to extract the predictions
    for y in range(0, scores.shape[2]):
        for x in range(0, scores.shape[3]):
            # Extract the score (probability of text presence)
            score = scores[0, 0, y, x]
            if score < min_confidence:
                continue

            # Extract the geometry data for the bounding box
            offsetX, offsetY = (x * 4.0, y * 4.0)
            angle = geometry[0, 4, y, x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = geometry[0, 0, y, x] + geometry[0, 2, y, x]
            w = geometry[0, 1, y, x] + geometry[0, 3, y, x]

            endX = int(offsetX + (cos * geometry[0, 1, y, x]) + (sin * geometry[0, 2, y, x]))
            endY = int(offsetY - (sin * geometry[0, 1, y, x]) + (cos * geometry[0, 2, y, x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rectangles.append((startX, startY, endX, endY))
            confidences.append(float(score))

    # Apply non-maxima suppression to suppress weak, overlapping boxes
    boxes = cv2.dnn.NMSBoxes(rectangles, confidences, min_confidence, nms_threshold=0.4)

    # Check if boxes is None or empty
    if boxes is None or len(boxes) == 0:
        print("No text detected")
        return

    # Draw the detected text bounding boxes on the original image
    for i in boxes:
        (startX, startY, endX, endY) = rectangles[i[0]]
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # Draw the bounding box on the image
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Display the output image with text detection
    cv2.imshow("Text Detection", orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Usage example
detect_text_with_east("./Data/Pictures/1.png", "./outsource/frozen_east_text_detection.pb")
