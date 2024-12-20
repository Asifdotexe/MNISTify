import cv2
import numpy as np
from PIL import Image

def preprocess_image(image: Image) -> np.ndarray:
    """
    Preprocess the uploaded image for digit recognition.

    :param image: PIL Image object uploaded by the user.
    :return: Preprocessed image ready for model prediction.
    """
    # Convert to grayscale
    grayscale_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    
    # Threshold to remove noise (optional)
    _, binary_image = cv2.threshold(grayscale_image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours to crop the digit
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        cropped_image = binary_image[y:y+h, x:x+w]
    else:
        cropped_image = binary_image  # Use full image if no contour is found

    # Resize while keeping aspect ratio
    original_height, original_width = cropped_image.shape[:2]
    scale = 20.0 / max(original_height, original_width)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_image = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Pad to 28x28
    canvas = np.zeros((28, 28), dtype="uint8")
    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    # Normalize pixel values to [0, 1]
    normalized_image = canvas.astype('float32') / 255.0

    # Expand dimensions to match model input shape
    return np.expand_dims(normalized_image, axis=(0, -1))
