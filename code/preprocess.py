import cv2
import numpy as np
from PIL import Image

def preprocess_image(image: Image) -> np.ndarray:
    """
    Preprocess the uploaded image for digit recognition.
    
    :param image: PIL Image object uploaded by the user.
    :return: Preprocessed image ready for model prediction.
    """
    grayscale_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(grayscale_image, (28, 28))
    inverted_image = cv2.bitwise_not(resized_image)
    normalized_image = inverted_image.astype('float32') / 255.0
    return np.expand_dims(normalized_image, axis=(0, -1))
