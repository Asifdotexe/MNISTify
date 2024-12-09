from tensorflow.keras.models import load_model
import numpy as np

def load_trained_model(model_path: str):
    """Load the trained model from the specified path.
    
    :param model_path: Path to the trained model file.
    :return: Loaded Keras model.
    """
    return load_model(model_path)

def predict_digit(image: np.ndarray, model) -> tuple[int, dict]:
    """Predict the digit in the given image using the trained model.
    
    :param image: Preprocessed image for prediction.
    :param model: Trained Keras model.
    :return: Predicted digit and confidence scores for each digit.
    """
    predictions = model.predict(image)
    predicted_digit = np.argmax(predictions)
    confidence_scores = {str(i): round(score * 100, 2) for i, score in enumerate(predictions[0])}
    return predicted_digit, confidence_scores
