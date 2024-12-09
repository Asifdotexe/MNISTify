from tensorflow.keras.models import load_model
import numpy as np


def load_trained_model(model_path: str):
    """Load the trained model from the specified path.
    
    :param model_path: Path to the trained model file.
    :return: Loaded Keras model.
    """
    return load_model(model_path)