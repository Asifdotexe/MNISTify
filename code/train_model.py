import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
sns.set_style('whitegrid')
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

# Load and preprocess the MNIST dataset
def load_mnist_data():
    """
    Load and preprocess the MNIST dataset.
    
    :return: Preprocessed training and test data with labels.
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    return train_images, train_labels, test_images, test_labels

# Data augmentation
def augment_data(train_images):
    """
    Apply data augmentation to the training dataset using ImageDataGenerator.
    
    :param train_images: Training images.
    :return: Augmented data generator.
    """
    datagen = ImageDataGenerator(
        rotation_range=10,         # Rotate images by up to 10 degrees
        width_shift_range=0.1,     # Shift the image width by up to 10%
        height_shift_range=0.1,    # Shift the image height by up to 10%
        zoom_range=0.1,            # Zoom in/out on the image by up to 10%
        shear_range=0.1,           # Apply shearing transformations
        fill_mode='nearest'        # Fill in missing pixels after transformations
    )
    datagen.fit(train_images)
    return datagen

# Build a CNN model for digit recognition
def build_cnn_model():
    """
    Build a CNN model for digit recognition.
    
    :return: Compiled CNN model.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the CNN model and save it to disk
def train_and_save_model(model, train_images, train_labels, model_path):
    """
    Train the CNN model and save it to disk.
    
    :param model: CNN model to be trained.
    :param train_images: Preprocessed training images.
    :param train_labels: One-hot encoded training labels.
    :param model_path: Path to save the trained model.
    """
    history = model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)
    model.save(model_path)
    return history

# Evaluate the trained model on the test dataset
def evaluate_model(model, test_images: np.ndarray, test_labels: np.ndarray):
    """Evaluate the trained model on the test dataset.
    
    :param model: Trained Keras model.
    :param test_images: Preprocessed test images.
    :param test_labels: One-hot encoded test labels.
    :return: Test loss and accuracy.
    """
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    return test_loss, test_acc


# Plot training history for accuracy and loss
def plot_training_history(history, save_path: str):
    """Plot training history for accuracy and loss, and save the figure.
    
    :param history: History object containing training history.
    :param save_path: Path to save the training plot.
    """
    # Plot Accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    # Load and preprocess MNIST data
    train_images, train_labels, test_images, test_labels = load_mnist_data()

    # Build and train the model
    model = build_cnn_model()
    history = train_and_save_model(model, train_images, train_labels, "../saved_model/mnist_digit_recognizer_updated.keras")
    model.save("mnist_digit_recognizer_updated.keras")
    print("Model saved successfully")

    # Plot the training history
    plot_training_history(history, "../saved_model/training_history.png")

    # Evaluate the trained model on the test dataset
    test_loss, test_acc = evaluate_model(model, test_images, test_labels)

    # Print evaluation results
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_acc}")
