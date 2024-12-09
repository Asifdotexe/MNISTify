# MNISTify 🧙‍♂️📟  
**Accurate Handwritten Digit Recognition Using Deep Learning**

## Overview
Handwritten digit recognition is an essential task in machine learning, especially for real-world applications like form digitization and automated systems. This project uses the MNIST dataset to train a Convolutional Neural Network (CNN) that can accurately recognize digits. It also includes a user-friendly **Streamlit app** where users can upload images of handwritten digits for prediction.

---

## Features 🚀
- **Deep Learning Model**: Trained using TensorFlow and Keras with high accuracy.
- **User-Friendly App**: Built with Streamlit for easy image upload and prediction.
- **Real-Time Predictions**: Predicts the digit and provides confidence scores.
- **Interactive Visualizations**: Displays the uploaded image and highlights the model’s results.

---

## Problem Statement 🧐
Handwritten digit recognition is critical for digitizing data and automating processes like:
- Scanning bank forms and checks.
- Processing postal addresses.
- Automating classroom attendance and surveys.

This project aims to simplify handwritten digit recognition by providing an easy-to-use tool powered by a robust CNN model.

---

## Dataset 📊
- **Name**: MNIST Handwritten Digit Dataset
- **Size**: 60,000 training images, 10,000 testing images
- **Description**: Each image is a 28x28 grayscale representation of digits (0–9).
- **Source**: [MNIST Database](http://yann.lecun.com/exdb/mnist/)

---

## App Workflow 🔄
1. Upload an image of a handwritten digit (JPG/PNG format).
2. The image is preprocessed to match the MNIST dataset dimensions.
3. The trained CNN model predicts the digit and confidence scores.
4. The app displays the digit along with a visualization.

---

## Model Architecture 🧠
The CNN model is designed for high accuracy:
- **Input Layer**: (28, 28, 1) grayscale images.
- **Hidden Layers**: Convolutional, MaxPooling, Dense layers with ReLU activation.
- **Output Layer**: 10 nodes with softmax activation for digit classification.

---

## Installation 🛠️
To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Asifdotexe/MNISTify.git
   cd MNISTify
    ```

2. Create a virtual environment (recommended):
    ```bash
    conda create --name mnistify python=3.12
    conda activate mnistify
    ``` 
    or
    ```bash
    python3 -m venv mnistify
    venv\Scripts\activate
    ```

3. Download all the dependencies
    ```bash
    pip install -r requirements.txt
    ```

<hr>
Thank you!