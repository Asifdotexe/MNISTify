import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from preprocess import preprocess_image
from predict import load_trained_model, predict_digit

def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("MNIST Digit Recognizer")
    st.markdown(
        "Upload a handwritten digit image, and the model will predict the digit along with confidence scores."
    )

    # Load the trained model
    model = load_trained_model("../saved_model/mnist_digit_recognizer.keras")

    # File uploader for user input
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # Preprocess the image
        image = Image.open(uploaded_file)
        preprocessed_image = preprocess_image(image)

        # Make a prediction
        predicted_digit, confidence_scores = predict_digit(preprocessed_image, model)

        # Display the predicted digit
        st.markdown(f"### Predicted Digit: **{predicted_digit}** 🎉")
        
        # Find the highest confidence score index
        highest_confidence_index = np.argmax(list(confidence_scores.values()))
        
        # Create a custom color list, highlighting the highest confidence score
        colors = ['gray' if i != highest_confidence_index else 'red' for i in range(10)]

        # Plotting the confidence distribution as a histogram using Plotly
        fig = go.Figure(data=[go.Bar(
            x=[str(i) for i in range(10)],  # Digits (0 to 9)
            y=list(confidence_scores.values()),  # Confidence scores
            marker_color=colors,  # Highlight the highest bar
        )])

        # Update layout for a white background
        fig.update_layout(
            title="Confidence Distribution for Each Digit",
            xaxis_title="Digits",
            yaxis_title="Confidence (%)",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background for aesthetics
            font=dict(color='black'),  # Set font color to black for better contrast
            xaxis=dict(
                tickangle=0,  # Horizontal tick labels for better readability
                showgrid=True,  # Show gridlines for reference
                gridcolor='lightgray',  # Light gridlines to differentiate between bars
            ),
            yaxis=dict(
                showgrid=True,  # Show gridlines for the y-axis
                gridcolor='lightgray',  # Light gridlines for better contrast
            )
        )

        # Show the interactive plot in Streamlit
        st.plotly_chart(fig)

        # # Optionally, display the raw confidence scores as text for reference
        # st.write("Raw Confidence Scores:", confidence_scores)

    else:
        st.info("Please upload an image to get started!")

if __name__ == "__main__":
    main()