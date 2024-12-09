import streamlit as st
from PIL import Image
import numpy as np
import plotly.graph_objects as go
from preprocess import preprocess_image  # Ensure this function correctly processes the image
from predict import load_trained_model, predict_digit

def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("MNISTify \n By Asif Sayyed")
    st.success("Handwritten digit recognition is a fundamental problem in computer vision and has wide applications, such as form digitization, automated data entry, and postal code recognition. The goal of this project is to build a Convolutional Neural Network (CNN) model to accurately recognize handwritten digits using the MNIST dataset. The app provides a user-friendly interface for uploading handwritten digit images and displays the model's predictions in real time.")
    st.divider()
    st.markdown(
        "Upload an image of a handwritten digit, and the model will predict the digit along with confidence scores."
    )

    # Load the trained model
    try:
        model = load_trained_model(r"..\saved_model\mnist_digit_recognizer.keras")
    except ValueError:
        model = load_trained_model("/home/adminuser/mnistify/saved_model/mnist_digit_recognizer.keras")

    # File uploader to upload an image
    uploaded_file = st.file_uploader("Upload your image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Open the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Make a prediction
        predicted_digit, confidence_scores = predict_digit(preprocessed_image, model)

        # Display the predicted digit
        st.markdown(f"### Predicted Digit: **{predicted_digit}** 🎉")

        # Use expander to show the confidence score chart
        with st.expander("View Confidence Scores"):
            # Find the highest confidence score index
            highest_confidence_index = np.argmax(list(confidence_scores.values()))

            # Create a custom color list, highlighting the highest confidence score
            colors = ['gray' if i != highest_confidence_index else 'green' for i in range(10)]

            # Plotting the confidence distribution as a histogram using Plotly
            fig = go.Figure(data=[go.Bar(
                x=[str(i) for i in range(10)],  # Digits (0 to 9)
                y=list(confidence_scores.values()),  # Confidence scores
                marker_color=colors,  # Highlight the highest bar
            )])

            # Update layout for the chart
            fig.update_layout(
                title="Confidence Distribution for Each Digit",
                xaxis_title="Digits",
                yaxis_title="Confidence (%)",
                showlegend=False,
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

        # Optionally, display the raw confidence scores as text for reference
        # st.write("Raw Confidence Scores:", confidence_scores)

    else:
        st.info("Please upload an image of a handwritten digit!")
        
    st.divider()
    st.markdown("<h4 style='text-align: center;'>Connect with me on</h4>", unsafe_allow_html=True)
        
                 # Social Media Links Section
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center;">
        <a href="https://www.linkedin.com/in/sayyedasif/" target="_blank">
           <svg fill="#000000" height="40px" width="40px" version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 310 310" xml:space="preserve" stroke="#b2ffb2"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"> <g id="XMLID_801_"> <path id="XMLID_802_" d="M72.16,99.73H9.927c-2.762,0-5,2.239-5,5v199.928c0,2.762,2.238,5,5,5H72.16c2.762,0,5-2.238,5-5V104.73 C77.16,101.969,74.922,99.73,72.16,99.73z"></path> <path id="XMLID_803_" d="M41.066,0.341C18.422,0.341,0,18.743,0,41.362C0,63.991,18.422,82.4,41.066,82.4 c22.626,0,41.033-18.41,41.033-41.038C82.1,18.743,63.692,0.341,41.066,0.341z"></path> <path id="XMLID_804_" d="M230.454,94.761c-24.995,0-43.472,10.745-54.679,22.954V104.73c0-2.761-2.238-5-5-5h-59.599 c-2.762,0-5,2.239-5,5v199.928c0,2.762,2.238,5,5,5h62.097c2.762,0,5-2.238,5-5v-98.918c0-33.333,9.054-46.319,32.29-46.319 c25.306,0,27.317,20.818,27.317,48.034v97.204c0,2.762,2.238,5,5,5H305c2.762,0,5-2.238,5-5V194.995 C310,145.43,300.549,94.761,230.454,94.761z"></path> </g> </g></svg>
        </a> &nbsp; &nbsp; &nbsp;
        <a href="https://github.com/Asifdotexe" target="_blank">
            <svg width="40px" height="40px" viewBox="0 0 20 20" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" fill="#000000" stroke="#000000"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"> <title>github [#b2ffb2142]</title> <desc>Created with Sketch.</desc> <defs> </defs> <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="Dribbble-Light-Preview" transform="translate(-140.000000, -7559.000000)" fill="#000000"> <g id="icons" transform="translate(56.000000, 160.000000)"> <path d="M94,7399 C99.523,7399 104,7403.59 104,7409.253 C104,7413.782 101.138,7417.624 97.167,7418.981 C96.66,7419.082 96.48,7418.762 96.48,7418.489 C96.48,7418.151 96.492,7417.047 96.492,7415.675 C96.492,7414.719 96.172,7414.095 95.813,7413.777 C98.04,7413.523 100.38,7412.656 100.38,7408.718 C100.38,7407.598 99.992,7406.684 99.35,7405.966 C99.454,7405.707 99.797,7404.664 99.252,7403.252 C99.252,7403.252 98.414,7402.977 96.505,7404.303 C95.706,7404.076 94.85,7403.962 94,7403.958 C93.15,7403.962 92.295,7404.076 91.497,7404.303 C89.586,7402.977 88.746,7403.252 88.746,7403.252 C88.203,7404.664 88.546,7405.707 88.649,7405.966 C88.01,7406.684 87.619,7407.598 87.619,7408.718 C87.619,7412.646 89.954,7413.526 92.175,7413.785 C91.889,7414.041 91.63,7414.493 91.54,7415.156 C90.97,7415.418 89.522,7415.871 88.63,7414.304 C88.63,7414.304 88.101,7413.319 87.097,7413.247 C87.097,7413.247 86.122,7413.234 87.029,7413.87 C87.029,7413.87 87.684,7414.185 88.139,7415.37 C88.139,7415.37 88.726,7417.2 91.508,7416.58 C91.513,7417.437 91.522,7418.245 91.522,7418.489 C91.522,7418.76 91.338,7419.077 90.839,7418.982 C86.865,7417.627 84,7413.783 84,7409.253 C84,7403.59 88.478,7399 94,7399" id="github-[#b2ffb2142]"> </path> </g> </g> </g> </g></svg>
        </a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
