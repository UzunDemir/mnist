import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 28, 28, 1))
    return img_array
# Streamlit App
st.title('MNIST Digit Classifier')

uploaded_image = st.file_uploader("Upload a digit image (MNIST format)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img, caption='Uploaded Image (Resized)', use_column_width=True)

    with col2:
        st.write("")
        if st.button('Classify', key='classify_btn'):
            try:
                # Preprocess the uploaded image
                img_array = preprocess_image(uploaded_image)

                # Make a prediction using the pre-trained model
                result = model.predict(img_array)
                predicted_class = np.argmax(result)

                st.success(f'Predicted Digit: {predicted_class}')
            except Exception as e:
                st.error(f'Error: {e}')
