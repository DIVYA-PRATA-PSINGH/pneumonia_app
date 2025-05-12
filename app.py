import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the model
model = load_model('pneumonia_cnn.h5')

# App title and description
st.title('Pneumonia Detection App')
st.write('Upload a chest X-ray image to check for pneumonia.')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and show the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    image = image.resize((150, 150))  # 👈 match training input size
    image = image.convert('RGB')
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Predict
    prediction = model.predict(image)
    probability = prediction[0][0]
    
    # Show result
    if probability > 0.5:
        st.write('Prediction: **Pneumonia**')
    else:
        st.write('Prediction: **Normal**')
