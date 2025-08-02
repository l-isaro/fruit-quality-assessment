# predict_app.py

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/fruit_model.h5")
    return model

model = load_model()

# Define class labels (adjust based on your dataset)
class_labels = ['freshapples', 'freshbanana', 'freshcucumber', 'rottenapples', 'rottenbanana', 'rottencucumber']

# Page setup
st.title("🍎 Fruit Quality Classifier")
st.write("Upload a fruit image and I'll predict its quality.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dim

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Display result
    st.success(f"Predicted: **{predicted_class}**")
    st.info(f"Confidence: {confidence * 100:.2f}%")
