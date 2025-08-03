import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from datetime import datetime
import matplotlib.pyplot as plt

from db import save_upload_metadata
from retrain_model import retrain_model

# Load model
@st.cache_resource
def load_model():
    current_dir = os.getcwd()
    st.write(f"📁 Current working directory: {current_dir}")
    st.write(os.listdir("../models"))
    return tf.keras.models.load_model("../models/fruit_model.keras")

model = load_model()
class_labels = [
    'freshapples', 'freshbanana', 'freshcucumber', 'freshokra',
    'freshoranges', 'freshpotato', 'freshtomato',
    'rottenapples', 'rottenbanana', 'rottencucumber', 'rottenokra',
    'rottenoranges', 'rottenpotato', 'rottentomato'
]

st.set_page_config(page_title="Fruit Quality App", layout="centered")
st.title("🍎 Fruit Quality Assessment")

# Create two tabs
tab1, tab2 = st.tabs(["🔍 Predict", "🔁 Retraining Upload"])

# ---------------------
# 🔍 Prediction Tab
# ---------------------
with tab1:
    uploaded_file = st.file_uploader("Upload an image for prediction", type=["jpg", "png", "jpeg"], key="predict_upload")

    if st.button("🔄 Reset"):
        st.session_state.clear()
        st.rerun()

    if st.button("🚀 Retrain Now"):
        with st.spinner("Retraining model..."):
            message = retrain_model()
            st.success(message)

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        pred_idx = np.argmax(predictions[0])
        st.success(f"Prediction: {class_labels[pred_idx]}")
        st.info(f"Confidence: {np.max(predictions[0]) * 100:.2f}%")

# ---------------------
# 🔁 Retraining Tab
# ---------------------
with tab2:
    st.header("🔁 Upload Images for Retraining")

    selected_class = st.selectbox("Select class for uploaded images", class_labels)
    uploaded_files = st.file_uploader(
        "Upload fruit images for retraining",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="retrain_upload"
    )

    if uploaded_files and selected_class:
        save_path = os.path.join("../data", "user_uploaded", selected_class)
        os.makedirs(save_path, exist_ok=True)

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            filename = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{uploaded_file.name}"
            filepath = os.path.join(save_path, filename)
            image.save(filepath)

            # Save metadata to MongoDB
            save_upload_metadata(filename, selected_class, filepath)

        st.success(f"✅ {len(uploaded_files)} image(s) saved under '{selected_class}' and logged to database.")
