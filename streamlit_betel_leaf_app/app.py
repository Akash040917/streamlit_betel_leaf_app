import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_camera_input import camera_input

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("streamlit_betel_leaf_app/models/Betel_Leaf_Model.h5")

model = load_model()

CLASS_NAMES = [
    "Anthracnose_Green",
    "BacterialLeafSpot_Green",
    "Healthy_Green",
    "Healthy_Red"
]

# -------------------------------
# App Layout
# -------------------------------
st.set_page_config(page_title="Betel Leaf Disease Detection", layout="wide")
st.title("Betel Leaf Disease Detection AI")

# -------------------------------
# File Upload Prediction
# -------------------------------
st.header("Upload Image for Prediction")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.markdown(f"### Prediction: **{predicted_class}**")
    st.markdown(f"### Confidence: **{confidence:.2f}%**")

# -------------------------------
# Camera Input Prediction
# -------------------------------
st.header("Live Camera Prediction")
captured_img = camera_input("Capture an image")
if captured_img:
    img = Image.open(captured_img).convert("RGB")
    st.image(img, caption="Captured Image", use_column_width=True)

    # Preprocess image
    img_array = np.expand_dims(np.array(img.resize((224, 224))) / 255.0, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.markdown(f"### Prediction: **{predicted_class}**")
    st.markdown(f"### Confidence: **{confidence:.2f}%**")





