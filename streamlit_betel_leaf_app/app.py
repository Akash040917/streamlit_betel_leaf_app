import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model from local models folder
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("streamlit_betel_leaf_app/models/Betel_Leaf_Model.h5")

model = load_model()

# Define class names (customized)
CLASS_NAMES = [
    "Anthracnose_Green",       # Hijau_Antraknos
    "BacterialLeafSpot_Green", # Hijau_Karat
    "Healthy_Green",           # Hijau_Sehat
    "Healthy_Red"              # Merah_Sehat
]

# Streamlit App
st.set_page_config(page_title="Betel Leaf Disease Detection", layout="wide")
st.title("🍃 Betel Leaf Disease Detection")
st.write("Upload a Betel Leaf image to detect its health status.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize((224, 224))  # resize to model input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_index = np.argmax(score)
    predicted_class = CLASS_NAMES[class_index]
    confidence = 100 * np.max(score)

    # Show result
    st.markdown(f"### ✅ Prediction: **{predicted_class}**")
    st.markdown(f"### 🔥 Confidence: **{confidence:.2f}%**")






