import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your .keras model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("streamlit_betel_leaf_app/models/Betel_Leaf_Model.keras", compile=False)

model = load_model()

# Classes (update if different)
CLASS_NAMES = [
    "Anthracnose_Green",
    "BacterialLeafSpot_Green",
    "Healthy_Green",
    "Healthy_Red"
]

# Preprocess image according to model input shape
def preprocess_image(image: Image.Image):
    img_size = model.input_shape[1]  # dynamically get resolution
    img = image.convert("RGB").resize((img_size, img_size))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Prediction
def predict_image(image: Image.Image):
    processed = preprocess_image(image)
    preds = model.predict(processed)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    return CLASS_NAMES[class_idx], confidence, preds[0]

# Streamlit UI
st.title("🌿 Betel Leaf Disease Detection")
st.write("Upload an image or use your camera for live prediction.")

tab1, tab2 = st.tabs(["📁 Upload", "📷 Camera"])

with tab1:
    uploaded_file = st.file_uploader("Upload a betel leaf image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        label, conf, probs = predict_image(image)
        st.success(f"Prediction: **{label}** ({conf:.2%})")

with tab2:
    camera_file = st.camera_input("Take a photo")
    if camera_file:
        image = Image.open(camera_file)
        st.image(image, caption="Captured Image", use_container_width=True)
        label, conf, probs = predict_image(image)
        st.success(f"Prediction: **{label}** ({conf:.2%})")












