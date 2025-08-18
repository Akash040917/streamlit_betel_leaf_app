import streamlit as st
from streamlit_camera_input import camera_input
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("streamlit_betel_leaf_app/models/Betel_Leaf_Model.h5")

model = load_model()
CLASS_NAMES = ["Anthracnose_Green", "BacterialLeafSpot_Green", "Healthy_Green", "Healthy_Red"]

# Helper functions
def preprocess_image(img: Image.Image) -> np.ndarray:
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img_array: np.ndarray):
    pred = model.predict(img_array)
    score = tf.nn.softmax(pred[0])
    class_index = np.argmax(score)
    confidence = 100 * np.max(score)
    predicted_class = CLASS_NAMES[class_index]
    return predicted_class, confidence

# Camera Input
st.header("Live Camera Prediction")
captured_img = camera_input("Capture an image")

if captured_img:
    image = Image.open(captured_img).convert("RGB")
    st.image(image, caption="Captured Image", use_column_width=True)

    img_array = preprocess_image(image)
    predicted_class, confidence = predict_image(img_array)

    st.markdown(f"### Prediction: **{predicted_class}**")
    st.markdown(f"### Confidence: **{confidence:.2f}%**")



