import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import csv

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
tabs = st.tabs(["Home", "Predict", "About Leaf", "About Us", "Feedback"])

# -------------------------------
# Home Tab
# -------------------------------
with tabs[0]:
    st.title("🌿 Betel Leaf Disease Detection AI")
    st.markdown("""
        Welcome! This app detects the health status of Betel Leaves using a custom-trained ML model.  
        Go to the **Predict** tab to upload an image or use your webcam.
    """)

# -------------------------------
# Predict Tab
# -------------------------------
with tabs[1]:
    st.header("Upload Image for Prediction")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        class_index = np.argmax(score)
        predicted_class = CLASS_NAMES[class_index]
        confidence = 100 * np.max(score)

        st.markdown(f"### Prediction: **{predicted_class}**")
        st.markdown(f"### Confidence: **{confidence:.2f}%**")

    st.header("📷 Camera Capture")
    captured_img = st.camera_input("Take a picture")

    if captured_img is not None:
        img = Image.open(captured_img).convert("RGB")
        st.image(img, caption="Captured Image", use_column_width=True)

        # Preprocess
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        pred = model.predict(img_array)
        score = tf.nn.softmax(pred[0])
        class_index = np.argmax(score)
        predicted_class = CLASS_NAMES[class_index]
        confidence = 100 * np.max(score)

        st.markdown(f"### Prediction: **{predicted_class}**")
        st.markdown(f"### Confidence: **{confidence:.2f}%**")

# -------------------------------
# About Leaf Tab
# -------------------------------
with tabs[2]:
    st.header("About Betel Leaf (Piper betle)")
    st.markdown("""
    Betel leaf is a vine belonging to the Piperaceae family. 
    It is widely known for its medicinal properties and nutritional benefits.  

    **Common diseases affecting Betel leaves**:
    - Anthracnose (Green)
    - Bacterial Leaf Spot (Green)
    - Healthy Green
    - Healthy Red
    """)

    st.subheader("Classes used in model:")
    st.write(CLASS_NAMES)

# -------------------------------
# About Us Tab
# -------------------------------
with tabs[3]:
    st.header("About Us")
    st.markdown("""
    We are students from **Rajalakshmi Engineering College, Department of Mechatronics Engineering**,  
    working on Machine Learning and AI projects.
    """)

# -------------------------------
# Feedback Tab
# -------------------------------
with tabs[4]:
    st.header("Leave Your Feedback")
    with st.form("feedback_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Your Message / Feedback")
        
        submitted = st.form_submit_button("Submit Feedback")
        
        if submitted:
            with open("feedback.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([name, email, message])
            st.success("✅ Thank you for your feedback!")

    st.markdown("© 2025 ProjectASA2025")







