import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_camera_input import camera_input
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
    st.title(" Betel Leaf Disease Detection AI")
    st.image("images/banner.jpg", use_column_width=True)
    st.markdown("""
        Welcome! This app detects the health status of Betel Leaves using a custom-trained ML model.
        Navigate to the **Predict** tab to test your images in real-time or upload photos.
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

        # Resize & preprocess
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

    st.header("Live Camera Prediction")
    captured_img = camera_input("Capture an image")

    if captured_img is not None:
        img = Image.open(captured_img).convert("RGB")
        st.image(img, caption="Captured Image", use_column_width=True)

        # Resize & preprocess
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
    - Color variation: Healthy Green, Healthy Red  

    **Dataset used for training**: [Kaggle Betel Leaf Disease Classification](https://www.kaggle.com/datasets/achmadbauravindah/betel-leaf-disease-classification)  
    We sincerely thank the dataset owner for providing high-quality images for research.
    """)

    st.subheader("Classes used in model:")
    st.write(CLASS_NAMES)

# -------------------------------
# About Us Tab
# -------------------------------
with tabs[3]:
    st.header("About Us")
    st.markdown("""
    We are the students from **Rajalakshmi Engineering College, Department of Mechatronics Engineering**, who are interested in ML.
    We Learned how to Train the Model from scratch and deployed using streamlit app
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("images/member1.jpg", use_column_width=True)
        st.markdown("**Akash Raghuram R L**\n221201004@rajalakshmi.edu.in")
    with col2:
        st.image("images/member2.jpg", use_column_width=True)
        st.markdown("**Sarath Kumar R**\n221201048@rajalakshmi.edu.in")
    with col3:
        st.image("images/member3.jpg", use_column_width=True)
        st.markdown("**Abdul Rawoof M**\n221201001@rajalakshmi.edu.in")

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
            st.success("Thank you for your feedback! 🙏 We appreciate it.")

    st.markdown("© 2025 ProjectASA2025")

