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
    return tf.keras.models.load_model("streamlit_betel_leaf_app/models/Betel_Leaf_Model.keras")

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
st.set_page_config(page_title="Betel Leaf Disease Detection", page_icon="🌿", layout="wide")
tabs = st.tabs(["🏠 Home", "🔍 Predict", "🍃 About Leaf", "👨‍🎓 About Us", "✍ Feedback"])

# -------------------------------
# Home Tab
# -------------------------------
with tabs[0]:
    st.title("🌿 Betel Leaf Disease Detection AI")
    st.image("https://i.ibb.co/z8nKhFh/betel-leaf.jpg", caption="Betel Leaf", use_container_width=True)
    st.markdown("""
        Welcome to the **Betel Leaf Disease Detection App** 🌱  
        This app uses a **custom-trained AI model** to detect diseases in betel leaves.  

        👉 Go to the **Predict** tab to upload an image or use your webcam for real-time detection.  
    """)

# -------------------------------
# Predict Tab
# -------------------------------
with tabs[1]:
    st.header("🔍 Upload Image for Prediction")
    uploaded_file = st.file_uploader("Upload a betel leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        class_index = np.argmax(score)
        predicted_class = CLASS_NAMES[class_index]
        confidence = 100 * np.max(score)

        st.success(f"✅ Prediction: **{predicted_class}**")
        st.info(f"📊 Confidence: **{confidence:.2f}%**")

    st.divider()
    st.header("📷 Take a Photo with Camera")
    captured_img = st.camera_input("Take a picture of a betel leaf")

    if captured_img is not None:
        img = Image.open(captured_img).convert("RGB")
        st.image(img, caption="Captured Image", use_container_width=True)

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

        st.success(f"✅ Prediction: **{predicted_class}**")
        st.info(f"📊 Confidence: **{confidence:.2f}%**")

# -------------------------------
# About Leaf Tab
# -------------------------------
with tabs[2]:
    st.header("🍃 About Betel Leaf (Piper betle)")
    st.image("https://i.ibb.co/LJZ8K9R/betel-plant.jpg", caption="Betel Plant", use_container_width=True)
    st.markdown("""
    The **Betel leaf** is a vine from the Piperaceae family, widely known for its **medicinal and cultural significance**.  

    ### Common Diseases:
    - 🟢 **Anthracnose (Green)** → fungal disease causing dark lesions  
    - 🟢 **Bacterial Leaf Spot (Green)** → bacterial infection causing black spots  
    - 🟢 **Healthy Green** → disease-free green leaf  
    - 🔴 **Healthy Red** → naturally red variant of betel leaf  
    """)

    st.subheader("🔬 Model Classes:")
    st.write(CLASS_NAMES)

# -------------------------------
# About Us Tab
# -------------------------------
with tabs[3]:
    st.header("👨‍🎓 About Us")
    st.image("https://i.ibb.co/F0sQ2J3/team.jpg", caption="Our Team", use_container_width=True)
    st.markdown("""
    We are final-year students from  
    **Rajalakshmi Engineering College, Department of Mechatronics Engineering**,  
    working on innovative **AI & Machine Learning projects**. 🚀  

    This project was developed as part of our research on **AI in agriculture** 🌱.  
    """)

# -------------------------------
# Feedback Tab
# -------------------------------
with tabs[4]:
    st.header("✍ Leave Your Feedback")
    with st.form("feedback_form"):
        name = st.text_input("👤 Your Name")
        email = st.text_input("📧 Your Email")
        message = st.text_area("💬 Your Message / Feedback")
        
        submitted = st.form_submit_button("✅ Submit Feedback")
        
        if submitted:
            with open("feedback.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([name, email, message])
            st.success("🎉 Thank you for your feedback!")

    st.markdown("---")
    st.markdown("© 2025 **ProjectASA2025** | Built with ❤️ using Streamlit & TensorFlow")













