import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import csv
import matplotlib.pyplot as plt

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
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="Betel Leaf Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# -------------------------------
# Custom CSS for Professional Look
# -------------------------------
st.markdown("""
    <style>
        .main {
            background-color: #f9fafb;
        }
        .stTabs [role="tablist"] {
            justify-content: center;
        }
        h1, h2, h3 {
            color: #006400;
        }
        .footer {
            text-align: center;
            color: gray;
            font-size: 0.9em;
            padding: 20px 0;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Navbar using Tabs
# -------------------------------
tabs = st.tabs(["🏠 Home", "🔍 Predict", "🍃 About Leaf", "👨‍🎓 About Us", "✍ Feedback"])

# -------------------------------
# Home
# -------------------------------
with tabs[0]:
    col1, col2 = st.columns([1,1])
    with col1:
        st.title("🌿 Betel Leaf Disease Detection AI")
        st.markdown("""
            Welcome to the **Betel Leaf Disease Detection App** 🌱  

            This app uses a **custom-trained AI model** to identify diseases in betel leaves.  
            Upload an image or use your webcam for instant detection.  

            👉 Go to the **Predict** tab to try it out.
        """)
    with col2:
        st.image("https://i.ibb.co/z8nKhFh/betel-leaf.jpg", caption="Betel Leaf", use_container_width=True)

# -------------------------------
# Predict
# -------------------------------
with tabs[1]:
    st.header("🔍 Predict Betel Leaf Disease")

    col1, col2 = st.columns(2)

    # Upload
    with col1:
        uploaded_file = st.file_uploader("Upload a betel leaf image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Preprocess
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            class_index = np.argmax(score)
            predicted_class = CLASS_NAMES[class_index]
            confidence = 100 * np.max(score)

            st.success(f"✅ Prediction: **{predicted_class}**")
            st.info(f"📊 Confidence: **{confidence:.2f}%**")

            # Bar Chart for probabilities
            fig, ax = plt.subplots()
            ax.bar(CLASS_NAMES, score.numpy()*100)
            ax.set_ylabel("Confidence (%)")
            ax.set_title("Prediction Probabilities")
            st.pyplot(fig)

    # Camera
    with col2:
        captured_img = st.camera_input("📷 Or capture using camera")
        if captured_img:
            img = Image.open(captured_img).convert("RGB")
            st.image(img, caption="Captured Image", use_container_width=True)

            # Preprocess
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            pred = model.predict(img_array)
            score = tf.nn.softmax(pred[0])
            class_index = np.argmax(score)
            predicted_class = CLASS_NAMES[class_index]
            confidence = 100 * np.max(score)

            st.success(f"✅ Prediction: **{predicted_class}**")
            st.info(f"📊 Confidence: **{confidence:.2f}%**")

            # Bar Chart
            fig, ax = plt.subplots()
            ax.bar(CLASS_NAMES, score.numpy()*100)
            ax.set_ylabel("Confidence (%)")
            ax.set_title("Prediction Probabilities")
            st.pyplot(fig)

# -------------------------------
# About Leaf
# -------------------------------
with tabs[2]:
    st.header("🍃 About Betel Leaf (Piper betle)")
    col1, col2 = st.columns([1,2])
    with col1:
        st.image("https://i.ibb.co/LJZ8K9R/betel-plant.jpg", caption="Betel Plant", use_container_width=True)
    with col2:
        st.markdown("""
        The **Betel leaf** is a vine from the Piperaceae family, widely known for its  
        **medicinal, cultural, and traditional importance**.  

        ### Common Diseases:
        - 🟢 **Anthracnose (Green)** → fungal disease causing dark lesions  
        - 🟢 **Bacterial Leaf Spot (Green)** → bacterial infection causing black spots  
        - 🟢 **Healthy Green** → disease-free green leaf  
        - 🔴 **Healthy Red** → naturally red variant of betel leaf  
        """)

    st.subheader("🔬 Model Classes")
    st.write(CLASS_NAMES)

# -------------------------------
# About Us
# -------------------------------
with tabs[3]:
    st.header("👨‍🎓 About Us")
    col1, col2 = st.columns([1,2])
    with col1:
        st.image("https://i.ibb.co/F0sQ2J3/team.jpg", caption="Our Team", use_container_width=True)
    with col2:
        st.markdown("""
        We are final-year students from  
        **Rajalakshmi Engineering College, Dept. of Mechatronics Engineering** 🎓  

        Passionate about **AI & Machine Learning for agriculture** 🌱,  
        this project is part of our research on **AI in smart farming**. 🚀  
        """)

# -------------------------------
# Feedback
# -------------------------------
with tabs[4]:
    st.header("✍ Share Your Feedback")

    with st.form("feedback_form", clear_on_submit=True):
        name = st.text_input("👤 Your Name")
        email = st.text_input("📧 Your Email")
        message = st.text_area("💬 Your Feedback")

        submitted = st.form_submit_button("✅ Submit")

        if submitted:
            with open("feedback.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([name, email, message])
            st.success("🎉 Thank you for your valuable feedback!")

# -------------------------------
# Footer
# -------------------------------
st.markdown(
    "<div class='footer'>© 2025 ProjectASA2025 | Built with ❤️ using Streamlit & TensorFlow</div>",
    unsafe_allow_html=True
)

















