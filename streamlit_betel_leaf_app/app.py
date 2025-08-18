import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------
# Load Model
# -----------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("streamlit_betel_leaf_app/models/Betel_Leaf_Model.h5")

model = load_model()

# Class names
CLASS_NAMES = ["Anthracnose_Green", "BacterialLeafSpot_Green", "Healthy_Green", "Healthy_Red"]

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="🍃 Betel Leaf Disease Detection",
    layout="wide"
)

# -----------------------
# Sidebar / Tabs
# -----------------------
tabs = ["Home", "Predict", "About Leaf", "About Us", "Contact Us"]
selected_tab = st.sidebar.radio("Navigate", tabs)

# -----------------------
# HOME TAB
# -----------------------
if selected_tab == "Home":
    st.title("🍃 Betel Leaf Disease Detection")
    st.image("images/banner.jpg", use_column_width=True)
    st.markdown("""
    **Detect the health status of Betel Leaves using a custom-trained ML model.**
    """)
    st.markdown("""
    - 🌟 Fast Prediction  
    - ✅ Accurate Model  
    - 📚 Research-backed
    """)

# -----------------------
# PREDICT TAB
# -----------------------
elif selected_tab == "Predict":
    st.title("🖼️ Upload Image for Prediction")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

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

        # Show result
        st.markdown(f"### ✅ Prediction: **{predicted_class}**")
        st.markdown(f"### 🔥 Confidence: **{confidence:.2f}%**")

    st.markdown("---")
    st.title("📹 Webcam Prediction (Optional)")
    st.write("Coming soon: Real-time webcam-based prediction.")

# -----------------------
# ABOUT LEAF TAB
# -----------------------
elif selected_tab == "About Leaf":
    st.title("🌿 About Betel Leaf (Piper betle)")
    st.image("images/betel_leaf.jpg", use_column_width=True)
    st.markdown("""
    **Botanical Info:** Piper betle is a vine from the family Piperaceae.  
    **Nutritional & Medicinal Benefits:** Used in traditional medicine, improves digestion, anti-inflammatory, etc.  
    **Common Diseases:** Anthracnose, Bacterial Leaf Spot, and other fungal/bacterial infections.
    """)
    st.markdown("**Classes used in ML Model:**")
    st.markdown("""
    - **Anthracnose_Green:** Green leaves affected by anthracnose  
    - **BacterialLeafSpot_Green:** Green leaves with bacterial spots  
    - **Healthy_Green:** Healthy green leaves  
    - **Healthy_Red:** Healthy red leaves
    """)
    st.markdown("---")
    st.markdown("**Dataset:**")
    st.markdown("""
    This project uses the **Betel Leaf Disease Classification** dataset from [Kaggle](https://www.kaggle.com/datasets/achmadbauravindah/betel-leaf-disease-classification).  
    We sincerely thank the dataset owner for providing this valuable resource.
    """)

# -----------------------
# ABOUT US TAB
# -----------------------
elif selected_tab == "About Us":
    st.title("👩‍💻 About Us")
    st.markdown("""
    We are students from **Rajalakshmi Engineering College, Mechatronics Department**,  
    interested in ML and model training. This project demonstrates a professional workflow including dataset preparation, model training, and deployment.
    """)

    st.markdown("### Team Members")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image("images/member1.jpg", width=150)
        st.markdown("**Alice Kumar**")
        st.markdown("alice@college.edu")
    with col2:
        st.image("images/member2.jpg", width=150)
        st.markdown("**Bob Raj**")
        st.markdown("bob@college.edu")
    with col3:
        st.image("images/member3.jpg", width=150)
        st.markdown("**Charlie Devi**")
        st.markdown("charlie@college.edu")

# -----------------------
# CONTACT US TAB
# -----------------------
elif selected_tab == "Contact Us":
    st.title("📬 Contact Us")
    st.markdown("""
    **Rajalakshmi Engineering College, Mechatronics Department**  
    Email: alice@college.edu, bob@college.edu, charlie@college.edu  
    LinkedIn: [College Page](https://www.linkedin.com)  
    """)

# -----------------------
# Footer / Copyright
# -----------------------
st.markdown("---")
st.markdown("© 2025 ProjectASA2025. All rights reserved.", unsafe_allow_html=True)




