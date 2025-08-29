import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import csv

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("streamlit_betel_leaf_app/models/Betel_Leaf_Model.keras")

model = load_model()

CLASS_NAMES = [
    "Anthracnose (Green)",
    "Bacterial Leaf Spot (Green)",
    "Healthy Green",
    "Healthy Red"
]

# Set page config
st.set_page_config(page_title="Betel Leaf Disease Detector", layout="wide")

# Custom CSS for clean look
st.markdown("""
<style>
body {
    background-color: #f5f5f5;
    color: #333;
}
h1, h2, h3 {
    color: #006400;
}
.nav-link {
    font-size: 1.05em;
    color: #006400;
}
.footer {
    text-align: center;
    color: #666;
    font-size: 0.9em;
    padding: 20px 0;
}
.box {
    background-color: #fff;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Navigation tabs
tabs = st.tabs(["Home", "Predict", "About Betel Leaf", "About Us", "Feedback"])

with tabs[0]:
    st.header("Home")
    st.markdown("""
<div class="box">
### Dataset Overview
This application is built on a **custom dataset** of approximately **4,000 high-resolution betel leaf images** categorized into four classes:  
- **Anthracnose (Green)** – fungal disease  
- **Bacterial Leaf Spot (Green)** – bacterial infection  
- **Healthy Green** – disease-free green leaves  
- **Healthy Red** – naturally red variant leaves  
The dataset is sourced from a Kaggle collection specifically created for betel leaf disease classification. :contentReference[oaicite:2]{index=2}

### Model Performance
The AI model was trained using these classes to deliver accurate predictions. (If you have specific accuracy metrics—e.g., validation accuracy of 93.5%—you can insert them here.)

Navigate to the **Predict** tab to evaluate your images.
</div>
""", unsafe_allow_html=True)
    st.markdown("[Go to Predict →](#predict)")

with tabs[1]:
    st.header("Predict Betel Leaf Disease")

    st.markdown("<div class='box'><h3>Upload an Image</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a betel leaf image (jpg/png)...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        img = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, 0)
        preds = tf.nn.softmax(model.predict(img_array)[0])
        idx = int(np.argmax(preds))
        st.success(f"Prediction: **{CLASS_NAMES[idx]}**")
        st.info(f"Confidence: **{100 * np.max(preds):.2f}%**")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='box'><h3>Use Webcam</h3>", unsafe_allow_html=True)
    captured_img = st.camera_input("Capture a betel leaf image")
    if captured_img:
        img = Image.open(captured_img).convert("RGB")
        st.image(img, caption="Captured Image", use_column_width=True)
        img_resized = img.resize((224, 224))
        img_array = np.expand_dims(np.array(img_resized) / 255.0, 0)
        pred = tf.nn.softmax(model.predict(img_array)[0])
        idx = int(np.argmax(pred))
        st.success(f"Prediction: **{CLASS_NAMES[idx]}**")
        st.info(f"Confidence: **{100 * np.max(pred):.2f}%**")
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[2]:
    st.header("About Betel Leaf")
    st.markdown("""
<div class="box">
**Piper betle**, commonly called betel leaf, is a culturally and medicinally significant plant—often dubbed the “green gold” of South Asia. It is used in traditional medicine across Ayurveda and is revered for:

- **Antimicrobial, antioxidant, and wound-healing** properties :contentReference[oaicite:3]{index=3}  
- **Digestive benefits** and use in cultural rituals like poojas and Thamboolam :contentReference[oaicite:4]{index=4}  
- **Potential analgesic, cooling, antifungal, and anti-inflammatory** effects :contentReference[oaicite:5]{index=5}

Hydroxychavicol, a prominent bioactive compound in betel leaf, displays bactericidal and fungicidal activity, including inhibition of biofilm formation :contentReference[oaicite:6]{index=6}.

The leaf continues to be a symbol of cultural reverence and natural remedy in many traditions.
</div>
""", unsafe_allow_html=True)

with tabs[3]:
    st.header("About Us")
    st.markdown("<div class='box'>", unsafe_allow_html=True)
    cols = st.columns(3)
    names = ["Member A", "Member B", "Member C"]
    regs = ["RegNo1", "RegNo2", "RegNo3"]
    emails = ["a@example.com", "b@example.com", "c@example.com"]
    for col, name, reg, email in zip(cols, names, regs, emails):
        with col:
            st.image("https://via.placeholder.com/150", caption=name, use_column_width=True)
            st.markdown(f"**{name}**  \nRegistration No.: {reg}  \nEmail: {email}")
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[4]:
    st.header("Feedback")
    with st.form("feedback_form", clear_on_submit=True):
        name = st.text_input("Name")
        email = st.text_input("Email")
        message = st.text_area("Your feedback or suggestions")
        submitted = st.form_submit_button("Submit")
        if submitted:
            with open("feedback.csv", "a", newline="") as f:
                csv.writer(f).writerow([name, email, message])
            st.success("Thank you for your feedback and suggestions.")


















