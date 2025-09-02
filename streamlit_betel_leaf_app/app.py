# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import os, io, csv, time, pandas as pd

# Optional matplotlib plotting
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# -----------------------
# Page config & style
# -----------------------
st.set_page_config(page_title="Betel Leaf Disease Detector", layout="wide")

PRIMARY = "#1f6f3a"
ACCENT = "#e9f5ec"
BG = "#f7fbf7"

st.markdown(f"""
<style>
:root {{
  --primary: {PRIMARY};
  --accent: {ACCENT};
  --bg: {BG};
}}
.stApp {{
  background-color: var(--bg);
  color: #0b3d14;
}}
.header-title {{
  color: var(--primary);
  font-size: 32px;
  font-weight: 700;
  margin-bottom: 0.15rem;
}}
.section-sub {{
  color: #0b3d14;
  margin-top: -8px;
  margin-bottom: 12px;
  font-size: 14px;
  opacity: 0.9;
}}
.card {{
  background: white;
  border-radius: 10px;
  padding: 18px;
  box-shadow: 0 2px 6px rgba(15, 30, 20, 0.06);
  margin-bottom: 16px;
}}
.muted {{
  color: #4b5d4b;
  font-size: 13px;
}}
footer {{
  display:none;
}}
</style>
""", unsafe_allow_html=True)

# -----------------------
# Model loading
# -----------------------
@st.cache_resource
def load_model_from_paths():
    possible_paths = [
        "streamlit_betel_leaf_app/models/Betel_Leaf_Model.keras",
        "models/Betel_Leaf_Model.keras",
        "Betel_Leaf_Model.keras"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            try:
                m = tf.keras.models.load_model(p)
                return m, p
            except Exception as e:
                st.warning(f"Found model at {p} but failed to load: {e}")
    return None, None

model, model_path = load_model_from_paths()
CLASS_NAMES = [
    "Anthracnose_Green",
    "BacterialLeafSpot_Green",
    "Healthy_Green",
    "Healthy_Red",
]

# -----------------------
# Helper functions
# -----------------------
def preprocess_pil_image_advanced(pil_img, target_size=(224,224)):
    pil_img = pil_img.convert("RGB")
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.resize(target_size, resample=Image.LANCZOS)
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.3)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(1.1)
    pil_img = ImageEnhance.Brightness(pil_img).enhance(1.05)
    pil_img = ImageOps.autocontrast(pil_img)
    arr = np.array(pil_img).astype(np.float32)/255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def tta_predictions(model, pil_img, tta_transforms=None, target_size=(224,224)):
    if tta_transforms is None:
        tta_transforms = [
            lambda im: im,
            lambda im: ImageOps.mirror(im),
            lambda im: im.rotate(15, expand=False),
            lambda im: im.rotate(-15, expand=False),
            lambda im: ImageEnhance.Color(im).enhance(0.9),
            lambda im: ImageEnhance.Color(im).enhance(1.1),
            lambda im: ImageEnhance.Brightness(im).enhance(1.1),
            lambda im: ImageEnhance.Brightness(im).enhance(0.9)
        ]
    probs_list = []
    for tfm in tta_transforms:
        im2 = tfm(pil_img.copy())
        arr = preprocess_pil_image_advanced(im2, target_size)
        preds = model.predict(arr, verbose=0)
        probs = tf.nn.softmax(preds[0]).numpy()
        probs_list.append(probs)
    avg_probs = np.mean(np.stack(probs_list, axis=0), axis=0)
    return avg_probs

def predict_with_tta(model, pil_img, T=0.8):
    avg_probs = tta_predictions(model, pil_img)
    logits = np.log(avg_probs + 1e-12)
    scaled_logits = logits / T
    scaled_probs = tf.nn.softmax(scaled_logits).numpy()
    idx = int(np.argmax(scaled_probs))
    return idx, scaled_probs

def file_size_human(path):
    try:
        s = os.path.getsize(path)
        for unit in ['B','KB','MB','GB']:
            if s < 1024.0: return f"{s:3.1f}{unit}"
            s /= 1024.0
        return f"{s:.1f}TB"
    except: return "Unknown"

# -----------------------
# Tabs
# -----------------------
tabs = st.tabs(["Home", "Predict", "About Betel Leaf", "About Us", "Feedback"])

# -----------------------
# HOME
# -----------------------
with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-title">Betel Leaf Disease Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Professional AI tool for detecting betel leaf diseases.</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("### Dataset & Model")
        st.markdown("Trained on ~4000 images across 4 classes.")
        st.write(", ".join(CLASS_NAMES))
        st.markdown("### Model info")
        if model is not None:
            st.success(f"Model loaded from `{model_path}` ({file_size_human(model_path)})")
        else: st.error("Model not found.")
        st.markdown("### Sources")
        st.markdown("- Kaggle dataset: https://www.kaggle.com/datasets/achmadbauravindah/betel-leaf-disease-classification")
        st.markdown("- GitHub repo: https://github.com/Akash040917/streamlit_betel_leaf_app")
    with col2:
        st.markdown("### Quick Actions")
        st.markdown("- Use Predict tab to run inference.\n- Update About Us with team info.")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# PREDICT
# -----------------------
with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-title">Predict Betel Leaf Condition</div>', unsafe_allow_html=True)
    start_cam = st.checkbox("Start Camera")
    if start_cam:
        captured = st.camera_input("Take a photo")
        if captured:
            img = Image.open(captured)
            st.image(img, caption="Captured Image", use_column_width=True)
            if model:
                with st.spinner("Predicting..."):
                    idx, probs = predict_with_tta(model, img)
                    st.success(f"Prediction: {CLASS_NAMES[idx]}")
                    st.write(f"Confidence: {100*np.max(probs):.2f}%")
                    df_probs = pd.DataFrame({"class":CLASS_NAMES,"probability":probs*100})
                    st.table(df_probs.style.format({"probability":"{:.2f}%"}))
    st.markdown("---")
    st.subheader("Or upload an image")
    uploaded_file = st.file_uploader("Upload betel leaf image", type=["jpg","jpeg","png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        if model:
            with st.spinner("Predicting..."):
                idx, probs = predict_with_tta(model, img)
                st.success(f"Prediction: {CLASS_NAMES[idx]}")
                st.write(f"Confidence: {100*np.max(probs):.2f}%")
                df_probs = pd.DataFrame({"class":CLASS_NAMES,"probability":probs*100})
                st.table(df_probs.style.format({"probability":"{:.2f}%"}))
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# ABOUT BETEL LEAF
# -----------------------
with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-title">About Piper betle (Betel Leaf)</div>', unsafe_allow_html=True)
    st.image("streamlit_betel_leaf_app/images/betel.jpg", caption="Piper betle", use_column_width=True)
    st.markdown("""
**Piper betle** is a perennial vine from the Piperaceae family, widely cultivated in South and Southeast Asia.
Heart-shaped leaves are used in traditional medicine, culinary applications, and cultural rituals.
Key phytochemicals include **hydroxychavicol** and **eugenol**, which exhibit antimicrobial and antioxidant properties.
""")
    st.markdown("### Varieties & Classes")
    st.markdown("- Green vs Red leaves\n- Regional cultivars (e.g., Banaras Pan, GI-protected in India)")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# ABOUT US
# -----------------------
with tabs[3]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-title">Our Team</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-sub">
    We are final-year Mechatronics students passionate about AI and Deep Learning.
    This project uses a Kaggle betel leaf dataset with <b>4 classes</b> and achieves ~85% validation accuracy.
    Our mission: provide a practical, easy-to-use AI tool for disease detection in betel leaves.
    </div>
    """, unsafe_allow_html=True)
    cols = st.columns(3)
    members = [
        {"name":"Abdul Rawoof M","reg":"221201001","email":"221201001@rajalakshmi.edu.in","img":"streamlit_betel_leaf_app/images/member3.jpg","role":"AI Model & Preprocessing"},
        {"name":"Akash Raghuram R L","reg":"221201004","email":"221201004@rajalakshmi.edu.in","img":"streamlit_betel_leaf_app/images/member1.jpg","role":"Frontend & Streamlit App"},
        {"name":"Sarath Kumar R","reg":"221201048","email":"221201048@rajalakshmi.edu.in","img":"streamlit_betel_leaf_app/images/member2.jpg","role":"Data Collection & Evaluation"},
    ]
    for c,m in zip(cols,members):
        with c:
            st.image(m["img"], width=220)
            st.markdown(f"**{m['name']}**\n*{m['role']}*\nRegistration: {m['reg']}\nEmail: {m['email']}", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# FEEDBACK
# -----------------------
with tabs[4]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-title">Feedback</div>', unsafe_allow_html=True)
    with st.form("feedback_form", clear_on_submit=True):
        fname = st.text_input("Full name")
        femail = st.text_input("Email")
        ftype = st.selectbox("Feedback type", ["Bug","Feature","Data","Other"])
        rating = st.slider("Rate app (1-5)", 1,5,4)
        fmsg = st.text_area("Feedback")
        submit = st.form_submit_button("Submit")
        if submit:
            try:
                with open("feedback.csv","a",newline="",encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), fname, femail, ftype, rating, fmsg, model_path])
                st.success("Feedback saved!")
            except Exception as e:
                st.error(f"Failed: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# Footer
# -----------------------
st.markdown(f"""
<div style="padding:20px 0; text-align:center; color:#4b5d4b;">
© {time.strftime('%Y')} ProjectASA2025 — Built with Streamlit & TensorFlow
</div>
""", unsafe_allow_html=True)




