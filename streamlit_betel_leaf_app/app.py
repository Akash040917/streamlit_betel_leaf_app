# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import os, io, csv, time, pandas as pd
import traceback

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

/* ✅ ADDED: remove empty white box issue */
.block-container > div:has(> div:empty) {{
    display: none;
}}
</style>
""", unsafe_allow_html=True)

# -----------------------
# Model loading (robust)
# -----------------------
MODEL_BASENAMES = [
    "betel_leaf_efficientnetv2.keras",
    "betel_leaf_model.keras",
    "mobilenetv2_final.keras",
    "efficientnet_model.keras",
]

SEARCH_DIRS = [
    os.path.join("streamlit_betel_leaf_app", "models"),
    "models",
    ".",
]

@st.cache_resource
def load_model_from_paths():
    attempted = []
    load_errors = []
    for d in SEARCH_DIRS:
        for base in MODEL_BASENAMES:
            p = os.path.join(d, base)
            if os.path.exists(p):
                attempted.append(p)
                try:
                    m = tf.keras.models.load_model(p, compile=False)
                    return m, p, None
                except Exception as e:
                    tb = traceback.format_exc()
                    load_errors.append(f"{p}: {str(e)}")
            else:
                attempted.append(p)

    attempted_str = "\n".join(attempted)
    if load_errors:
        err_msg = "Tried these paths:\n" + attempted_str + "\n\nLoad errors:\n" + "\n".join(load_errors)
    else:
        err_msg = "Model not found. Tried paths:\n" + attempted_str
    return None, None, err_msg

model, model_path, model_err = load_model_from_paths()

CLASS_NAMES = [
    "Anthracnose_Green",
    "BacterialLeafSpot_Green",
    "Healthy_Green",
    "Healthy_Red",
]

from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

if model is not None:
    IMG_H, IMG_W = model.input_shape[1], model.input_shape[2]
else:
    IMG_H, IMG_W = 300, 300

# -----------------------
# Helper functions
# -----------------------
def preprocess_pil_image_advanced(pil_img, target_size=None):
    if target_size is None:
        target_size = (IMG_W, IMG_H)

    pil_img = pil_img.convert("RGB")
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.resize(target_size, resample=Image.LANCZOS)

    arr = np.array(pil_img).astype(np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

def tta_predictions(model, pil_img, tta_transforms=None, target_size=None):
    if target_size is None:
        target_size = (IMG_W, IMG_H)

    if tta_transforms is None:
        tta_transforms = [
            lambda im: im,
            lambda im: ImageOps.mirror(im),
            lambda im: im.rotate(15, expand=False),
            lambda im: im.rotate(-15, expand=False),
        ]

    probs_list = []
    for tfm in tta_transforms:
        im2 = tfm(pil_img.copy())
        arr = preprocess_pil_image_advanced(im2, target_size)
        preds = model.predict(arr, verbose=0)[0]

        if abs(np.sum(preds) - 1.0) < 0.05:
            probs = preds
        else:
            probs = tf.nn.softmax(preds).numpy()

        probs_list.append(probs)

    avg_probs = np.mean(np.stack(probs_list, axis=0), axis=0)
    return avg_probs

def predict_with_tta(model, pil_img, T=1.0):
    avg_probs = tta_predictions(model, pil_img, target_size=(IMG_W, IMG_H))
    if T != 1.0:
        logits = np.log(avg_probs + 1e-12) / T
        avg_probs = tf.nn.softmax(logits).numpy()
    return int(np.argmax(avg_probs)), avg_probs

def file_size_human(path):
    try:
        s = os.path.getsize(path)
        for unit in ['B','KB','MB','GB']:
            if s < 1024:
                return f"{s:3.1f}{unit}"
            s /= 1024
    except:
        pass
    return "Unknown"

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
        st.markdown("Trained on ~10000 images across 4 classes.")
        st.write(", ".join(CLASS_NAMES))
        st.markdown("### Model info")
        if model is not None and model_path:
            st.success(f"Model loaded from `{model_path}` ({file_size_human(model_path)})")
        else:
            st.error("Model not found or failed to load.")
    with col2:
        st.markdown("### Quick Actions")
        st.markdown("""
        - 🧠 **Run Disease Detection**  
        - 🧬 **Learn About Betel Leaves**  
        - 👨‍💻 **Meet the Team**  
        - 💬 **Share Feedback**
        """)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# PREDICT
# -----------------------
with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-title">Predict Betel Leaf Condition</div>', unsafe_allow_html=True)

    # ✅ ADDED NOTE (ONLY ADDITION)
    st.markdown("""
    <div style="
        background:#fff7e6;
        border-left:5px solid #f4a100;
        padding:10px;
        border-radius:6px;
        margin-bottom:12px;
        font-size:14px;">
    ⚠️ <b>Note:</b> This AI model is developed for <b>testing and training purposes only</b>.
    </div>
    """, unsafe_allow_html=True)

    start_cam = st.checkbox("Start Camera")
    if start_cam:
        captured = st.camera_input("Take a photo")
        if captured:
            img = Image.open(captured)
            st.image(img, caption="Preview", width=300)
            if model:
                idx, probs = predict_with_tta(model, img)
                st.success(f"Prediction: {CLASS_NAMES[idx]}")

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload betel leaf image", type=["jpg","jpeg","png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Preview", width=300)
        if model:
            idx, probs = predict_with_tta(model, img)
            st.success(f"Prediction: {CLASS_NAMES[idx]}")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# ABOUT BETEL LEAF
# -----------------------
with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-title">About Piper betle (Betel Leaf)</div>', unsafe_allow_html=True)
    st.image("streamlit_betel_leaf_app/images/betel.jpg", caption="Piper betle", width=450)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# ABOUT US
# -----------------------
with tabs[3]:
    st.header("Our Team")
    st.write(
        "We are final-year Mechatronics Engineering students with a strong interest in "
        "Artificial Intelligence and Deep Learning.\n\n"
        "This project uses a Kaggle betel leaf image dataset with **4 disease classes** "
        "and achieves approximately **85% validation accuracy**.\n\n"
        "Our objective is to develop a practical and user-friendly AI-based system "
        "for automated betel leaf disease detection."
    )

    st.markdown("""
    **• Abdul Rawoof M**  
    *Deep Learning Model Development & Image Preprocessing*  
    Registration No: 221201001  

    **• Akash Raghuram R L**  
    *Application Development & System Integration*  
    Registration No: 221201004  

    **• Sarath Kumar R**  
    *Dataset Preparation, Testing & Performance Evaluation*  
    Registration No: 221201048  
    """)

# -----------------------
# FEEDBACK
# -----------------------
with tabs[4]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-title">Feedback</div>', unsafe_allow_html=True)
    with st.form("feedback_form", clear_on_submit=True):
        fname = st.text_input("Full name")
        femail = st.text_input("Email")
        fmsg = st.text_area("Feedback")
        submit = st.form_submit_button("Submit")
        if submit:
            with open("feedback.csv", "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"), fname, femail, fmsg])
            st.success("Feedback saved!")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# Footer
# -----------------------
st.markdown(f"""
<div style="padding:20px 0; text-align:center; color:#4b5d4b;">
© {time.strftime('%Y')} ProjectASA2025 — Built with Streamlit & TensorFlow
</div>
""", unsafe_allow_html=True)












