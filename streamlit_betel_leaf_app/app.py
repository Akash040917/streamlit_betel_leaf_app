st.write(st.secrets)
# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import os, io, csv, time, pandas as pd
import traceback
import gspread
from oauth2client.service_account import ServiceAccountCredentials

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
    """
    Tries all combinations of SEARCH_DIRS x MODEL_BASENAMES.
    Continues on load errors and returns first successfully loaded model.
    Returns (model_or_None, path_tried_or_None, error_message_or_None)
    """
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
        err_msg = "Tried these paths (files exist and some failed to load or were missing):\n"
        err_msg += attempted_str + "\n\nLoad errors:\n" + "\n".join(load_errors)
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

# ==== NEW: use EfficientNetV2 preprocessing exactly like Colab ====
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

if model is not None:
    IMG_H, IMG_W = model.input_shape[1], model.input_shape[2]
else:
    IMG_H, IMG_W = 300, 300  # fallback, should not be used normally

# -----------------------
# Helper functions 
# -----------------------
def preprocess_pil_image_advanced(pil_img, target_size=None):
    """
    Preprocess exactly like training/Colab:
    - RGB
    - resize to model input size
    - EfficientNetV2 preprocess_input
    """
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
    """
    Simple TTA: original + mirror + small rotations.
    Always uses the same preprocessing as training.
    """
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

        # if last layer is already softmax, sum ≈ 1
        if abs(np.sum(preds) - 1.0) < 0.05:
            probs = preds
        else:
            probs = tf.nn.softmax(preds).numpy()

        probs_list.append(probs)

    avg_probs = np.mean(np.stack(probs_list, axis=0), axis=0)
    return avg_probs

def predict_with_tta(model, pil_img, T=1.0):
    """
    Main prediction helper used by UI.
    Returns (index, probabilities).
    """
    avg_probs = tta_predictions(model, pil_img, target_size=(IMG_W, IMG_H))

    # Optional temperature scaling (T != 1.0 sharpens or smooths)
    if T is not None and T != 1.0:
        logits = np.log(avg_probs + 1e-12)
        logits = logits / T
        avg_probs = tf.nn.softmax(logits).numpy()

    idx = int(np.argmax(avg_probs))
    return idx, avg_probs

def file_size_human(path):
    try:
        s = os.path.getsize(path)
        for unit in ['B','KB','MB','GB']:
            if s < 1024.0:
                return f"{s:3.1f}{unit}"
            s /= 1024.0
        return f"{s:.1f}TB"
    except:
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
            if model_err:
                with st.expander("Model loader details (click to expand)"):
                    st.code(model_err)
                st.info(
                    "Make sure the model file is committed to the repository under one of these folders:\n"
                    + ", ".join(SEARCH_DIRS)
                )
            st.markdown(
                "Make sure `models/betel_leaf_efficientnetv2.keras` is present in the repo "
                "or enable Git LFS and re-add the model."
            )
        st.markdown("### Sources")
        st.markdown("- Kaggle dataset: https://www.kaggle.com/datasets/achmadbauravindah/betel-leaf-disease-classification")
        st.markdown("- GitHub repo: https://github.com/Akash040917/streamlit_betel_leaf_app")
        st.markdown("- Google Colab Repository: https://colab.research.google.com/drive/1N9yE22hXCalUVC_ir7nzaj9e7pgolTVu?usp=sharing")
    with col2:
        st.markdown("### Quick Actions")
        st.markdown("""
        - 🧠 **Run Disease Detection** → Go to the *Predict* tab and upload or capture an image.  
        - 🧬 **Learn About Betel Leaves** → Visit the *About Betel Leaf* tab for details and varieties.  
        - 👨‍💻 **Meet the Team** → Check out the *About Us* tab to know our developers.  
        - 💬 **Share Your Thoughts** → Use the *Feedback* tab to help us improve the app.
        """)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# PREDICT
# -----------------------
with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-title">Predict Betel Leaf Condition</div>', unsafe_allow_html=True)

# Disclaimer
    st.markdown("""
    <div style="
        background:#fff7e6;
        border-left:5px solid #f4a100;
        padding:10px;
        border-radius:6px;
        margin-bottom:12px;
        font-size:14px;">
⚠️ <b>Note:</b> This AI model is developed as part of an <b>academic research project</b>.  
Predictions are generated using the trained dataset and have been evaluated on sample real-world images, 
but the results are <b>research-oriented and not standardized for large-scale agricultural use</b>.  
Outputs are intended for <b>educational and research purposes</b>, with scope for further model enhancement.

    """, unsafe_allow_html=True)
    
    start_cam = st.checkbox("Start Camera")
    if start_cam:
        captured = st.camera_input("Take a photo")
        if captured:
            img = Image.open(captured)
            st.image(img, caption="Preview", width=300)
            if model:
                with st.spinner("Predicting..."):
                    idx, probs = predict_with_tta(model, img)
                    st.success(f"Prediction: {CLASS_NAMES[idx]}")
                    st.write(f"Confidence: {100*np.max(probs):.2f}%")
                    df_probs = pd.DataFrame({"class": CLASS_NAMES, "probability": probs*100})
                    st.table(df_probs.style.format({"probability": "{:.2f}%"}))
            else:
                st.warning("Model not available. See Home tab for details.")

    st.markdown("---")
    st.subheader("Or upload an image")
    uploaded_file = st.file_uploader("Upload betel leaf image", type=["jpg","jpeg","png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Preview", width=300)
        if model:
            with st.spinner("Predicting..."):
                idx, probs = predict_with_tta(model, img)
                st.success(f"Prediction: {CLASS_NAMES[idx]}")
                st.write(f"Confidence: {100*np.max(probs):.2f}%")
                df_probs = pd.DataFrame({"class": CLASS_NAMES, "probability": probs*100})
                st.table(df_probs.style.format({"probability": "{:.2f}%"}))
        else:
            st.warning("Model not available. See Home tab for details.")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# ABOUT BETEL LEAF
# -----------------------
with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-title">About Piper betle (Betel Leaf)</div>', unsafe_allow_html=True)
    st.image("streamlit_betel_leaf_app/images/betel.jpg", caption="Piper betle", width=450)
    st.markdown("""
    **Piper betle** is a perennial vine from the Piperaceae family, widely cultivated in South and Southeast Asia.  
    Heart-shaped leaves are used in traditional medicine, culinary applications, and cultural rituals.  
    Key phytochemicals include **hydroxychavicol** and **eugenol**, which exhibit antimicrobial and antioxidant properties.
    """)
    st.markdown("### Varieties & Classes 🌿")
    st.markdown("""
    **Betel leaves** are broadly categorized based on color and regional variety:
    - 🟢 **Green Varieties:** Common in South India; softer texture and mild aroma.  
    - 🔴 **Red Varieties:** Thicker leaves, stronger flavor, preferred for traditional uses.  
    - 📍 **Regional Cultivars:** Includes *Banarasi Pan*, *Kalkatta Pan*, and other GI-protected varieties of India.  
    - 🌱 Each type differs in taste, medicinal value, and oil content.
    """)
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
    ### 👥 Team Members

    **• Abdul Rawoof M**  
    *Deep Learning Model Development & Image Preprocessing*  
    Registration No: 221201001  
    Email: 221201001@rajalakshmi.edu.in  

    **• Akash Raghuram R L**  
    *Application Development & System Integration*  
    Registration No: 221201004  
    Email: 221201004@rajalakshmi.edu.in  

    **• Sarath Kumar R**  
    *Dataset Preparation, Testing & Performance Evaluation*  
    Registration No: 221201048  
    Email: 221201048@rajalakshmi.edu.in  
    """)

def save_to_gsheet(data):
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]

    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        st.secrets["gcp_service_account"], scope
    )

    client = gspread.authorize(creds)
    sheet = client.open("Feedback").sheet1

    sheet.append_row(data)
    
# -----------------------
# FEEDBACK
# -----------------------
with tabs[4]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-title">Feedback</div>', unsafe_allow_html=True)

    with st.form("feedback_form", clear_on_submit=True):
        fname = st.text_input("Full name")
        femail = st.text_input("Email")
        ftype = st.selectbox("Feedback type", ["Bug", "Feature", "Data", "Other"])
        rating = st.slider("Rate app (1-5)", 1, 5, 4)
        fmsg = st.text_area("Feedback")

        submit = st.form_submit_button("Submit")

        # ✅ MOVE INSIDE
        if submit:
            try:
                row = [
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    fname,
                    femail,
                    ftype,
                    rating,
                    fmsg,
                    model_path
                ]

                save_to_gsheet(row)
                st.success("Feedback saved successfully!")

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
