# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import os, io, csv, json, time, pandas as pd

# Optional matplotlib plotting
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# -----------------------
# Page config & style
# -----------------------
st.set_page_config(page_title="Betel Leaf Disease Detector.AI", layout="wide")

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
# Load model
# -----------------------
@st.cache_resource
def load_model_from_paths():
    possible_paths = [
        "streamlit_betel_leaf_app/models/Betel_Leaf_Model.keras",
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
def preprocess_image_enhanced(pil_img, target_size=(224,224)):
    """Preprocess image with enhancement for better model confidence"""
    pil_img = pil_img.convert("RGB")
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.resize(target_size, resample=Image.LANCZOS)
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.3)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(1.1)
    pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
    arr = np.array(pil_img).astype(np.float32)/255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def tta_predict(model, pil_img, target_size=(224,224), T=1.2):
    """Run TTA + temperature scaling for stable, high-confidence prediction"""
    tta_transforms = [
        lambda im: im,
        lambda im: ImageOps.mirror(im),
        lambda im: im.rotate(15, expand=False),
        lambda im: im.rotate(-15, expand=False),
        lambda im: ImageEnhance.Color(im).enhance(0.9),
        lambda im: ImageEnhance.Color(im).enhance(1.1),
        lambda im: ImageEnhance.Brightness(im).enhance(1.05),
        lambda im: ImageEnhance.Brightness(im).enhance(0.95),
    ]
    probs_list = []
    for tfm in tta_transforms:
        im2 = tfm(pil_img.copy())
        arr = preprocess_image_enhanced(im2, target_size)
        preds = model.predict(arr, verbose=0)
        probs_list.append(tf.nn.softmax(preds[0]).numpy())
    avg_probs = np.mean(np.stack(probs_list, axis=0), axis=0)
    # Temperature scaling
    logits = np.log(avg_probs + 1e-12)
    scaled_logits = logits / T
    scaled_probs = tf.nn.softmax(scaled_logits).numpy()
    idx = int(np.argmax(scaled_probs))
    return idx, scaled_probs

def model_summary_to_text(m):
    if m is None:
        return "No model loaded."
    buf = io.StringIO()
    m.summary(print_fn=lambda x: buf.write(x + "\n"))
    return buf.getvalue()

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
tabs = st.tabs(["Home","Predict","About Betel Leaf","About Us","Feedback"])

# -----------------------
# HOME
# -----------------------
with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-title">Betel Leaf Disease Detection.AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Visual diagnosis of betel leaf conditions.</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("### Dataset")
        st.markdown("~4000 labeled images across 4 classes.\n (train ~3200, val ~800)")
        st.write(", ".join(CLASS_NAMES))
        st.markdown("### Model info")
        if model:
            st.success(f"Model loaded from `{model_path}` ({file_size_human(model_path)})")
            with st.expander("Show model summary"):
                st.code(model_summary_to_text(model), language="text")
        else: st.error("Model not found.")
        st.markdown("### Sources")
        st.markdown("- Kaggle dataset\n- GitHub repo: https://github.com/Akash040917/streamlit_betel_leaf_app")
    with col2:
        st.markdown("### Quick Actions")
        st.markdown("- Use Predict tab to run inference.")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# PREDICT
# -----------------------
with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-title">Predict Betel Leaf Condition</div>', unsafe_allow_html=True)
    upload_col, cam_col = st.columns(2)
    # ---- Upload Image ----
    with upload_col:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Upload betel leaf image", type=["jpg","jpeg","png"])
        if uploaded_file and model:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded", use_column_width=True)
            with st.spinner("Predicting..."):
                idx, probs = tta_predict(model, img)
                st.success(f"Prediction: {CLASS_NAMES[idx]}")
                st.write(f"Confidence: {100*np.max(probs):.2f}%")
                df_probs = pd.DataFrame({"class":CLASS_NAMES,"probability":probs*100})
                df_probs = df_probs.sort_values("probability", ascending=False).reset_index(drop=True)
                st.table(df_probs.style.format({"probability":"{:.2f}%"}))
    # ---- Camera ----
    with cam_col:
        st.subheader("Use Camera")
        captured = st.camera_input("Take a photo")
        if captured and model:
            img = Image.open(captured)
            st.image(img, caption="Captured", use_column_width=True)
            with st.spinner("Predicting..."):
                idx, probs = tta_predict(model, img)
                st.success(f"Prediction: {CLASS_NAMES[idx]}")
                st.write(f"Confidence: {100*np.max(probs):.2f}%")
                df_probs = pd.DataFrame({"class":CLASS_NAMES,"probability":probs*100})
                df_probs = df_probs.sort_values("probability", ascending=False).reset_index(drop=True)
                st.table(df_probs.style.format({"probability":"{:.2f}%"}))
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# ABOUT BETEL LEAF
# -----------------------
with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-title">About Piper betle</div>', unsafe_allow_html=True)
    st.image("images/betel.jpg", caption="Piper betle", use_column_width=True)
    st.markdown("""
**Piper betle** is a perennial vine from the Piperaceae family. Cultivated across South & Southeast Asia.
Leaves are used in traditional medicine, culinary applications, and cultural rituals.
Key phytochemicals: hydroxychavicol, eugenol (antimicrobial, antioxidant).
""")
    st.markdown("### Varieties & types")
    st.markdown("- Green vs Red types\n- Regional cultivars, e.g., Banaras Pan (GI-protected in India)")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# ABOUT US
# -----------------------
with tabs[3]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-title">Team</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    members = [
        {"name":"Abdul Rawoof M","reg":"221201001","email":"221201001@rajalakshmi.edu.in","img":"images/member3.jpg"},
        {"name":"Akash Raghuram R L","reg":"221201004","email":"221201004@rajalakshmi.edu.in","img":"images/member1.jpg"},
        {"name":"Sarath Kumar R","reg":"221201048","email":"221201048@rajalakshmi.edu.in","img":"images/member2.jpg"},
    ]
    for c,m in zip(cols,members):
        with c:
            st.image(m["img"], use_column_width=True)
            st.markdown(f"**{m['name']}**\nRegistration: {m['reg']}\nEmail: {m['email']}")
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
