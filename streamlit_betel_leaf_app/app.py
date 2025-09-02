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
BG = "#f7fbf7"

st.markdown(f"""
<style>
.stApp {{background-color:{BG}; color:#0b3d14;}}
.header-title {{color:{PRIMARY}; font-size:32px; font-weight:700; margin-bottom:0.15rem;}}
.section-sub {{color:#0b3d14; margin-top:-8px; margin-bottom:12px; font-size:14px; opacity:0.9;}}
.card {{background:white; border-radius:10px; padding:18px; box-shadow:0 2px 6px rgba(15,30,20,0.06); margin-bottom:16px;}}
footer {{display:none;}}
</style>
""", unsafe_allow_html=True)

# -----------------------
# Load model
# -----------------------
@st.cache_resource
def load_model():
    paths = ["streamlit_betel_leaf_app/models/Betel_Leaf_Model.keras","models/Betel_Leaf_Model.keras","Betel_Leaf_Model.keras"]
    for p in paths:
        if os.path.exists(p):
            try:
                return tf.keras.models.load_model(p), p
            except Exception as e:
                st.warning(f"Found model at {p} but failed: {e}")
    return None, None

model, model_path = load_model()
CLASS_NAMES = ["Anthracnose_Green","BacterialLeafSpot_Green","Healthy_Green","Healthy_Red"]

# -----------------------
# Helper functions
# -----------------------
def preprocess(img, size=(224,224)):
    img = img.convert("RGB")
    img = ImageOps.exif_transpose(img)
    img = img.resize(size, Image.LANCZOS)
    img = ImageEnhance.Sharpness(img).enhance(1.2)
    img = ImageEnhance.Contrast(img).enhance(1.05)
    arr = np.array(img).astype(np.float32)/255.0
    return np.expand_dims(arr, axis=0)

def tta_predict(model, img):
    transforms = [
        lambda x: x,
        lambda x: ImageOps.mirror(x),
        lambda x: x.rotate(15),
        lambda x: x.rotate(-15),
        lambda x: ImageEnhance.Color(x).enhance(0.9),
        lambda x: ImageEnhance.Color(x).enhance(1.1)
    ]
    probs_list = []
    for tfm in transforms:
        arr = preprocess(tfm(img))
        pred = model.predict(arr)
        probs_list.append(tf.nn.softmax(pred[0]).numpy())
    avg_probs = np.mean(np.array(probs_list), axis=0)
    return avg_probs

def temp_scale(probs, T=0.7):
    logits = np.log(probs+1e-12)
    scaled = logits/T
    return tf.nn.softmax(scaled).numpy()

# -----------------------
# Tabs
# -----------------------
tabs = st.tabs(["Home","Predict","About Betel Leaf","About Us","Feedback"])

# -----------------------
# HOME
# -----------------------
with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-title">Betel Leaf Disease Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">AI-powered detection of betel leaf diseases. Trained on 4 classes from Kaggle dataset.</div>', unsafe_allow_html=True)
    col1,col2 = st.columns([2,1])
    with col1:
        st.markdown("### Dataset & Classes")
        st.markdown("~4000 labeled images, 4 classes: "+", ".join(CLASS_NAMES))
        st.markdown("### Model Info")
        if model: st.success(f"Loaded `{model_path}`")
        else: st.error("Model not found!")
        st.markdown("### Sources")
        st.markdown("- Kaggle dataset: https://www.kaggle.com/datasets/achmadbauravindah/betel-leaf-disease-classification")
        st.markdown("- GitHub repo: https://github.com/Akash040917/streamlit_betel_leaf_app")
    with col2:
        st.markdown("### Quick Actions")
        st.markdown("- Use Predict tab to run inference.\n- About Us: Team & project info.")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# PREDICT
# -----------------------
with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-title">Predict Betel Leaf Condition</div>', unsafe_allow_html=True)
    upload_col, cam_col = st.columns(2)
    with upload_col:
        st.subheader("Upload Image")
        file = st.file_uploader("", type=["jpg","jpeg","png"])
        if file:
            img = Image.open(file)
            st.image(img, caption="Uploaded", use_column_width=True)
            if model:
                with st.spinner("Predicting..."):
                    probs = tta_predict(model,img)
                    probs = temp_scale(probs,T=0.7)
                    idx = int(np.argmax(probs))
                    st.success(f"Prediction: {CLASS_NAMES[idx]}")
                    st.write(f"Confidence: {100*np.max(probs):.2f}%")
                    df = pd.DataFrame({"class":CLASS_NAMES,"probability":probs*100}).sort_values("probability",ascending=False)
                    st.table(df.style.format({"probability":"{:.2f}%"}))
    with cam_col:
        st.subheader("Camera Input")
        captured = st.camera_input("Take a photo")
        if captured:
            img = Image.open(captured)
            st.image(img, caption="Captured", use_column_width=True)
            if model:
                with st.spinner("Predicting..."):
                    probs = tta_predict(model,img)
                    probs = temp_scale(probs,T=0.7)
                    idx = int(np.argmax(probs))
                    st.success(f"Prediction: {CLASS_NAMES[idx]}")
                    st.write(f"Confidence: {100*np.max(probs):.2f}%")
                    df = pd.DataFrame({"class":CLASS_NAMES,"probability":probs*100}).sort_values("probability",ascending=False)
                    st.table(df.style.format({"probability":"{:.2f}%"}))
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# ABOUT BETEL LEAF
# -----------------------
with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-title">About Piper betle</div>', unsafe_allow_html=True)
    st.markdown(
        """
        Piper betle is a perennial vine from Piperaceae, widely cultivated in South & Southeast Asia.
        Leaves are heart-shaped, used in traditional medicine, culinary & cultural rituals.
        Key phytochemicals: hydroxychavicol & eugenol (antimicrobial, antioxidant).
        """
    )
    st.image("streamlit_betel_leaf_app/images/betel.jpg", width=600)
    st.markdown("### Varieties")
    st.markdown("- Green vs Red\n- Regional cultivars (e.g., Banaras Pan)")
    st.markdown("### Sources")
    st.markdown("- Wikipedia: Piper betle\n- Wikimedia Commons\n- GitHub: https://github.com/Akash040917/streamlit_betel_leaf_app")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# ABOUT US
# -----------------------
with tabs[3]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-title">Our Team</div>', unsafe_allow_html=True)
    st.markdown(
        "<div class='section-sub'>Final-year Mechatronics students passionate about AI. Project trained on Kaggle dataset (4 classes, ~85% val accuracy).</div>",
        unsafe_allow_html=True
    )
    cols = st.columns([1,1,1])
    members = [
        {"name":"Abdul Rawoof M","reg":"221201001","email":"221201001@rajalakshmi.edu.in","img":"streamlit_betel_leaf_app/images/member3.jpg","role":"AI Model & Preprocessing"},
        {"name":"Akash Raghuram R L","reg":"221201004","email":"221201004@rajalakshmi.edu.in","img":"streamlit_betel_leaf_app/images/member1.jpg","role":"Frontend & Streamlit App"},
        {"name":"Sarath Kumar R","reg":"221201048","email":"221201048@rajalakshmi.edu.in","img":"streamlit_betel_leaf_app/images/member2.jpg","role":"Data Collection & Evaluation"}
    ]
    for c,m in zip(cols,members):
        with c:
            st.image(m["img"], width=220)
            st.markdown(f"**{m['name']}**\n*{m['role']}*\nReg: {m['reg']}\nEmail: {m['email']}", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# FEEDBACK
# -----------------------
with tabs[4]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-title">Feedback</div>', unsafe_allow_html=True)
    with st.form("feedback_form", clear_on_submit=True):
        fname = st.text_input("Full Name")
        femail = st.text_input("Email")
        ftype = st.selectbox("Type", ["Bug","Feature","Data","Other"])
        rating = st.slider("Rate (1-5)", 1,5,4)
        fmsg = st.text_area("Feedback")
        submit = st.form_submit_button("Submit")
        if submit:
            try:
                with open("feedback.csv","a",newline="",encoding="utf-8") as f:
                    csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"), fname, femail, ftype, rating, fmsg, model_path])
                st.success("Saved successfully!")
            except Exception as e:
                st.error(f"Failed: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# Footer
# -----------------------
st.markdown(f"<div style='text-align:center;color:#4b5d4b;padding:20px 0;'>&copy; {time.strftime('%Y')} ProjectASA2025 — Built with Streamlit & TensorFlow</div>", unsafe_allow_html=True)


