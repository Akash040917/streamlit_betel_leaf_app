# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import io
import csv
import json
import time
import pandas as pd

# Optional plotting: try to import matplotlib, otherwise we'll fallback to st.bar_chart
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# -----------------------
# Page config & style
# -----------------------
st.set_page_config(page_title="Betel Leaf Disease Detector", layout="wide")

# Professional color theme & small UI polish
PRIMARY = "#1f6f3a"
ACCENT = "#e9f5ec"
BG = "#f7fbf7"

st.markdown(
    f"""
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
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Model loading
# -----------------------
@st.cache_resource
def load_model_from_paths():
    """
    Try a few common paths for the model file (.keras or .h5).
    Returns (model, model_path) or (None, None) on failure.
    """
    possible_paths = [
        "/mnt/data/Betel_Leaf_Model.keras",
        "/mnt/data/Betel_Leaf_Model.h5",
        "streamlit_betel_leaf_app/models/Betel_Leaf_Model.keras",
        "streamlit_betel_leaf_app/models/Betel_Leaf_Model.h5",
        "models/Betel_Leaf_Model.keras",
        "models/Betel_Leaf_Model.h5",
        "Betel_Leaf_Model.keras",
        "Betel_Leaf_Model.h5",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            try:
                m = tf.keras.models.load_model(p)
                return m, p
            except Exception as e:
                # try loading with custom objects or fallback handling (not implemented)
                st.warning(f"Found model at {p} but it failed to load: {e}")
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
def preprocess_pil_image(pil_img, target_size=(224, 224)):
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(target_size)
    arr = np.array(pil_img) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr

def predict_array(arr):
    if model is None:
        raise RuntimeError("Model not loaded. Place model file in repository or /mnt/data and restart.")
    preds = model.predict(arr)
    probs = tf.nn.softmax(preds[0]).numpy()
    idx = int(np.argmax(probs))
    return idx, probs

def model_summary_to_text(m):
    if m is None:
        return "No model loaded."
    try:
        buf = io.StringIO()
        m.summary(print_fn=lambda x: buf.write(x + "\n"))
        return buf.getvalue()
    except Exception as e:
        return f"Failed to generate model summary: {e}"

def file_size_human(path):
    try:
        s = os.path.getsize(path)
        for unit in ['B','KB','MB','GB']:
            if s < 1024.0:
                return f"{s:3.1f}{unit}"
            s /= 1024.0
        return f"{s:.1f}TB"
    except Exception:
        return "Unknown"

# -----------------------
# App layout - Tabs
# -----------------------
tabs = st.tabs(["Home", "Predict", "About Betel Leaf", "About Us", "Feedback"])

# -----------------------
# HOME
# -----------------------
with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-title">Betel Leaf Disease Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">A lightweight, production-ready UI for visual diagnosis of betel leaf conditions.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Dataset — what this app was trained on")
        st.markdown(
            """
            This application was developed using a dedicated betel leaf image dataset containing approximately **4,000** high-resolution images.
            The dataset contains four categories and was prepared for supervised classification (images were manually labeled and preprocessed).
            """
        )
        st.write("**Classes (what the model predicts):**")
        st.write(", ".join(CLASS_NAMES))
        st.markdown(
            "The original public dataset is available on Kaggle. If you have local dataset statistics (train/val/test split or per-class counts), place a `dataset_stats.json` file in the repository root and the app will attempt to show them."
        )

        st.markdown("### Model information")
        if model is not None:
            st.success(f"Model loaded from `{model_path}` (size: {file_size_human(model_path)})")
            with st.expander("Show model summary"):
                st.code(model_summary_to_text(model), language="text")
        else:
            st.error("Model not found. Place `Betel_Leaf_Model.keras` or `Betel_Leaf_Model.h5` in the repo root or /mnt/data and restart.")
            st.info("Paths checked: /mnt/data, models/, streamlit_betel_leaf_app/models/")

        st.markdown("### Model performance / metrics")
        st.markdown(
            """
            If you have training/validation metrics (for example `history.json` or `metrics.json` exported from training),
            place them in the repository root. The app will display validation accuracy and loss charts automatically.
            """
        )
        # try to load metrics.json if present
        if os.path.exists("metrics.json"):
            try:
                with open("metrics.json", "r") as f:
                    metrics = json.load(f)
                st.metric("Validation Accuracy", f"{metrics.get('val_accuracy', 'N/A')}")
                st.metric("Validation Loss", f"{metrics.get('val_loss', 'N/A')}")
                if "val_accuracy_history" in metrics:
                    df_hist = pd.DataFrame(metrics["val_accuracy_history"], columns=["val_accuracy"])
                    st.line_chart(df_hist["val_accuracy"])
            except Exception as e:
                st.warning(f"Could not read metrics.json: {e}")

    with col2:
        st.markdown("### Quick actions")
        st.markdown("- Use the Predict tab to run inference (Upload or Webcam).")
        st.markdown("- Replace placeholder images and member data in About Us with your project images and details.")
        st.markdown("---")
        st.markdown("### Links & dataset source")
        st.markdown(
            "- Kaggle dataset: https://www.kaggle.com/datasets/achmadbauravindah/betel-leaf-disease-classification\n"
            "- Research sources (see About tab for details)."
        )

    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------
# PREDICT
# -----------------------
with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-title">Predict Betel Leaf Condition</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upload an image or use your camera. Upload and Webcam are separate, independent flows.</div>', unsafe_allow_html=True)

    upload_col, cam_col = st.columns(2)

    # ---------- Upload flow ----------
    with upload_col:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Upload betel leaf image (jpg, png)", type=["jpg", "jpeg", "png"], key="uploader")
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded image", use_column_width=True)
                arr = preprocess_pil_image(image, target_size=(224,224))

                # Prediction
                if model is None:
                    st.error("Model not loaded. Cannot predict.")
                else:
                    with st.spinner("Running model inference..."):
                        idx, probs = predict_array(arr)
                        predicted_class = CLASS_NAMES[idx]
                        confidence = 100 * float(np.max(probs))
                        st.success(f"Prediction: {predicted_class}")
                        st.write(f"Confidence: {confidence:.2f}%")

                        # Probabilities table
                        df_probs = pd.DataFrame({
                            "class": CLASS_NAMES,
                            "probability": (probs * 100)
                        })
                        df_probs = df_probs.sort_values("probability", ascending=False).reset_index(drop=True)
                        st.table(df_probs.style.format({"probability":"{:.2f}%"}))

                        # Visualize probabilities
                        if HAS_MPL:
                            fig, ax = plt.subplots(figsize=(6,3))
                            bars = ax.barh(df_probs["class"], df_probs["probability"])
                            ax.invert_yaxis()
                            ax.set_xlabel("Probability (%)")
                            ax.set_xlim(0,100)
                            ax.set_title("Prediction probabilities")
                            for bar in bars:
                                w = bar.get_width()
                                ax.text(w + 1.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center')
                            st.pyplot(fig)
                        else:
                            st.bar_chart(df_probs.set_index("class")["probability"])

            except Exception as e:
                st.error(f"Failed to process image: {e}")

    # ---------- Webcam flow ----------
    with cam_col:
        st.subheader("Use Camera")
        captured = st.camera_input("Take a photo with your camera", key="camera_input")
        if captured is not None:
            try:
                img = Image.open(captured)
                st.image(img, caption="Captured image", use_column_width=True)
                arr = preprocess_pil_image(img, target_size=(224,224))

                if model is None:
                    st.error("Model not loaded. Cannot predict.")
                else:
                    with st.spinner("Running model inference..."):
                        idx, probs = predict_array(arr)
                        predicted_class = CLASS_NAMES[idx]
                        confidence = 100 * float(np.max(probs))
                        st.success(f"Prediction: {predicted_class}")
                        st.write(f"Confidence: {confidence:.2f}%")

                        df_probs = pd.DataFrame({
                            "class": CLASS_NAMES,
                            "probability": (probs * 100)
                        })
                        df_probs = df_probs.sort_values("probability", ascending=False).reset_index(drop=True)
                        st.table(df_probs.style.format({"probability":"{:.2f}%"}))

                        if HAS_MPL:
                            fig, ax = plt.subplots(figsize=(6,3))
                            bars = ax.barh(df_probs["class"], df_probs["probability"])
                            ax.invert_yaxis()
                            ax.set_xlabel("Probability (%)")
                            ax.set_xlim(0,100)
                            ax.set_title("Prediction probabilities")
                            for bar in bars:
                                w = bar.get_width()
                                ax.text(w + 1.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center')
                            st.pyplot(fig)
                        else:
                            st.bar_chart(df_probs.set_index("class")["probability"])

            except Exception as e:
                st.error(f"Failed to process captured image: {e}")

    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------
# ABOUT BETEL LEAF (researched content)
# -----------------------
with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-title">About Piper betle (Betel leaf)</div>', unsafe_allow_html=True)

    st.markdown(
        """
        **Piper betle** (betel leaf) is a perennial vine from the Piperaceae family, widely cultivated across South and Southeast Asia.
        The leaf has important cultural, culinary and medicinal applications. Scientific research highlights its rich phytochemistry
        (phenolics, essential oils) and multiple bioactivities including antimicrobial, antioxidant, anti-inflammatory and wound-healing properties.
        """
    )

    st.markdown("**Key highlights (selected research):**")
    st.markdown(
        "- Betel leaf is used traditionally as a stimulant, antiseptic, mouthwash and for digestive complaints. "
        "Modern reviews summarize its ethnopharmacological uses and pharmacology."
    )
    st.markdown(
        "- Several studies identify **hydroxychavicol**, eugenol and related phenolics as active constituents with antibacterial and antifungal activity."
    )
    st.markdown(
        "- Betel leaf extracts have been investigated for antioxidant, anti-inflammatory and even antiproliferative effects in preclinical studies."
    )

    st.markdown("---")
    st.markdown("**Sources & further reading**")
    st.markdown(
        "- Dataset used for training (Kaggle): https://www.kaggle.com/datasets/achmadbauravindah/betel-leaf-disease-classification\n"
        "- Review (comprehensive phytochemistry and uses): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9170825/\n"
        "- Review of antibacterial/antifungal evidence: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8073370/\n"
        "- Hydroxychavicol and antibacterial action: https://www.sciencedirect.com/science/article/abs/pii/S0891584918301242\n"
    )

    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------
# ABOUT US (3 members)
# -----------------------
with tabs[3]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-title">Team</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Project team — provide names, registration numbers and contact emails below.</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    # Replace these placeholders with real file paths or URLs for member images.
    default_member_images = [
        "https://via.placeholder.com/600x400.png?text=Member+1",
        "https://via.placeholder.com/600x400.png?text=Member+2",
        "https://via.placeholder.com/600x400.png?text=Member+3",
    ]
    # Placeholder data (please replace)
    members = [
        {"name": "Member 1", "reg": "REGNO1", "email": "member1@example.com", "img": default_member_images[0]},
        {"name": "Member 2", "reg": "REGNO2", "email": "member2@example.com", "img": default_member_images[1]},
        {"name": "Member 3", "reg": "REGNO3", "email": "member3@example.com", "img": default_member_images[2]},
    ]

    for c, m in zip(cols, members):
        with c:
            st.image(m["img"], use_column_width=True)
            st.markdown(f"**{m['name']}**  \nRegistration: {m['reg']}  \nEmail: {m['email']}")

    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------
# FEEDBACK
# -----------------------
with tabs[4]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-title">Feedback</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">We appreciate feedback and suggestions to improve the model and UI.</div>', unsafe_allow_html=True)

    with st.form("feedback_form", clear_on_submit=True):
        fname = st.text_input("Full name")
        femail = st.text_input("Email")
        fmsg = st.text_area("Feedback and suggestions")
        submit = st.form_submit_button("Submit")

        if submit:
            try:
                os.makedirs(".", exist_ok=True)
                with open("feedback.csv", "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), fname, femail, fmsg])
                st.success("Thank you for your feedback and suggestions.")
            except Exception as e:
                st.error(f"Failed to save feedback: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(
    f"""
    <div style="padding:20px 0; text-align:center; color:#4b5d4b;">
        © {time.strftime('%Y')} ProjectASA2025 — Built with Streamlit & TensorFlow
    </div>
    """,
    unsafe_allow_html=True,
)



















