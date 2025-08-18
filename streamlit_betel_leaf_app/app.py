import os
import numpy as np
import streamlit as st
from PIL import Image
import urllib.request

# Optional: use gdown for Google Drive URLs
def _download(url: str, dest_path: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if "drive.google.com" in url:
        try:
            import gdown  # type: ignore
            gdown.download(url, dest_path, fuzzy=True)
            return
        except Exception as e:
            st.error(f"gdown download failed: {e}")
            st.stop()
    # Fallback to urllib
    urllib.request.urlretrieve(url, dest_path)

MODEL_PATH = "models/Betel_Leaf_Model.h5"  # or models/model.keras

@st.cache_resource
def load_model_cached():
    import tensorflow as tf
    # If model file isn't present, try to download via secret
    if not os.path.exists(MODEL_PATH):
        url = st.secrets.get("MODEL_URL", None)
        if not url:
            st.error(
                "Model file not found at 'models/model.h5' and no MODEL_URL secret set.\n"
                "Either commit your model to models/ or set MODEL_URL in Streamlit secrets."
            )
            st.stop()
        _download(url, MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

def preprocess(img: Image.Image, size=(224, 224)):
    img = img.convert("RGB").resize(size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, 0)
    return arr

def get_img_size():
    size_str = st.secrets.get("IMG_SIZE", "224,224")
    try:
        h, w = [int(s.strip()) for s in size_str.split(",")]
        return (h, w)
    except Exception:
        return (224, 224)

def get_class_names(num: int):
    default = [f"class_{i}" for i in range(num)]
    names_str = st.secrets.get("CLASS_NAMES", None)
    if not names_str:
        return default
    names = [s.strip() for s in names_str.split(",")]
    if len(names) < num:
        names += [f"class_{i}" for i in range(len(names), num)]
    return names[:num]

st.set_page_config(page_title="Betel Leaf Detection", page_icon="🌿", layout="centered")
st.title("🌿 Betel Leaf Detection")
st.caption("Upload a betel leaf image or take a picture. The model runs server-side.")

# Input choice
source = st.radio("Choose input source", ["Upload", "Camera"], horizontal=True)
file = None
if source == "Upload":
    file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
else:
    snap = st.camera_input("Take a photo")
    if snap:
        file = snap

if file:
    image = Image.open(file)
    st.image(image, caption="Input image", use_column_width=True)

    with st.spinner("Loading model and predicting..."):
        model = load_model_cached()
        size = get_img_size()
        x = preprocess(image, size)
        preds = model.predict(x)

    # Normalize and display predictions
    preds = np.array(preds)
    if preds.ndim == 2 and preds.shape[0] == 1:
        probs = preds[0]
    elif preds.ndim == 1:
        probs = preds
    else:
        probs = preds.flatten()

    # Softmax if not normalized
    if probs.min() < 0 or probs.max() > 1:
        e = np.exp(probs - np.max(probs))
        probs = e / e.sum()

    class_names = get_class_names(len(probs))
    top_idx = int(np.argmax(probs))
    st.subheader(f"Prediction: :green[{class_names[top_idx]}]")
    st.write(f"Confidence: {float(probs[top_idx]):.4f}")
    st.divider()
    st.write("Class probabilities:")
    st.dataframe(
        {
            "class": class_names,
            "probability": [float(p) for p in probs],
        },
        hide_index=True,
        use_container_width=True,
    )

with st.expander("ℹ️ Setup tips"):
    st.markdown(
        "- Put your model at `models/model.h5` **or** set a `MODEL_URL` secret to auto-download on deploy.\n"
        "- Set `CLASS_NAMES` secret like `Healthy,Diseased` to show nice labels.\n"
        "- Set `IMG_SIZE` secret like `224,224` to match your training size."
    )
