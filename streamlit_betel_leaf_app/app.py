import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import threading
import csv
import io
import time

st.set_page_config(page_title="Betel Leaf Disease Detection", layout="wide")

# -------------------------------
# Load Model (cached)
# -------------------------------
@st.cache_resource
def load_model():
    # Put your file next to app.py or adjust the path
    return tf.keras.models.load_model("streamlit_betel_leaf_app/models/Betel_Leaf_Model.keras", compile=False)

model = load_model()
# infer input shape (ignore batch dim)
_, INPUT_H, INPUT_W, INPUT_C = model.input_shape

CLASS_NAMES = [
    "Anthracnose_Green",
    "BacterialLeafSpot_Green",
    "Healthy_Green",
    "Healthy_Red",
]

predict_lock = threading.Lock()  # TF-safe

def preprocess_pil(pil_img: Image.Image) -> np.ndarray:
    # Convert to RGB always
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize((INPUT_W, INPUT_H))
    arr = np.asarray(pil_img).astype("float32") / 255.0
    if INPUT_C == 1:
        # convert to grayscale channel if model expects 1
        arr = np.mean(arr, axis=-1, keepdims=True)
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(pil_img: Image.Image):
    x = preprocess_pil(pil_img)
    with predict_lock:
        preds = model.predict(x, verbose=0)
    probs = tf.nn.softmax(preds[0]).numpy()
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]) * 100.0, probs

# -------------------------------
# UI Tabs
# -------------------------------
tabs = st.tabs(["Home", "Live Predict", "Upload Predict", "About Leaf", "About Us", "Feedback"])

# -------------------------------
# Home
# -------------------------------
with tabs[0]:
    st.title("🌿 Betel Leaf Disease Detection AI (Live)")
    st.markdown(
        "Use **Live Predict** for continuous webcam inference, or **Upload Predict** to test images."
    )

# -------------------------------
# Live Predict (continuous)
# -------------------------------
with tabs[1]:
    st.header("📹 Live Webcam Prediction (Continuous)")
    st.caption("No OpenCV required. Works entirely with Pillow + streamlit-webrtc.")

    # Controls
    colA, colB, colC = st.columns(3)
    with colA:
        run_every_n = st.number_input("Run inference every N frames", min_value=1, max_value=10, value=3, step=1)
    with colB:
        show_probs = st.checkbox("Show per-class probabilities", value=False)
    with colC:
        label_pos = st.selectbox("Label position", ["top-left", "top-right", "bottom-left", "bottom-right"], index=0)

    # Simple FPS meter (optional overlay)
    want_fps = st.checkbox("Show FPS", value=True)

    # Drawing helpers
    def draw_label(img_pil: Image.Image, text: str, position: str = "top-left"):
        draw = ImageDraw.Draw(img_pil)
        # Try to load a nicer font if available, else default
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 20)
        except Exception:
            font = ImageFont.load_default()

        w, h = img_pil.size
        pad = 8
        tw, th = draw.textlength(text, font=font), 22  # approximate height
        if position == "top-left":
            xy = (pad, pad)
        elif position == "top-right":
            xy = (w - pad - tw, pad)
        elif position == "bottom-left":
            xy = (pad, h - pad - th)
        else:
            xy = (w - pad - tw, h - pad - th)

        # semi-transparent rectangle behind text
        rect = [xy[0] - 4, xy[1] - 2, xy[0] + tw + 4, xy[1] + th + 2]
        draw.rectangle(rect, fill=(0, 0, 0, 128))
        draw.text(xy, text, fill=(255, 255, 255), font=font)

    class LivePredictor(VideoTransformerBase):
        def __init__(self):
            self.frame_count = 0
            self.last_pred = ("—", 0.0, None)  # (label, conf, probs)
            self.last_time = time.time()
            self.fps = 0.0

        def transform(self, frame: av.VideoFrame):
            # Get RGB frame
            img = frame.to_ndarray(format="rgb24")
            pil = Image.fromarray(img)

            # FPS calc
            now = time.time()
            dt = now - self.last_time
            self.last_time = now
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps > 0 else (1.0 / dt)

            # Run inference every N frames to save CPU
            self.frame_count += 1
            if self.frame_count % int(run_every_n) == 0:
                label, conf, probs = predict_image(pil)
                self.last_pred = (label, conf, probs)

            # Draw overlays
            label, conf, probs = self.last_pred
            overlay_text = f"{label}  ({conf:.1f}%)"
            draw_label(pil, overlay_text, label_pos)

            if want_fps:
                draw_label(pil, f"FPS: {self.fps:.1f}", "bottom-right")

            if show_probs and probs is not None:
                # Build a small probability table string
                lines = [f"{c}: {p*100:.1f}%" for c, p in zip(CLASS_NAMES, probs)]
                draw_label(pil, " | ".join(lines), "bottom-left")

            return np.array(pil)

    webrtc_streamer(
        key="betel-live",
        video_transformer_factory=LivePredictor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# -------------------------------
# Upload Predict (single image)
# -------------------------------
with tabs[2]:
    st.header("🖼️ Upload Image for Prediction")
    uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded", use_column_width=True)

        label, conf, probs = predict_image(img)
        st.success(f"Prediction: **{label}** ({conf:.2f}%)")
        st.progress(min(int(conf), 100))

        with st.expander("Class probabilities"):
            for cname, p in zip(CLASS_NAMES, probs):
                st.write(f"{cname}: {p*100:.2f}%")

# -------------------------------
# About Leaf
# -------------------------------
with tabs[3]:
    st.header("About Betel Leaf (Piper betle)")
    st.markdown(
        """
**Common classes**:
- Anthracnose (Green)
- Bacterial Leaf Spot (Green)
- Healthy Green
- Healthy Red
"""
    )

# -------------------------------
# About Us
# -------------------------------
with tabs[4]:
    st.header("About Us")
    st.markdown(
        """
We are students from **Rajalakshmi Engineering College, Department of Mechatronics Engineering**,  
working on Machine Learning and AI projects.
"""
    )

# -------------------------------
# Feedback
# -------------------------------
with tabs[5]:
    st.header("💬 Leave Your Feedback")
    with st.form("feedback_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Your Message / Feedback")
        submitted = st.form_submit_button("Submit")
        if submitted:
            with open("feedback.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([name, email, message])
            st.success("✅ Thank you for your feedback!")
    st.caption("© 2025 ProjectASA2025")











