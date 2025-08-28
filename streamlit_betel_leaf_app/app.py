import streamlit as st
import av
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "streamlit_betel_leaf_app/models/Betel_Leaf_Model.keras",
        compile=False
    )

model = load_model()
INPUT_HEIGHT, INPUT_WIDTH = model.input_shape[1:3]

CLASS_NAMES = [
    "Anthracnose_Green",
    "BacterialLeafSpot_Green",
    "Healthy_Green",
    "Healthy_Red"
]

# -------------------------------
# Live Video Transformer
# -------------------------------
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_pil = Image.fromarray(img).convert("RGB")
        img_resized = img_pil.resize((INPUT_WIDTH, INPUT_HEIGHT))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        score = tf.nn.softmax(preds[0])
        class_index = np.argmax(score)
        predicted_class = CLASS_NAMES[class_index]
        confidence = 100 * np.max(score)

        # overlay prediction text on frame
        import cv2
        text = f"{predicted_class} ({confidence:.1f}%)"
        cv2.putText(img, text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

# -------------------------------
# App Layout
# -------------------------------
st.set_page_config(page_title="Betel Leaf Live Detection", layout="wide")
st.title("🌿 Betel Leaf Live Monitoring")

st.markdown("Show your betel leaf in front of the webcam for **real-time disease detection**.")

webrtc_streamer(
    key="betel-live",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
)










