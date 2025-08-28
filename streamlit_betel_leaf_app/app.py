import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("streamlit_betel_leaf_app/models/Betel_Leaf_Model.keras")

model = load_model()

CLASS_NAMES = [
    "Anthracnose_Green",
    "BacterialLeafSpot_Green",
    "Healthy_Green",
    "Healthy_Red"
]

# -------------------------------
# Custom Video Transformer
# -------------------------------
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Preprocess for model
        img_resized = tf.image.resize(img, (224, 224))
        img_array = np.expand_dims(img_resized / 255.0, axis=0)

        # Predict
        preds = model.predict(img_array, verbose=0)
        score = tf.nn.softmax(preds[0])
        class_index = np.argmax(score)
        label = f"{CLASS_NAMES[class_index]} ({100*np.max(score):.2f}%)"

        # Draw label on frame
        import cv2
        cv2.putText(img, label, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return img

# -------------------------------
# Streamlit Layout
# -------------------------------
st.set_page_config(page_title="Live Betel Leaf Monitoring", page_icon="🌿")

st.title("🌿 Live Betel Leaf Disease Monitoring")
st.write("This demo uses your webcam to monitor betel leaves in **real-time**.")

webrtc_streamer(
    key="live-monitor",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False}
)









