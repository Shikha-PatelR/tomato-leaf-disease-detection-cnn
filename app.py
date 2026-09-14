import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# =========================
# Configuration
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tomato_disease_cnn.keras"
)

IMAGE_SIZE = (128, 128)

CLASS_NAMES = [
    "Early Blight",
    "Late Blight",
    "Healthy"
]

DISEASE_INFO = {
    "Early Blight": {
        "description": "The model detected patterns associated with Tomato Early Blight.",
        "advice": "Remove severely affected leaves and maintain good airflow around plants."
    },
    "Late Blight": {
        "description": "The model detected patterns associated with Tomato Late Blight.",
        "advice": "Remove infected plant material and avoid prolonged leaf wetness."
    },
    "Healthy": {
        "description": "The model detected a healthy-looking tomato leaf.",
        "advice": "Continue regular monitoring and maintain proper plant care."
    }
}

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Tomato Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# =========================
# Custom styling
# =========================
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Title
# =========================
st.markdown(
    '<div class="main-title">🌿 Tomato Disease Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">CNN-based AI system for tomato leaf disease classification</div>',
    unsafe_allow_html=True
)

st.divider()

# =========================
# Load model
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# =========================
# Upload image
# =========================
st.subheader("📷 Upload Tomato Leaf Image")

uploaded_file = st.file_uploader(
    "Choose a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# Prediction
# =========================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf Image",
        width="stretch"
    )

    if st.button("🔍 Detect Disease", type="primary"):

        with st.spinner("Analyzing leaf image..."):

            resized_image = image.resize(IMAGE_SIZE)

            image_array = np.array(
                resized_image,
                dtype=np.float32
            )

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            predictions = model.predict(
                image_array,
                verbose=0
            )

            predicted_index = np.argmax(predictions[0])

            predicted_class = CLASS_NAMES[predicted_index]

            confidence = (
                predictions[0][predicted_index] * 100
            )

        # =========================
        # Display prediction
        # =========================
        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        st.subheader("🎯 Prediction Result")

        st.success(
            f"Prediction: {predicted_class}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.markdown(
            f"**Description:** "
            f"{DISEASE_INFO[predicted_class]['description']}"
        )

        st.markdown(
            f"**Recommended Action:** "
            f"{DISEASE_INFO[predicted_class]['advice']}"
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # =========================
        # Probabilities
        # =========================
        st.subheader("📊 Class Probabilities")

        for class_name, probability in zip(
            CLASS_NAMES,
            predictions[0]
        ):
            st.write(
                f"**{class_name}**: "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                float(probability)
            )

else:
    st.info(
        "Upload a tomato leaf image to begin disease detection."
    )

# =========================
# Footer
# =========================
st.divider()

st.caption(
    "Tomato Disease Detection • "
    "Deep Learning CNN • College Project"
)