import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageStat

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

# Confidence threshold for warning
CONFIDENCE_THRESHOLD = 0.70

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
        margin-bottom: 25px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-top: 20px;
    }

  .guide-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #f5f8f5;
    color: #222222 !important;
    margin-bottom: 20px;
}

.guide-box b {
    color: #222222 !important;
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
    '<div class="subtitle">'
    'CNN-based AI system for tomato leaf disease classification'
    '</div>',
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
# Photo Guidelines
# =========================
st.subheader("📸 For Better Detection")

st.markdown(
    """
    <div class="guide-box">

    **Before taking/uploading a photo:**

    ✅ Keep <b>one tomato leaf</b> clearly visible<br>
    ✅ Keep the leaf near the center<br>
    ✅ Use natural or bright light<br>
    ✅ Avoid strong shadows<br>
    ✅ Avoid blurry photos<br>
    ✅ Try to keep the complete leaf inside the frame<br>
    ✅ Avoid objects covering the leaf

    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# Input selection
# =========================
st.subheader("📷 Choose Image Source")

input_mode = st.radio(
    "Select how you want to provide the leaf image:",
    ["📤 Upload Image", "📷 Take Photo"],
    horizontal=True
)

image = None
source_name = ""

# =========================
# Upload Image
# =========================
if input_mode == "📤 Upload Image":

    uploaded_file = st.file_uploader(
        "Choose a tomato leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        source_name = "Uploaded Leaf Image"

# =========================
# Camera Input
# =========================
else:

    camera_photo = st.camera_input(
        "Take a picture of the tomato leaf"
    )

    if camera_photo is not None:
        image = Image.open(camera_photo).convert("RGB")
        source_name = "Camera Photo"

# =========================
# Image processing / quality checks
# =========================
if image is not None:

    st.subheader("🖼️ Image Preview")

    st.image(
        image,
        caption=source_name,
        width="stretch"
    )

    # -------------------------
    # Basic image information
    # -------------------------
    width, height = image.size

    gray_image = image.convert("L")

    stats = ImageStat.Stat(gray_image)
    mean_brightness = stats.mean[0]

    # Simple blur estimate using image gradients
    gray_array = np.asarray(gray_image, dtype=np.float32)

    if gray_array.shape[0] > 2 and gray_array.shape[1] > 2:

        dx = np.diff(gray_array, axis=1)
        dy = np.diff(gray_array, axis=0)

        sharpness_score = (
            np.var(dx) + np.var(dy)
        )

    else:
        sharpness_score = 0

    # -------------------------
    # Quality warnings
    # -------------------------
    quality_warnings = []

    if width < 150 or height < 150:
        quality_warnings.append(
            "Image resolution is quite low."
        )

    if mean_brightness < 45:
        quality_warnings.append(
            "Image appears too dark."
        )

    if mean_brightness > 220:
        quality_warnings.append(
            "Image appears very bright/overexposed."
        )

    if sharpness_score < 20:
        quality_warnings.append(
            "Image may be blurry."
        )

    if quality_warnings:

        st.warning("⚠️ Image Quality Warning")

        for warning in quality_warnings:
            st.write(f"• {warning}")

        st.info(
            "For better detection, try taking another clearer photo."
        )

    else:

        st.success(
            "✅ Image quality looks suitable for detection."
        )

    # =========================
    # Detect button
    # =========================
    if st.button(
        "🔍 Detect Disease",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Analyzing leaf image..."):

            # IMPORTANT:
            # Model already contains Rescaling(1/255)
            # Therefore DO NOT divide image_array by 255 here.

            resized_image = image.resize(
                IMAGE_SIZE
            )

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

            probabilities = predictions[0]

            predicted_index = np.argmax(
                probabilities
            )

            predicted_class = CLASS_NAMES[
                predicted_index
            ]

            confidence = float(
                probabilities[predicted_index]
            )

        # =========================
        # Prediction result
        # =========================
        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        st.subheader("🎯 Prediction Result")

        # -------------------------
        # Confidence handling
        # -------------------------
        if confidence >= CONFIDENCE_THRESHOLD:

            st.success(
                f"Prediction: {predicted_class}"
            )

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

        else:

            st.warning(
                "⚠️ The model is not sufficiently confident."
            )

            st.metric(
                "Top Prediction",
                predicted_class
            )

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.info(
                "Please capture a clearer photo of a single "
                "tomato leaf with good lighting and try again."
            )

        # -------------------------
        # Disease information
        # -------------------------
        st.markdown(
            f"**Description:** "
            f"{DISEASE_INFO[predicted_class]['description']}"
        )

        st.markdown(
            f"**Recommended Action:** "
            f"{DISEASE_INFO[predicted_class]['advice']}"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

        # =========================
        # Class probabilities
        # =========================
        st.subheader("📊 Class Probabilities")

        for class_name, probability in zip(
            CLASS_NAMES,
            probabilities
        ):

            percentage = float(
                probability * 100
            )

            st.write(
                f"**{class_name}**: "
                f"{percentage:.2f}%"
            )

            st.progress(
                float(probability)
            )

        # =========================
        # Final guidance
        # =========================
        st.divider()

        st.caption(
            "⚠️ AI prediction is an automated screening result "
            "and should not replace expert agricultural diagnosis."
        )

else:

    st.info(
        "Upload a tomato leaf image or take a photo "
        "to begin disease detection."
    )

# =========================
# Footer
# =========================
st.divider()

st.caption(
    "Tomato Disease Detection • "
    "Deep Learning CNN • College Project"
)
