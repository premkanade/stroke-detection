import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

# -----------------------
# Page Configuration
# -----------------------

st.set_page_config(
    page_title="Stroke Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI-Based Stroke Detection")
st.write("Upload a Brain CT Scan to predict whether it is **Normal** or **Stroke**.")

# -----------------------
# Load Model
# -----------------------

@st.cache_resource
def load_my_model():

    model = load_model(
        "models/stroke_detection_model.keras",
        custom_objects={
            "preprocess_input": preprocess_input
        },
        compile=False
    )
    
    st.write("Model loaded from:", "models/stroke_detection_model.keras")
    st.write("Model input shape:", model.input_shape)

    return model

# -----------------------
# Load Model
# -----------------------

model = load_my_model()

# -----------------------
# Upload Image
# -----------------------

uploaded_file = st.file_uploader(
    "Choose a CT Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))
    img = np.array(img)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    st.write("Raw Prediction:", prediction)
    st.write("Predicted Index:", np.argmax(prediction))

    confidence = np.max(prediction)
    predicted_class = np.argmax(prediction)

    classes = ["Normal", "Stroke"]

    st.subheader("Prediction")
    st.success(classes[predicted_class])

    st.write(f"Confidence: {confidence*100:.2f}%")