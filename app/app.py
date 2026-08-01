import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
from gradcam import make_gradcam_heatmap, overlay_heatmap
from pdf_report import create_pdf
import os
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Stroke Detection",
    page_icon="🧠",
    layout="centered"
)

# ==========================================
# Custom CSS
# ==========================================

st.markdown("""
<style>

.stApp{
    background-color:#F5F7FA;
}

.result-normal{
    background:#d4edda;
    color:#155724;
    padding:25px;
    border-radius:15px;
    border-left:10px solid #28a745;
    font-size:24px;
    font-weight:bold;
}

.result-stroke{
    background:#f8d7da;
    color:#721c24;
    padding:25px;
    border-radius:15px;
    border-left:10px solid #dc3545;
    font-size:24px;
    font-weight:bold;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
}

.metric-card{
    background:white;
    padding:18px;
    border-radius:15px;
    box-shadow:0 3px 10px rgba(0,0,0,.15);
    text-align:center;
    margin-bottom:15px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# Header
# ==========================================

st.markdown("""
<div style="
    background: linear-gradient(90deg,#0E76A8,#1E88E5);
    padding:25px;
    border-radius:15px;
    text-align:center;
    color:white;
    box-shadow:0px 4px 12px rgba(0,0,0,0.2);
">

<h1>🧠 AI-Based Stroke Detection System</h1>

<h4>Brain CT Image Classification using EfficientNetB0</h4>

<p>
Deep Learning • Explainable AI • Medical Image Analysis
</p>

</div>
""", unsafe_allow_html=True)

st.write("")

st.markdown("---")

st.subheader("📊 Model Dashboard")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🤖 Model",
        "EfficientNetB0"
    )

with col2:
    st.metric(
        "🎯 Accuracy",
        "76%"
    )

with col3:
    st.metric(
        "🧠 Classes",
        "2"
    )

with col4:
    st.metric(
        "📷 Input",
        "224×224"
    )

st.markdown("---")

# ==========================================
# Sidebar
# ==========================================

st.sidebar.title("🧠 Stroke Detection")


st.sidebar.markdown("---")

st.sidebar.info(
"""
### Model Information

**Model:** EfficientNetB0

**Dataset:** Brain CT Images

**Classes**
- Normal
- Stroke

**Test Accuracy:** 76%

**Framework**
- TensorFlow
- Streamlit
"""
)

st.sidebar.markdown("---")

st.sidebar.success(
"Upload a Brain CT Scan to receive an AI prediction."
)

st.sidebar.markdown("---")

st.sidebar.subheader("📖 About Project")

st.sidebar.write("""
This application uses **EfficientNetB0** to detect **Stroke** from Brain CT images.

### Features
- 🧠 Stroke Detection
- 📊 Confidence Score
- 📈 Prediction Probabilities
- 🔥 Grad-CAM Visualization

### Technology
- TensorFlow
- EfficientNetB0
- Streamlit

**Note:** This application is developed for educational and research purposes.
""")

# ==========================================
# Load Model
# ==========================================

@st.cache_resource
def load_my_model():

    model = load_model(
        "models/stroke_detection_model.keras",
        custom_objects={
            "preprocess_input": preprocess_input
        },
        compile=False
    )

    return model


model = load_my_model()
classes = ["Normal", "Stroke"]

# ==========================================
# Prediction History
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

# ==========================================
# About This Application
# ==========================================

with st.expander("ℹ️ About This Application", expanded=False):

    st.markdown("""
### 🧠 AI-Based Stroke Detection System

This application detects **Stroke** from Brain CT images using the **EfficientNetB0** deep learning model.

### Workflow

1️⃣ Upload a Brain CT Scan

2️⃣ Image is preprocessed

3️⃣ EfficientNetB0 analyzes the CT scan

4️⃣ AI predicts:
- 🟢 Normal
- 🔴 Stroke

5️⃣ Confidence score and prediction probabilities are displayed

6️⃣ Grad-CAM highlights the regions that influenced the prediction

---

### ⚠️ Disclaimer

This application is developed for **educational and research purposes only**. It should not be used as a substitute for professional medical diagnosis.
""")
# ==========================================
# Workflow
# ==========================================

with st.expander("📊 System Workflow", expanded=False):

    st.image(
        "app/assets/workflow.png",
        caption="AI-Based Stroke Detection Workflow",
        use_container_width=True
    )

# ==========================================
# Upload Image
# ==========================================
st.markdown("""
<div style="
background:white;
padding:25px;
border-radius:15px;
box-shadow:0px 4px 12px rgba(0,0,0,0.15);
text-align:center;
margin-bottom:15px;
">

<h2 style="color:#0E76A8;">
📤 Upload Brain CT Scan
</h2>

<p style="font-size:18px;">
Drag & Drop or Browse a CT scan image
</p>

<p style="color:gray;">
Supported Formats: JPG • JPEG • PNG
</p>

</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# ==========================================
# Prediction
# ==========================================

if uploaded_file is not None:
    tab1, tab2, tab3, tab4 = st.tabs([
    "🧠 Prediction",
    "🔥 Grad-CAM",
    "📄 Report",
    "ℹ️ About"
    ])
    
    with tab1:
        # Read image
        image = Image.open(uploaded_file).convert("RGB")

        # Display image
        st.markdown("---")
        st.subheader("🧠 Image Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
            image,
            caption="Original CT Scan",
            use_container_width=True
        )
        # ==========================================
        # Image Analysis
        # ==========================================

        st.markdown("---")
        st.subheader("🖼 Image Analysis")

        width, height = image.size

        col1, col2 = st.columns(2)

        with col1:
            st.metric("📄 Format", uploaded_file.type.split("/")[-1].upper())
            st.metric("🎨 Color Mode", image.mode)

        with col2:
            st.metric("📏 Resolution", f"{width} × {height}")
            st.metric("🤖 Status", "Ready")

        if width < 224 or height < 224:
            st.warning("⚠ Image resolution is lower than the recommended 224×224.")
        else:
            st.success("✅ Image quality is suitable for prediction.")

        with col2:
        # Grad-CAM image will appear here
            pass
        # Preprocess image
        img = image.resize((224, 224))
        img = np.array(img)
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img = img.astype(np.float32)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        # Prediction
        with st.spinner("🧠 AI is analysing the Brain CT Scan..."):

            progress = st.progress(0)

        status = st.empty()

        import time

        steps = [
            "Loading EfficientNetB0...",
            "Preprocessing Image...",
            "Extracting Features...",
            "Running AI Model...",
            "Generating Prediction...",
            "Preparing Explainable AI..."
        ]

        for i, step in enumerate(steps):

            status.info(step)

            progress.progress((i + 1) / len(steps))

            time.sleep(0.4)

        prediction = model.predict(img)

        progress.empty()
        status.empty()
        
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        prediction_name = classes[predicted_class]

        new_record = {
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Prediction": prediction_name,
            "Confidence": f"{confidence:.2f}%"
        }

        # Avoid duplicate consecutive entries
        if (
            len(st.session_state.history) == 0 or
            st.session_state.history[-1]["Prediction"] != prediction_name or
            st.session_state.history[-1]["Confidence"] != f"{confidence:.2f}%"
        ):
            st.session_state.history.append(new_record)

            normal_prob = prediction[0][0] * 100
            stroke_prob = prediction[0][1] * 100

            classes = ["Normal", "Stroke"]
    
        # ======================================
        # Generate PDF Report
        # ======================================

        prediction_result = classes[predicted_class]

        create_pdf(
            "stroke_report.pdf",
            prediction_result,
            confidence,
            normal_prob,
            stroke_prob
        )
    
        # ======================================
        # Prediction Result
        # ======================================

        st.markdown("---")

        st.subheader("Prediction Result")

        if predicted_class == 0:

            st.markdown(f"""
            <div class="result-normal">
                🟢 NORMAL
                <br><br>
                Confidence : {confidence:.2f}%
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="result-stroke">
                🔴 STROKE DETECTED
                <br><br>
                Confidence : {confidence:.2f}%
            </div>
            """, unsafe_allow_html=True)

        # ======================================
        # Prediction Probabilities
        # ======================================

        st.markdown("---")
        st.subheader("📊 AI Confidence Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="🎯 Confidence",
                value=f"{confidence:.2f}%"
            )

        with col2:
            st.metric(
                label="🟢 Normal",
                value=f"{normal_prob:.2f}%"
            )
        st.progress(min(max(float(normal_prob)/100,0),1))

        with col3:
            st.metric(
                label="🔴 Stroke",
                value=f"{stroke_prob:.2f}%"
            )
        st.progress(min(max(float(stroke_prob)/100,0),1))
    
        st.markdown("### 🎯 AI Confidence Gauge")

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=confidence,
                number={'suffix': "%"},
                title={'text': "Model Confidence"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#0E76A8"},
                    'steps': [
                        {'range': [0, 50], 'color': "#ffcccc"},
                        {'range': [50, 75], 'color': "#ffe699"},
                        {'range': [75, 100], 'color': "#ccffcc"}
                    ]
                }
         )
        )

        fig.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)
        
    # ======================================
    # Grad-CAM Visualization
    # ======================================
    with tab2:

        st.markdown("""
        <div style="
        background: linear-gradient(90deg,#6A11CB,#2575FC);
        padding:15px;
        border-radius:12px;
        text-align:center;
        color:white;
        margin-bottom:20px;
        ">
        <h2>🔥 Explainable AI Visualization</h2>
        <p>Grad-CAM highlights the regions that influenced the model's decision.</p>
        </div>
        """, unsafe_allow_html=True)

        heatmap = make_gradcam_heatmap(img, model)

        original = np.array(image)

        heatmap_img, overlay_img = overlay_heatmap(original, heatmap)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(
                original,
                caption="🖼️ Original CT Scan",
                use_container_width=True
            )

        with col2:
            st.image(
                heatmap_img,
                caption="🔥 Attention Heatmap",
                use_container_width=True
            )

        with col3:
            st.image(
                overlay_img,
                caption="🧠 AI Explanation Overlay",
                use_container_width=True
            )

        with st.expander("Model Output"):
            st.write("Raw Prediction:", prediction)
            st.write("Predicted Index:", predicted_class)
    
    # ======================================
    # Debug
    # ======================================

    with st.expander("Model Output"):
        st.write("Raw Prediction:", prediction)
        st.write("Predicted Index:", predicted_class)
    
    st.markdown("---")

    # ======================================
    # Download PDF Report
    # ======================================

    with tab3:

        st.subheader("📄 Medical Report")

        st.info(
        "Download a PDF report containing the prediction result and confidence."
        )

        with open("stroke_report.pdf", "rb") as pdf_file:

            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_file,
                file_name="Stroke_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    
    with tab4:

        st.markdown("""
        <div style="
        background:white;
        padding:25px;
        border-radius:15px;
        box-shadow:0px 4px 10px rgba(0,0,0,0.15);
        ">

        <h2 style="color:#0E76A8;">🧠 About This Application</h2>

        <hr>

        <h4>Project Description</h4>

        <p>
        This AI-powered application detects Stroke from Brain CT Scan images using
        the EfficientNetB0 Deep Learning model.
        </p>

        <h4>Key Features</h4>

        <ul>
        <li>✅ Stroke Detection</li>
        <li>✅ EfficientNetB0 Transfer Learning</li>
        <li>✅ Grad-CAM Explainable AI</li>
        <li>✅ Confidence Analysis</li>
        <li>✅ PDF Report Generation</li>
        <li>✅ Interactive Dashboard</li>
        </ul>

        <h4>Technologies Used</h4>

        <ul>
        <li>TensorFlow</li>
        <li>EfficientNetB0</li>
        <li>Streamlit</li>
        <li>OpenCV</li>
        <li>NumPy</li>
        <li>Pillow</li>
        <li>ReportLab</li>
        </ul>

        <hr>

        <center>

        <h4>Final Year Project</h4>

        <b>AI-Based Stroke Detection Using Brain CT Images</b>

        <br><br>

        Developed by

        <br>

        <b>Prem Kanade</b>

        </center>

        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        st.subheader("📜 Prediction History")

        if len(st.session_state.history) > 0:

            df = pd.DataFrame(st.session_state.history)

            st.dataframe(
                df[::-1],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info("No predictions made yet.")
# ======================================
# Footer
# ======================================

st.markdown("---")

st.markdown("""
<div class="footer">

### 🧠 AI-Based Stroke Detection System

Developed using <b>TensorFlow</b>, <b>EfficientNetB0</b>, and <b>Streamlit</b>

© 2026 Final Year Project

</div>
""", unsafe_allow_html=True)

