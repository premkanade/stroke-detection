# 🧠 AI-Based Stroke Detection Using CT Brain Images

An AI-powered stroke detection system that classifies Brain CT scan images as **Normal** or **Stroke** using **EfficientNetB0 Transfer Learning**. The project also includes a **Streamlit web application** for easy image upload and prediction.

---

## 📌 Features

- Detects Stroke from Brain CT images
- EfficientNetB0 Transfer Learning
- Streamlit Web Application
- Confidence Score Prediction
- Grad-CAM Heatmap Visualization
- Confusion Matrix & ROC Curve
- Model Evaluation Report

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- EfficientNetB0
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Pillow

---

## 📂 Project Structure

```
STROKE DETECTION/
│
├── app/
│   └── app.py
│
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   └── stroke_detection_model.keras
│
├── src/
│   ├── train.py
│   ├── model.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── gradcam.py
│   ├── confusion_matrix.py
│   ├── roc_curve.py
│   └── plot_history.py
│
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/stroke-detection.git
```

Go to the project folder

```bash
cd stroke-detection
```

Create Virtual Environment

```bash
python -m venv venv
```

Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python src/train.py
```

---

## 📊 Evaluate Model

```bash
python src/evaluate.py
```

---

## 🔥 Generate Grad-CAM

```bash
python src/gradcam.py
```

---

## 🌐 Run the Streamlit App

```bash
streamlit run app/app.py
```

Open your browser and visit:

```
http://localhost:8501
```

---

## 📈 Model Performance

- Test Accuracy: **75.07%**
- Architecture: **EfficientNetB0**
- Image Size: **224 × 224**
- Classes:
  - Normal
  - Stroke

---

## 📷 Sample Output

- Uploaded CT Image
- Prediction (Normal / Stroke)
- Confidence Score
- Grad-CAM Heatmap

---

## 👨‍💻 Authors

**Prem Kanade**

Sandip Institute of Technology & Research Centre

---

## 📄 License

This project is developed for educational and research purposes.