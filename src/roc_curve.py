"""
ROC Curve
Stroke Detection Project
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

from tensorflow.keras.applications.efficientnet import preprocess_input

from dataset import load_datasets
from config import MODEL_PATH

# ---------------------------------
# Load Model
# ---------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "preprocess_input": preprocess_input
    }
)

# ---------------------------------
# Load Dataset
# ---------------------------------

_, _, test_dataset, class_names = load_datasets()

# ---------------------------------
# True Labels
# ---------------------------------

y_true = np.concatenate([
    np.argmax(labels.numpy(), axis=1)
    for images, labels in test_dataset
])

# ---------------------------------
# Predictions
# ---------------------------------

predictions = model.predict(test_dataset)

# Probability of Stroke class
y_score = predictions[:, 1]

# ---------------------------------
# ROC Curve
# ---------------------------------

fpr, tpr, thresholds = roc_curve(y_true, y_score)

roc_auc = auc(fpr, tpr)

print(f"\nAUC Score : {roc_auc:.4f}")

# ---------------------------------
# Plot
# ---------------------------------

plt.figure(figsize=(7,6))

plt.plot(
    fpr,
    tpr,
    color="blue",
    linewidth=2,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--",
    color="red"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend(loc="lower right")

plt.grid(True)

plt.tight_layout()

plt.savefig("results/roc_curve.png", dpi=300)

plt.show()

print("\nROC Curve saved in results/roc_curve.png")