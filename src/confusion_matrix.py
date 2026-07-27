"""
Confusion Matrix
Stroke Detection Project
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

from dataset import load_datasets
from config import MODEL_PATH

from tensorflow.keras.applications.efficientnet import preprocess_input

# -------------------------------
# Load Model
# -------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={
        "preprocess_input": preprocess_input
    }
)

# -------------------------------
# Load Dataset
# -------------------------------

_, _, test_dataset, class_names = load_datasets()

# -------------------------------
# True Labels
# -------------------------------

y_true = np.concatenate(
    [np.argmax(labels.numpy(), axis=1) for images, labels in test_dataset]
)

# -------------------------------
# Predictions
# -------------------------------

predictions = model.predict(test_dataset)

y_pred = np.argmax(predictions, axis=1)

# -------------------------------
# Confusion Matrix
# -------------------------------

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix\n")
print(cm)

# -------------------------------
# Plot
# -------------------------------

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig("results/confusion_matrix.png", dpi=300)

plt.show()

print("\nConfusion Matrix saved in results/confusion_matrix.png")