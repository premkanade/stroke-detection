"""
Model Evaluation
Stroke Detection Project
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from tensorflow.keras.applications.efficientnet import preprocess_input

from dataset import load_datasets
from config import MODEL_PATH

print("=" * 60)
print("Loading Saved Model...")
print("=" * 60)

# Load trained model
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "preprocess_input": preprocess_input
    }
)

# Load datasets
_, _, test_dataset, class_names = load_datasets()

print("\nClasses:", class_names)

print("Class Names:", class_names)
print("Test File Count:", len(test_dataset.file_paths))

print("\nFirst 10 Test Images:\n")

for i in range(10):
    print(test_dataset.file_paths[i])

print("\nPredicting Test Dataset...")

# Predictions
y_pred_prob = model.predict(test_dataset)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\nSample Predictions:\n")

for i in range(10):
    print(
        f"Image {i}: Probabilities = {y_pred_prob[i]}, "
        f"Predicted = {y_pred[i]}"
    )
# True labels
y_true = np.concatenate([
    np.argmax(labels.numpy(), axis=1)
    for images, labels in test_dataset
])

print("\nClassification Report\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)