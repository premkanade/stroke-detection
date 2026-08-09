"""
Train MobileNetV2 Model
Stroke Detection Project
"""

import os
import sys
import tensorflow as tf
import pickle

# =====================================
# Add src folder to Python path
# =====================================

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src")
    )
)

from dataset import load_datasets
from mobilenet_model import create_model
from config import EPOCHS


# =====================================
# Paths
# =====================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "mobilenetv2.keras"
)

HISTORY_PATH = os.path.join(
    MODELS_DIR,
    "mobilenetv2_history.pkl"
)


# =====================================
# Load Dataset
# =====================================

print("\nLoading Dataset...")

(
    train_dataset,
    validation_dataset,
    test_dataset,
    class_names
) = load_datasets()

print("\nClass Names:", class_names)

# =====================================
# MobileNetV2 Preprocessing
# =====================================

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

train_dataset = train_dataset.map(
    lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y)
)

validation_dataset = validation_dataset.map(
    lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y)
)

test_dataset = test_dataset.map(
    lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y)
)


# =====================================
# Create Model
# =====================================

print("\nCreating MobileNetV2 Model...")

model = create_model()


# =====================================
# Compile Model
# =====================================

MOBILENET_LEARNING_RATE = 0.00005

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=MOBILENET_LEARNING_RATE
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# =====================================
# Model Summary
# =====================================

model.summary()


# =====================================
# Callbacks
# =====================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)


# =====================================
# Train Model
# =====================================

print("\nStarting MobileNetV2 Training...\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ]
)


# =====================================
# Save Training History
# =====================================

with open(HISTORY_PATH, "wb") as f:
    pickle.dump(history.history, f)

print("\nTraining History Saved Successfully!")


# =====================================
# Evaluate Model
# =====================================

print("\nEvaluating MobileNetV2 Model...")

loss, accuracy = model.evaluate(test_dataset)

print(f"\nTest Accuracy : {accuracy * 100:.2f}%")
print(f"Test Loss     : {loss:.4f}")


# =====================================
# Final Information
# =====================================

print(f"\nModel saved at: {MODEL_PATH}")

print("\n✅ MobileNetV2 Training Completed Successfully!")