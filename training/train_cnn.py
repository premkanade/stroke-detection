"""
Train CNN Model
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
from cnn_model import create_model
from config import EPOCHS, LEARNING_RATE

# =====================================
# Create models folder if not exists
# =====================================

os.makedirs(
    os.path.join(os.path.dirname(__file__), "..", "models"),
    exist_ok=True
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "cnn.keras"
)

HISTORY_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "cnn_history.pkl"
)

# =====================================
# Load Dataset
# =====================================

train_dataset, validation_dataset, test_dataset, class_names = load_datasets()

# =====================================
# Create Model
# =====================================

model = create_model()

# =====================================
# Compile Model
# =====================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

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
# Save History
# =====================================

with open(HISTORY_PATH, "wb") as f:
    pickle.dump(history.history, f)

print("\nTraining History Saved Successfully!")

# =====================================
# Evaluate Model
# =====================================

print("\nEvaluating Model...")

loss, accuracy = model.evaluate(test_dataset)

print(f"\nTest Accuracy : {accuracy*100:.2f}%")
print(f"Test Loss     : {loss:.4f}")

print(f"\nModel saved at: {MODEL_PATH}")

print("\n✅ CNN Training Completed Successfully!")