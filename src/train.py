"""
Train EfficientNetB0 Model
Stroke Detection Project
"""

import os
import tensorflow as tf

from dataset import load_datasets
from model import create_model
from config import EPOCHS, LEARNING_RATE, MODEL_PATH

# ===============================
# Create models folder
# ===============================
os.makedirs("models", exist_ok=True)

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

train_dataset, val_dataset, test_dataset, class_names = load_datasets()

print("\n========================")
print("CLASS NAMES:", class_names)
print("========================\n")
exit()

print("\nClasses:", class_names)

# ===============================
# Improve Dataset Performance
# ===============================
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(AUTOTUNE)

print("\nCreating Model...")

model = create_model()

print("\nCompiling Model...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Summary:\n")
model.summary()

# ===============================
# Callbacks
# ===============================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# ===============================
# Class Weights
# ===============================

class_weight = {
    0: 1.0,   # Normal
    1: 1.63   # Stroke
}

print("\nStarting Training...\n")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ],
    verbose=1
)

import pickle

os.makedirs("results", exist_ok=True)

with open("results/history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("\nTraining history saved successfully!")

print("\nTraining Completed Successfully!")

print("\nEvaluating Model...")

loss, accuracy = model.evaluate(test_dataset)

print(f"\nTest Accuracy : {accuracy*100:.2f}%")
print(f"Test Loss     : {loss:.4f}")

print("\nModel saved at:", MODEL_PATH)