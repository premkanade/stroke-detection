"""
CNN Model
Stroke Detection Project
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

from config import IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES


def create_model():

    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # Data Augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.10)(x)
    x = layers.RandomZoom(0.10)(x)

    # Normalize
    x = layers.Rescaling(1./255)(x)

    # Block 1
    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Block 2
    x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Block 3
    x = layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Block 4
    x = layers.Conv2D(256, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="predictions"
    )(x)

    model = Model(inputs, outputs)

    return model