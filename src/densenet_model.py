"""
DenseNet121 Model
Stroke Detection Project
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

from config import IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES


def create_model():

    # =====================================
    # Input Layer
    # =====================================

    inputs = layers.Input(
        shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )

    # =====================================
    # Data Augmentation
    # =====================================

    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.10)(x)
    x = layers.RandomZoom(0.10)(x)

    # =====================================
    # DenseNet121 Preprocessing
    # =====================================

    x = layers.Lambda(
        preprocess_input,
        name="preprocess"
    )(x)

    # =====================================
    # DenseNet121 Base Model
    # =====================================

    base_model = DenseNet121(
        include_top=False,
        weights="imagenet",
        input_tensor=x
    )

    # =====================================
    # Fine-Tuning
    # =====================================

    base_model.trainable = True

    # Freeze all layers except the last 20
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    print("\nTrainable DenseNet121 Layers:\n")

    for layer in base_model.layers[-20:]:
        print(layer.name, layer.trainable)

    # =====================================
    # Classification Head
    # =====================================

    x = base_model.output

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="predictions"
    )(x)

    # =====================================
    # Final Model
    # =====================================

    model = Model(
        inputs,
        outputs
    )

    return model