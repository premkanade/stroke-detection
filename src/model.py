"""
EfficientNetB0 Model
Stroke Detection Project
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from config import IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES


def create_model():

    # Input Layer
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # ==========================================
    # Data Augmentation
    # ==========================================

    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
        layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),
    ], name="data_augmentation")

    x = data_augmentation(inputs)
    # EfficientNet preprocessing
    x = layers.Lambda(preprocess_input, name="preprocess")(x)

    # Base Model
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=x
    )

    # Fine-tuning
    base_model.trainable = True

    # Freeze all layers except the last 20
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    print("\nTrainable Layers:\n")

    for layer in base_model.layers[-20:]:
        print(layer.name, layer.trainable)

    # ==========================================
    # Classification Head
    # ==========================================

    x = base_model.output

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dense(
        512,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.0001)
    )(x)

    x = layers.Dropout(0.5)(x)

    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.0001)
    )(x)

    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="predictions"
    )(x)

    model = Model(inputs, outputs)

    return model