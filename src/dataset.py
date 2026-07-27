"""
Dataset Loader
Stroke Detection Project
"""

import tensorflow as tf

from config import (
    IMG_HEIGHT,
    IMG_WIDTH,
    BATCH_SIZE,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR
)


def load_datasets():

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    class_names = train_dataset.class_names
    print("Class Names:", class_names)

    return train_dataset, validation_dataset, test_dataset, class_names