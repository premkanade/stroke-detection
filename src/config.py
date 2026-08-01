"""
Configuration File
Stroke Detection Project
"""

# Image Size
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Batch Size
BATCH_SIZE = 32

# Number of Classes
NUM_CLASSES = 2

# Epochs
EPOCHS = 20

# Learning Rate
LEARNING_RATE = 0.0003

# Dataset Paths
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
TEST_DIR = "dataset/test"

# Model Save Path
MODEL_PATH = "models/stroke_detection_model.keras"