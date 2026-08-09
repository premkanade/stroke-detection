"""
Generic Grad-CAM
Works with CNN, EfficientNetB0, ResNet50,
DenseNet121 and MobileNetV2
"""

import tensorflow as tf
import numpy as np
import cv2


# ==========================================
# Find Last Convolutional Layer
# ==========================================

def find_last_conv_layer(model):

    for layer in reversed(model.layers):

        # Check nested models
        if isinstance(layer, tf.keras.Model):

            try:
                nested_layer = find_last_conv_layer(layer)

                if nested_layer is not None:
                    return nested_layer

            except Exception:
                pass

        # Standard Conv2D
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer

        # Other convolution-like layers
        if "conv" in layer.name.lower():

            try:
                output_shape = layer.output.shape

                if len(output_shape) == 4:
                    return layer

            except Exception:
                pass

    return None


# ==========================================
# Grad-CAM
# ==========================================

def make_gradcam_heatmap(img, model, pred_index=None):

    # Find appropriate convolutional layer
    last_conv_layer = find_last_conv_layer(model)

    if last_conv_layer is None:
        raise ValueError(
            "Could not find a convolutional layer for Grad-CAM."
        )

    print(
        f"Grad-CAM using layer: {last_conv_layer.name}"
    )

    # Create gradient model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    # Calculate gradients
    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img)

        if pred_index is None:
            pred_index = tf.argmax(
                predictions[0]
            )

        class_channel = predictions[:, pred_index]

    # Gradients
    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    # Global average pooling
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    # Feature maps
    conv_outputs = conv_outputs[0]

    # Weight feature maps
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    # Remove negative values
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize
    max_value = tf.reduce_max(heatmap)

    if max_value > 0:
        heatmap /= max_value

    return heatmap.numpy()


# ==========================================
# Overlay Heatmap
# ==========================================

def overlay_heatmap(original_img, heatmap):

    """
    Creates:
    1. Heatmap image
    2. Heatmap + original image overlay

    Returns:
        heatmap_img,
        overlay_img
    """

    # Convert original image to NumPy
    original_img = np.array(original_img)

    # Make sure image is uint8
    if original_img.dtype != np.uint8:

        original_img = np.clip(
            original_img,
            0,
            255
        ).astype(np.uint8)

    # Resize heatmap to original image size
    heatmap = cv2.resize(
        heatmap,
        (
            original_img.shape[1],
            original_img.shape[0]
        )
    )

    # Convert heatmap to 0-255
    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    # Apply color map
    heatmap_img = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    # OpenCV uses BGR
    # Streamlit expects RGB
    heatmap_img = cv2.cvtColor(
        heatmap_img,
        cv2.COLOR_BGR2RGB
    )

    # Create overlay
    overlay_img = cv2.addWeighted(
        original_img,
        0.6,
        heatmap_img,
        0.4,
        0
    )

    return heatmap_img, overlay_img