"""
Grad-CAM Visualization
"""

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# -----------------------
# Load trained model
# -----------------------

model = load_model(
    "models/stroke_detection_model.keras",
    custom_objects={"preprocess_input": preprocess_input},
    compile=False
)

print("Model Loaded Successfully!")

# Print model layers

for i, layer in enumerate(model.layers):
    print(i, layer.name)

print("\nEfficientNetB0 Internal Layers:\n")

last_conv_layer_name = "top_activation"

print("\nLast Conv Layer:", last_conv_layer_name)

# -----------------------
# Load Image
# -----------------------

IMAGE_PATH = "dataset/test/Stroke/58 (6).jpg"

img = cv2.imread(IMAGE_PATH)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

original = img.copy()

img = cv2.resize(img, (224, 224))

img = np.expand_dims(img, axis=0)

img = preprocess_input(img.astype(np.float32))

print("\nImage Loaded Successfully!")
print("Image Shape:", img.shape)

# Last convolution layer

last_conv_layer_name = "top_activation"

# -----------------------
# Create Grad-CAM Model
# -----------------------

grad_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=[
        model.get_layer(last_conv_layer_name).output,
        model.output
    ]
)

# -----------------------
# Compute Heatmap
# -----------------------

with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(img)

    predicted_class = tf.argmax(predictions[0])

    loss = predictions[:, predicted_class]

grads = tape.gradient(loss, conv_outputs)

pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

print("\nPrediction Vector:", predictions.numpy())
print("Predicted Class:", predicted_class.numpy())

print("Conv Output Shape:", conv_outputs.shape)
print("Gradient Shape:", grads.shape)

# -----------------------
# Generate Heatmap
# -----------------------

pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0]

heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

heatmap = tf.maximum(heatmap, 0)
heatmap /= tf.math.reduce_max(heatmap)

heatmap = heatmap.numpy()

print("\nHeatmap Generated!")
print("Heatmap Shape:", heatmap.shape)

# -----------------------
# Overlay Heatmap
# -----------------------

# Resize heatmap to original image size
heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))

# Convert heatmap to 0-255
heatmap = np.uint8(255 * heatmap)

# Apply color map
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Convert BGR to RGB
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# Overlay heatmap on original image
superimposed_img = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

# -----------------------
# Display Images
# -----------------------

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(original)
plt.title("Original CT")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(heatmap)
plt.title("Grad-CAM Heatmap")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(superimposed_img)
plt.title("Overlay")
plt.axis("off")

plt.tight_layout()

# Save result
cv2.imwrite(
    "results/gradcam_result.png",
    cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)
)

plt.show()

print("\nGrad-CAM saved successfully!")