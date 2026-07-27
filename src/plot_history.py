"""
Plot Training History
"""

import matplotlib.pyplot as plt
import pickle

# Load history
with open("results/history.pkl", "rb") as f:
    history = pickle.load(f)

# Accuracy Plot
plt.figure(figsize=(8,5))
plt.plot(history["accuracy"], label="Training Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("results/accuracy_plot.png")
plt.close()

# Loss Plot
plt.figure(figsize=(8,5))
plt.plot(history["loss"], label="Training Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("results/loss_plot.png")
plt.close()

print("Graphs saved successfully!")

# Show both graphs together
plt.show()