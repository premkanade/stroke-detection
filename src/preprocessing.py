import splitfolders
import os

# Source dataset
input_folder = "dataset/raw/Brain_Data_Organised"

# Destination folder
output_folder = "dataset"

# Split ratio
# Train = 70%
# Validation = 15%
# Test = 15%

splitfolders.ratio(
    input=input_folder,
    output=output_folder,
    seed=42,
    ratio=(0.7, 0.15, 0.15),
    group_prefix=None,
    move=False
)

print("=" * 50)
print("Dataset Successfully Split!")
print("=" * 50)