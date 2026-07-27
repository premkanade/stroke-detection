import os

# Dataset folders
folders = [
    "dataset/train",
    "dataset/val",
    "dataset/test"
]

print("=" * 60)
print("        DATASET INFORMATION")
print("=" * 60)

for folder in folders:
    print(f"\n📂 {folder}")

    for category in ["Normal", "Stroke"]:
        path = os.path.join(folder, category)

        if os.path.exists(path):
            total = len(os.listdir(path))
            print(f"{category:<10}: {total} images")
        else:
            print(f"{category:<10}: Folder not found")

print("\n" + "=" * 60)