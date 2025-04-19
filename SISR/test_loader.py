import os
from data.data_loader import get_loader

# ✅ Set to your generated LR folder
data_path = os.getenv("DATA_ROOT", "E:/Datasets/DIV2K_train_LR_X2")

if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"❌ Data folder not found: {data_path}\n"
        "➡ Set DATA_ROOT or update the fallback path."
    )

# Load 2 images as a test batch
loader = get_loader(data_path, batch_size=2, train=True)

# Get one batch and print the tensor sizes
batch = next(iter(loader))

print("✅ DataLoader test successful!")
print("Input (LR):", batch["input"].shape)
print("Target (HR):", batch["target"].shape)