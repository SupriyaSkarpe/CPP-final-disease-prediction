import os, shutil, random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCE_DIR = os.path.join(BASE_DIR, "ecg_images")
DEST_DIR = os.path.join(BASE_DIR, "ecg_dataset")

CLASSES = ["normal", "arrhythmia"]
SPLIT = 0.8

for split in ["train", "test"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(DEST_DIR, split, cls), exist_ok=True)

for cls in CLASSES:
    src = os.path.join(SOURCE_DIR, cls)
    images = os.listdir(src)
    random.shuffle(images)

    split_idx = int(len(images) * SPLIT)
    train_imgs = images[:split_idx]
    test_imgs = images[split_idx:]

    for img in train_imgs:
        shutil.copy(
            os.path.join(src, img),
            os.path.join(DEST_DIR, "train", cls, img)
        )

    for img in test_imgs:
        shutil.copy(
            os.path.join(src, img),
            os.path.join(DEST_DIR, "test", cls, img)
        )

    print(f"{cls}: {len(train_imgs)} train | {len(test_imgs)} test")

print("✅ ECG dataset split completed")
