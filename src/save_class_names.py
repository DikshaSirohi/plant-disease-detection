import os, json
import tensorflow as tf
from src.config import TRAIN_DIR, IMG_SIZE, BATCH_SIZE

def main():
    tmp = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    os.makedirs("models", exist_ok=True)
    with open("models/class_names.json", "w", encoding="utf-8") as f:
        json.dump(tmp.class_names, f, indent=2)
    print("Saved models/class_names.json")

if __name__ == "__main__":
    main()