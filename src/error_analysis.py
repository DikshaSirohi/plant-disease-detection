import os, shutil, sys
import numpy as np
import tensorflow as tf
from src.config import TEST_DIR, IMG_SIZE, BATCH_SIZE, MIS_DIR

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def main(model_path: str, max_per_pair: int = 30):
    os.makedirs(MIS_DIR, exist_ok=True)
    model = tf.keras.models.load_model(model_path)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    class_names = test_ds.class_names
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    # filepaths in deterministic order
    file_paths = []
    for root, _, files in os.walk(TEST_DIR):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                file_paths.append(os.path.join(root, fn))
    file_paths.sort()

    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    y_prob = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    pair_count = {}
    copied = 0

    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if t == p:
            continue

        true_name = class_names[int(t)]
        pred_name = class_names[int(p)]
        key = (true_name, pred_name)

        pair_count[key] = pair_count.get(key, 0) + 1
        if pair_count[key] > max_per_pair:
            continue

        src = file_paths[i]
        dst_dir = os.path.join(MIS_DIR, true_name, f"pred_{pred_name}")
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))
        copied += 1

    print(f"Copied {copied} misclassified images to: {MIS_DIR}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.error_analysis <model_path>")
    main(sys.argv[1])