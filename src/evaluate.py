import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import TRAIN_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE, REPORT_DIR, CM_DIR


def model_has_rescaling(model: tf.keras.Model) -> bool:
    """Return True if the model already contains a Rescaling layer anywhere."""
    def walk(m):
        for layer in m.layers:
            if isinstance(layer, tf.keras.layers.Rescaling):
                return True
            if isinstance(layer, tf.keras.Model) and walk(layer):
                return True
        return False
    return walk(model)


def load_class_names() -> list:
    """
    Lock class order to training order.
    If models/class_names.json exists, use it (most stable).
    Otherwise derive from TRAIN_DIR via image_dataset_from_directory.
    """
    json_path = os.path.join("models", "class_names.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    tmp_train = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="int",
    )
    return tmp_train.class_names


def main(model_path: str):
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(CM_DIR, exist_ok=True)

    print("Loading model...")
    model = tf.keras.models.load_model(model_path)

    print("Loading class_names from TRAIN_DIR (training label order)...")
    class_names = load_class_names()
    print(f"Classes expected: {len(class_names)}")

    print("Loading test dataset with forced class order...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        class_names=class_names,   # IMPORTANT: align labels to training order
        label_mode="int",
    )

    # Baseline was trained with inputs rescaled to [0,1] outside the model.
    # If the loaded model doesn't already include a Rescaling layer, apply it here.
    if not model_has_rescaling(model):
        print("Model has NO Rescaling layer -> applying Rescaling(1/255) in evaluation.")
        rescale = tf.keras.layers.Rescaling(1.0 / 255)
        test_ds = test_ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        print("Model already has Rescaling -> no extra normalization applied.")

    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    print("Predicting...")
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    y_prob = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    report_text = classification_report(y_true, y_pred, target_names=class_names)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    report_path = os.path.join(REPORT_DIR, "classification_report.txt")
    csv_path = os.path.join(REPORT_DIR, "per_class_metrics.csv")
    cm_path = os.path.join(CM_DIR, "confusion_matrix.png")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    pd.DataFrame(report_dict).transpose().to_csv(csv_path, index=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()

    print("\nEvaluation complete! Saved:")
    print(f"- {report_path}")
    print(f"- {csv_path}")
    print(f"- {cm_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.evaluate <model_path>")
    main(sys.argv[1])