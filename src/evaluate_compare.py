import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

from src.config import TEST_DIR, IMG_SIZE, BATCH_SIZE, OUT_DIR


def load_test_dataset():
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="int",
    )
    class_names = test_ds.class_names
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    return test_ds, y_true, class_names


def save_confusion_matrix(cm, class_names, out_path, title):
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def evaluate_model(model_path: str, test_ds, y_true, class_names, tag: str):
    model = tf.keras.models.load_model(model_path)
    y_prob = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = report_dict["macro avg"]["f1-score"]
    weighted_f1 = report_dict["weighted avg"]["f1-score"]

    cm = confusion_matrix(y_true, y_pred)

    # Output folders per model
    report_dir = os.path.join(OUT_DIR, "reports", tag)
    cm_dir = os.path.join(OUT_DIR, "confusion_matrices", tag)
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)

    # Save report files
    with open(os.path.join(report_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    pd.DataFrame(report_dict).transpose().to_csv(
        os.path.join(report_dir, "per_class_metrics.csv"), index=True
    )

    save_confusion_matrix(
        cm,
        class_names,
        os.path.join(cm_dir, "confusion_matrix.png"),
        title=f"Confusion Matrix: {tag}",
    )

    # Also save predictions for deeper analysis later
    pred_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "true_name": [class_names[i] for i in y_true],
        "pred_name": [class_names[i] for i in y_pred],
        "confidence": np.max(y_prob, axis=1),
    })
    pred_df.to_csv(os.path.join(report_dir, "predictions.csv"), index=False)

    summary = {
        "tag": tag,
        "model_path": model_path,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    print("\nSaved outputs for:", tag)
    print("-", os.path.join(report_dir, "classification_report.txt"))
    print("-", os.path.join(report_dir, "per_class_metrics.csv"))
    print("-", os.path.join(report_dir, "predictions.csv"))
    print("-", os.path.join(cm_dir, "confusion_matrix.png"))

    return summary


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage:\n"
            "  python -m src.evaluate_compare <model1_path>\n"
            "  python -m src.evaluate_compare <model1_path> <model2_path>\n"
        )

    model_paths = sys.argv[1:]
    test_ds, y_true, class_names = load_test_dataset()

    summaries = []
    if len(model_paths) == 1:
        summaries.append(evaluate_model(model_paths[0], test_ds, y_true, class_names, tag="model"))
    else:
        summaries.append(evaluate_model(model_paths[0], test_ds, y_true, class_names, tag="baseline"))
        summaries.append(evaluate_model(model_paths[1], test_ds, y_true, class_names, tag="improved"))

        # Save comparison table
        comp_df = pd.DataFrame(summaries)
        out_path = os.path.join(OUT_DIR, "reports", "model_comparison.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        comp_df.to_csv(out_path, index=False)

        print("\nModel comparison saved:")
        print("-", out_path)

        print("\nComparison (quick view):")
        print(comp_df[["tag", "accuracy", "macro_f1", "weighted_f1"]])

    print("\nDone.")


if __name__ == "__main__":
    main()