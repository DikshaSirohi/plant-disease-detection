import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import classification_report

# ------------------------------------------------
# 1. Load trained model
# ------------------------------------------------
MODEL_PATH = r"C:\NIET\Projects\plant_disease_model.keras"
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully:", MODEL_PATH)

# ------------------------------------------------
# 2. Dataset paths (your dataset has train/valid, not test)
#    Your folder is nested like:
#    ...\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train
#    ...\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid
# ------------------------------------------------
DATASET_ROOT = (
    r"C:\NIET\Projects\PlantDiseaseProject\New Plant Diseases Dataset(Augmented)"
    r"\New Plant Diseases Dataset(Augmented)"
)
EVAL_SPLIT_FOLDER = "valid"   # use "valid" as test/evaluation set

eval_dir = os.path.join(DATASET_ROOT, EVAL_SPLIT_FOLDER)

print("DATASET_ROOT:", DATASET_ROOT)
print("eval_dir:", eval_dir)
print("exists DATASET_ROOT?", os.path.exists(DATASET_ROOT))
print("exists eval_dir?", os.path.exists(eval_dir))

if not os.path.exists(DATASET_ROOT):
    raise FileNotFoundError(
        f"DATASET_ROOT not found.\nGot: {DATASET_ROOT}\n"
        f"Fix the path to the folder that contains 'train' and 'valid'."
    )

if not os.path.exists(eval_dir):
    print("Folders inside DATASET_ROOT:", os.listdir(DATASET_ROOT))
    raise FileNotFoundError(
        f"Evaluation folder not found.\nExpected: {eval_dir}\n"
        f"Available folders above were printed. Use one of them (train/valid)."
    )

# ------------------------------------------------
# 3. Create evaluation dataset
# ------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

raw_eval_ds = tf.keras.utils.image_dataset_from_directory(
    eval_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# IMPORTANT: Save class names BEFORE mapping (mapping can drop this attribute)
class_names = raw_eval_ds.class_names
print("Number of classes:", len(class_names))
print("Classes:", class_names)

# ------------------------------------------------
# 4. Preprocessing (same as training)
# ------------------------------------------------
normalization = keras.layers.Rescaling(1.0 / 255)

eval_ds = raw_eval_ds.map(
    lambda x, y: (normalization(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# ------------------------------------------------
# 5. Evaluate on evaluation set
# ------------------------------------------------
results = model.evaluate(eval_ds, verbose=1)

# Handle both cases: model compiled with (loss, accuracy) or more metrics
eval_loss = results[0]
eval_acc = None
if len(results) >= 2:
    eval_acc = results[1]

print(f"\nEval Loss: {eval_loss:.4f}")
if eval_acc is not None:
    print(f"Eval Accuracy: {eval_acc * 100:.2f}%")

# ------------------------------------------------
# 6. Per-class report
# ------------------------------------------------
# True labels
y_true = np.concatenate([y.numpy() for _, y in eval_ds], axis=0)

# Predicted labels
y_prob = model.predict(eval_ds, verbose=1)
y_pred = np.argmax(y_prob, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))