# ============================================================
# PART 0: Imports
# ============================================================
import os
import json
import tensorflow as tf
from tensorflow import keras
 
from src.config import TRAIN_DIR, VAL_DIR, IMG_SIZE, BATCH_SIZE, SEED, BASELINE_MODEL_PATH
 
# ============================================================
# PART 1: Basic path checks
# ============================================================
print("Train folder exists:", os.path.exists(TRAIN_DIR))
print("Valid folder exists:", os.path.exists(VAL_DIR))
 
# ============================================================
# PART 2: Dataset statistics
# ============================================================
counts = {}
for cls in os.listdir(TRAIN_DIR):
    cls_path = os.path.join(TRAIN_DIR, cls)
    if os.path.isdir(cls_path):
        counts[cls] = len([
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
 
print("Total classes:", len(counts))
print("Total training images:", sum(counts.values()))
 
# ============================================================
# PART 3: Create datasets
# ============================================================
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)
 
valid_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)
 
class_names = train_ds.class_names
num_classes = len(class_names)
print("Confirmed classes:", num_classes)
 
# ============================================================
# PART 4: Augmentation (applied only during training)
# ============================================================
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.10),
    keras.layers.RandomZoom(0.10),
    keras.layers.RandomContrast(0.10),
], name="augmentation")
 
train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE
)
 
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)
 
# ============================================================
# PART 5: Build model
# Normalization (Rescaling) is inside the model so it's always
# applied correctly at inference time too.
# ============================================================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # freeze pretrained layers
 
inputs  = keras.Input(shape=(224, 224, 3), name="image")
x       = keras.layers.Rescaling(1.0 / 255, name="rescale")(inputs)
x       = base_model(x, training=False)
x       = keras.layers.GlobalAveragePooling2D()(x)
x       = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
 
model = keras.Model(inputs, outputs, name="baseline_mobilenetv2")
 
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
 
model.summary()
 
# ============================================================
# PART 6: Train
# ============================================================
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=5,
)
 
# ============================================================
# PART 7: Save model and class names
# ============================================================
os.makedirs("models", exist_ok=True)
model.save(BASELINE_MODEL_PATH)
print("Saved baseline model ->", BASELINE_MODEL_PATH)
 
# Save class names so evaluate.py and app.py use the same label order
class_names_path = os.path.join("models", "class_names.json")
with open(class_names_path, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)
print("Saved class names ->", class_names_path)