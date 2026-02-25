# ============================================================
# PART 0: Imports
# ============================================================
import os
import tensorflow as tf
from tensorflow import keras

# ============================================================
# PART 1: Dataset Path & Basic Checks (run once)
# ============================================================
dataset_path = r"C:\NIET\Projects\PlantDiseaseProject\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)"

train_dir = os.path.join(dataset_path, "train")
valid_dir = os.path.join(dataset_path, "valid")

print("Dataset path exists:", os.path.exists(dataset_path))
print("Train folder exists:", os.path.exists(train_dir))
print("Valid folder exists:", os.path.exists(valid_dir))

# ============================================================
# PART 2: Dataset Statistics (class count & imbalance check)
# ============================================================
counts = {}

for cls in os.listdir(train_dir):
    cls_path = os.path.join(train_dir, cls)
    if os.path.isdir(cls_path):
        counts[cls] = len([
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

print("Total classes:", len(counts))
print("Total training images:", sum(counts.values()))

print("\nSmallest classes:")
for k in sorted(counts, key=counts.get)[:10]:
    print(k, counts[k])

# ============================================================
# PART 3: Create Train & Validation Datasets
# ============================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("\nConfirmed classes:", num_classes)

# ============================================================
# PART 4: Preprocessing (Augmentation ONLY)
# NOTE: Normalization is moved inside the model for reproducibility.
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
# PART 5: Sanity Check (VERY IMPORTANT)
# ============================================================
for images, labels in train_ds.take(1):
    print("\nTrain batch shape:", images.shape)
    print("Label shape:", labels.shape)

# ============================================================
# PART 6: Build Model (Transfer Learning - MobileNetV2)
# ============================================================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3), name="image")
x = keras.layers.Rescaling(1.0 / 255, name="rescale_1_over_255")(inputs)  # <-- FIX: normalization inside model
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs, name="baseline_mobilenetv2")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# PART 7: Train the Model (Initial Training)
# ============================================================
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=5
)

# ============================================================
# PART 8: Save Model (ensure folder exists)
# ============================================================
os.makedirs("models", exist_ok=True)
model.save(r"models\baseline.keras")
print("Baseline model saved successfully -> models\\baseline.keras")