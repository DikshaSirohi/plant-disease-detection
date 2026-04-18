import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
 
from src.config import TRAIN_DIR, VAL_DIR, IMG_SIZE, BATCH_SIZE, SEED, IMPROVED_MODEL_PATH
 
# ============================================================
# Resume settings
# Set RESUME_STAGE2 = True to continue fine-tuning from a
# saved checkpoint instead of training from scratch.
# ============================================================
RESUME_STAGE2        = True   # True = resume fine-tuning only
STAGE2_TOTAL_EPOCHS  = 10     # total fine-tuning epochs planned
STAGE2_INITIAL_EPOCH = 1      # which epoch to resume from (0-indexed)
 
 
def make_datasets():
    """Load train and validation datasets and convert labels to one-hot."""
    train_sparse = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        label_mode="int",
    )
    val_sparse = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="int",
    )
 
    class_names = train_sparse.class_names
    num_classes = len(class_names)
 
    def to_one_hot(x, y):
        y = tf.cast(tf.squeeze(y), tf.int32)
        y = tf.ensure_shape(y, [None])
        return x, tf.one_hot(y, depth=num_classes)
 
    train = train_sparse.map(to_one_hot, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val   = val_sparse.map(to_one_hot, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    train_sparse = train_sparse.prefetch(tf.data.AUTOTUNE)
 
    return train_sparse, train, val, class_names, num_classes
 
 
def compute_class_weights(train_sparse):
    """Compute per-class weights to handle class imbalance."""
    y_all = np.concatenate([y.numpy() for _, y in train_sparse], axis=0)
    classes = np.unique(y_all)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_all)
    return {int(c): float(w) for c, w in zip(classes, weights)}
 
 
def build_callbacks():
    """ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau."""
    base        = IMPROVED_MODEL_PATH.replace(".keras", "")
    best_path   = base + "_best.keras"
    last_path   = base + "_last.keras"
    interrupt_path = base + "_interrupt.keras"
 
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            best_path, save_best_only=True,
            monitor="val_accuracy", mode="max", verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            last_path, save_best_only=False, verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            patience=4, restore_best_weights=True,
            monitor="val_accuracy", mode="max", verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            patience=2, factor=0.3, min_lr=1e-6,
            monitor="val_loss", mode="min", verbose=1,
        ),
    ]
    return callbacks, best_path, last_path, interrupt_path
 
 
def find_backbone(model: keras.Model) -> keras.Model:
    """Find the EfficientNet backbone inside the model."""
    for layer in model.layers:
        if isinstance(layer, keras.Model) and "efficientnet" in layer.name.lower():
            return layer
    raise ValueError("Could not find EfficientNet backbone inside the model.")
 
 
def unfreeze_top_30_percent(model: keras.Model):
    """Unfreeze the top 30% of the backbone layers for fine-tuning."""
    backbone = find_backbone(model)
    backbone.trainable = True
    fine_tune_at = int(len(backbone.layers) * 0.7)  # keep bottom 70% frozen
    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False
    return model
 
 
def main():
    os.makedirs(os.path.dirname(IMPROVED_MODEL_PATH), exist_ok=True)
 
    train_sparse, train_ds, val_ds, class_names, num_classes = make_datasets()
    class_weight = compute_class_weights(train_sparse)
    print("Classes:", num_classes)
 
    callbacks, best_path, last_path, interrupt_path = build_callbacks()
    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
 
    model = None
    try:
        if RESUME_STAGE2:
            print("\nResuming Stage 2 (fine-tuning) from:", IMPROVED_MODEL_PATH)
            model = tf.keras.models.load_model(IMPROVED_MODEL_PATH, compile=False)
            model = unfreeze_top_30_percent(model)
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                loss=loss_fn,
                metrics=["accuracy"],
            )
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=STAGE2_TOTAL_EPOCHS,
                initial_epoch=STAGE2_INITIAL_EPOCH,
                class_weight=class_weight,
                callbacks=callbacks,
            )
            model.save(IMPROVED_MODEL_PATH)
            print("Saved model to:", IMPROVED_MODEL_PATH)
            return
 
        raise RuntimeError("Set RESUME_STAGE2=True or add Stage 1 training code.")
 
    except KeyboardInterrupt:
        print("\nInterrupted — saving snapshot...")
        if model is not None:
            model.save(interrupt_path)
            print("Snapshot saved to:", interrupt_path)
 
 
if __name__ == "__main__":
    main()