import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight

from src.config import TRAIN_DIR, VAL_DIR, IMG_SIZE, BATCH_SIZE, SEED, IMPROVED_MODEL_PATH


# =========================
# Resume settings
# =========================
RESUME_STAGE2 = True              # True = resume fine-tuning only
STAGE2_TOTAL_EPOCHS = 10          # you planned 10 epochs for stage 2
STAGE2_INITIAL_EPOCH = 1          # resume from stage2 epoch 2 (0->epoch1 done, 1->epoch2)


def make_datasets():
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
        y = tf.ensure_shape(y, [None])  # (batch,)
        y_oh = tf.one_hot(y, depth=num_classes)
        y_oh = tf.ensure_shape(y_oh, [None, num_classes])  # (batch, C)
        return x, y_oh

    train = train_sparse.map(to_one_hot, num_parallel_calls=tf.data.AUTOTUNE)
    val = val_sparse.map(to_one_hot, num_parallel_calls=tf.data.AUTOTUNE)

    train_sparse = train_sparse.prefetch(tf.data.AUTOTUNE)
    train = train.prefetch(tf.data.AUTOTUNE)
    val = val.prefetch(tf.data.AUTOTUNE)

    return train_sparse, train, val, class_names, num_classes


def compute_class_weights(train_sparse):
    y_all = []
    for _, y in train_sparse:
        y = tf.cast(tf.squeeze(y), tf.int32)
        y_all.append(y.numpy())
    y_all = np.concatenate(y_all, axis=0)

    classes = np.unique(y_all)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_all)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def build_callbacks():
    base = IMPROVED_MODEL_PATH
    if base.endswith(".keras"):
        best_path = base.replace(".keras", "_best.keras")
        last_path = base.replace(".keras", "_last.keras")
        interrupt_path = base.replace(".keras", "_interrupt.keras")
    else:
        best_path = base + "_best.keras"
        last_path = base + "_last.keras"
        interrupt_path = base + "_interrupt.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            best_path,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            last_path,
            save_best_only=False,  # always overwrite "last"
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=4,
            restore_best_weights=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=2,
            factor=0.3,
            min_lr=1e-6,
            monitor="val_loss",
            mode="min",
            verbose=1,
        ),
    ]
    return callbacks, best_path, last_path, interrupt_path


def find_backbone(model: keras.Model) -> keras.Model:
    # EfficientNetB0 is a nested keras.Model inside the full model
    for layer in model.layers:
        if isinstance(layer, keras.Model) and "efficientnet" in layer.name.lower():
            return layer
    raise ValueError("Could not find EfficientNet backbone inside loaded model.")


def apply_stage2_unfreeze(model: keras.Model):
    base_model = find_backbone(model)
    base_model.trainable = True

    fine_tune_at = int(len(base_model.layers) * 0.7)  # unfreeze top 30%
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    return model


def main():
    os.makedirs(os.path.dirname(IMPROVED_MODEL_PATH), exist_ok=True)

    train_sparse, train_ds, val_ds, class_names, num_classes = make_datasets()
    class_weight = compute_class_weights(train_sparse)

    print("Classes:", num_classes)
    print("Resume model path:", IMPROVED_MODEL_PATH)

    callbacks, best_path, last_path, interrupt_path = build_callbacks()
    print("Will save BEST to:", best_path)
    print("Will save LAST to:", last_path)
    print("Will save INTERRUPT snapshot to:", interrupt_path)

    # Label smoothing
    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=0.05)

    model = None
    try:
        if RESUME_STAGE2:
            print("\n✅ RESUMING STAGE 2 from improved.keras")

            # >>> THIS is the resume line <<<
            model = tf.keras.models.load_model(IMPROVED_MODEL_PATH, compile=False)

            # Re-apply your Stage 2 unfreeze policy
            model = apply_stage2_unfreeze(model)

            # Compile for fine-tuning
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                loss=loss_fn,
                metrics=["accuracy"],
            )

            print(f"Starting Stage 2 from initial_epoch={STAGE2_INITIAL_EPOCH} (i.e., Stage 2 Epoch 2)")
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=STAGE2_TOTAL_EPOCHS,
                initial_epoch=STAGE2_INITIAL_EPOCH,
                class_weight=class_weight,
                callbacks=callbacks,
            )

            model.save(IMPROVED_MODEL_PATH)
            print("\nSaved resumed model to:", IMPROVED_MODEL_PATH)
            return

        # If you ever want full training again, put your Stage1+Stage2 code here.
        raise RuntimeError("RESUME_STAGE2 is False, but full training code not included in this resume script.")

    except KeyboardInterrupt:
        print("\n\nInterrupted (Ctrl+C) ✅")
        if model is not None:
            print("Saving snapshot to:", interrupt_path)
            model.save(interrupt_path)
            print("Saved snapshot ✅")
        print("\nCheckpoints:")
        print("BEST:", best_path)
        print("LAST:", last_path)
        print("INTERRUPT:", interrupt_path)


if __name__ == "__main__":
    main()