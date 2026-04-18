import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
 
from src.config import TEST_DIR, IMG_SIZE, GRADCAM_DIR
 
 
def load_one_image_from_test():
    """Walk TEST_DIR and return the first image found as a float32 tensor (0..255)."""
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for root, _, files in os.walk(TEST_DIR):
        for fn in files:
            if fn.lower().endswith(exts):
                path = os.path.join(root, fn)
                raw  = tf.io.read_file(path)
                img  = tf.image.decode_image(raw, channels=3, expand_animations=False)
                img  = tf.image.resize(img, IMG_SIZE)
                img  = tf.cast(img, tf.float32)  # values 0..255
                return path, img
    raise FileNotFoundError(f"No images found under TEST_DIR: {TEST_DIR}")
 
 
def overlay_heatmap(img_rgb, heatmap, alpha=0.40):
    """Blend the Grad-CAM heatmap over the original image."""
    heat = tf.image.resize(heatmap[..., None], (img_rgb.shape[0], img_rgb.shape[1]))
    heat = tf.squeeze(heat).numpy()
 
    heat_color = (plt.get_cmap("jet")(heat)[:, :, :3] * 255).astype(np.float32)
    overlay    = np.clip((1 - alpha) * img_rgb + alpha * heat_color, 0, 255).astype(np.uint8)
    return overlay
 
 
def find_backbone_and_head(model: tf.keras.Model):
    """
    Find the backbone (nested Model with 4D output) and the
    classification head layers (GAP → Dropout → Dense).
    """
    backbone = None
    backbone_index = None
 
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.Model):
            try:
                if len(layer.output_shape) == 4:
                    backbone = layer
                    backbone_index = i
                    break
            except Exception:
                pass
 
    if backbone is None:
        raise ValueError("Could not find backbone (nested model with 4D output).")
 
    gap, dropout, dense = None, None, None
    for layer in model.layers[backbone_index + 1:]:
        if gap is None and isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            gap = layer
        elif dropout is None and isinstance(layer, tf.keras.layers.Dropout):
            dropout = layer
        elif isinstance(layer, tf.keras.layers.Dense):
            dense = layer
 
    if gap is None or dense is None:
        raise ValueError("Could not find GAP and/or Dense head layers.")
 
    return backbone, gap, dropout, dense
 
 
def main(model_path: str, out_name="gradcam_result.png"):
    os.makedirs(GRADCAM_DIR, exist_ok=True)
 
    print("Loading model:", model_path)
    model    = tf.keras.models.load_model(model_path)
    backbone, gap, dropout, dense = find_backbone_and_head(model)
 
    img_path, img = load_one_image_from_test()
    print("Image:", img_path)
 
    img_batch = tf.expand_dims(img, axis=0)
 
    # Get predicted class
    preds    = model(img_batch, training=False).numpy()[0]
    pred_idx = int(np.argmax(preds))
    pred_conf = float(preds[pred_idx])
 
    # Grad-CAM: compute gradient of predicted class score
    # w.r.t. the backbone's output feature map
    with tf.GradientTape() as tape:
        feature_map = backbone(img_batch, training=False)
        tape.watch(feature_map)
        x = gap(feature_map)
        if dropout is not None:
            x = dropout(x, training=False)
        class_score = dense(x)[:, pred_idx]
 
    grads        = tape.gradient(class_score, feature_map)      # (1, h, w, c)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))        # (c,)
    heatmap      = tf.squeeze(feature_map[0] @ pooled_grads[..., tf.newaxis])
    heatmap      = tf.maximum(heatmap, 0)
    heatmap      = (heatmap / (tf.reduce_max(heatmap) + 1e-8)).numpy()
 
    overlay   = overlay_heatmap(img.numpy(), heatmap)
    out_path  = os.path.join(GRADCAM_DIR, out_name)
 
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.title("Original");                    plt.imshow(img.numpy().astype(np.uint8)); plt.axis("off")
    plt.subplot(1, 3, 2); plt.title("Heatmap");                     plt.imshow(heatmap, cmap="jet");          plt.axis("off")
    plt.subplot(1, 3, 3); plt.title(f"Overlay (conf={pred_conf:.3f})"); plt.imshow(overlay);                 plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
 
    print("Grad-CAM saved to:", out_path)
 
 
if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.gradcam <model_path>")
    main(sys.argv[1])