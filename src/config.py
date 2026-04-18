import os
 
# ============================================================
# Dataset paths — update DATASET_ROOT to your local folder
# ============================================================
DATASET_ROOT = r"C:\NIET\Projects\PlantDiseaseProject\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)"
 
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR   = os.path.join(DATASET_ROOT, "valid")
TEST_DIR  = os.path.join(DATASET_ROOT, "test")
 
# ============================================================
# Training settings
# ============================================================
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
SEED       = 1337
 
# ============================================================
# Model save paths
# ============================================================
BASELINE_MODEL_PATH = os.path.join("models", "baseline.keras")
IMPROVED_MODEL_PATH = os.path.join("models", "improved.keras")
 
# ============================================================
# Output directories
# ============================================================
OUT_DIR     = "outputs"
REPORT_DIR  = os.path.join(OUT_DIR, "reports")
CM_DIR      = os.path.join(OUT_DIR, "confusion_matrices")
MIS_DIR     = os.path.join(OUT_DIR, "misclassified")
GRADCAM_DIR = os.path.join(OUT_DIR, "gradcam")