# Processing of Plant Leaf Images for Disease Detection using Deep Learning

## Overview
This project classifies plant leaf images into **38 classes** (healthy + diseases) using Deep Learning with Transfer Learning (TensorFlow/Keras).

Two models were implemented:
- A **Baseline model**
- An **Improved model**

Both models were evaluated using accuracy, per-class metrics, and confusion matrices.  
Grad-CAM is also included to visualize which part of the leaf the model focuses on.

---

## Dataset

**Dataset Name:** New Plant Diseases Dataset (Augmented)  
**Total Classes:** 38  

**Dataset Source (Kaggle):**  
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

### Dataset Split Used

- `train/` → used for training
- `valid/` → used for validation during training
- `test/` → manually created from validation (**20% per class**) for final evaluation

⚠️ The dataset is **NOT included** in this GitHub repository.  
Please download it from Kaggle and update dataset paths in `src/config.py`.

---

## Models

### Baseline Model
- Backbone: **MobileNetV2 (ImageNet)**
- Backbone frozen (no fine-tuning)
- GlobalAveragePooling + Dropout(0.2) + Dense(38)
- Trained for 5 epochs
- Test Accuracy: ~93%
- Saved as: `models/baseline.keras`

### Improved Model
- Backbone: **EfficientNetB0 (ImageNet)**
- Stronger data augmentation
- Label smoothing: 0.05
- Class weights used
- Two-stage training:
  1. Train classification head
  2. Fine-tune top 30% layers (low learning rate)
- Best Test Accuracy: ~99%
- Saved as: `models/improved_best.keras`

---

## Baseline vs Improved Comparison

| Feature | Baseline | Improved |
|----------|----------|----------|
| Model | MobileNetV2 | EfficientNetB0 |
| Fine-tuning | No | Yes |
| Augmentation | Basic | Stronger |
| Label smoothing | No | Yes (0.05) |
| Class weights | No | Yes |
| Test Accuracy | ~93% | ~99% |

---

## Evaluation Outputs

Results are saved inside:

- `final/baseline/`
  - classification_report_baseline.txt
  - per_class_metrics_baseline.csv
  - confusion_matrix_baseline.png

- `final/improved/`
  - classification_report_improved.txt
  - per_class_metrics_improved.csv
  - confusion_matrix_improved.png

Evaluation script:

python src/evaluate.py


---

## Grad-CAM

Grad-CAM generates a heatmap showing which region of the leaf image influenced the model’s prediction.

Script:

python src/gradcam.py


---

## Project Structure


app/
app.py

src/
baseline_train.py
train_improved.py
evaluate.py
evaluate_compare.py
gradcam.py
error_analysis.py
config.py
make_test_split.py
save_class_names.py
test_evaluation.py

models/
baseline.keras
improved_best.keras
class_names.json

final/
baseline/
improved/


---

## How to Run

### 1) Create Virtual Environment (Python 3.10)


python -m venv plant_env


Activate:

Windows:

plant_env\Scripts\activate


Mac/Linux:

source plant_env/bin/activate


### 2) Install Requirements


pip install -r requirements.txt


### 3) Update Dataset Path

Edit:

src/config.py


### 4) Train Models

Baseline:

python src/baseline_train.py


Improved:

python src/train_improved.py


### 5) Evaluate


python src/evaluate.py


---

## Notes
- Dataset is not uploaded to GitHub.
- Virtual environment (`plant_env`) is not uploaded.