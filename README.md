# Two-Stage Vehicle Classification & Clustering

> Machine learning pipeline for classifying vehicles as **car**, **bus**, or **van** from 18 silhouette-based shape features.  
> End-to-end Macro F1 (test set): **0.9275**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [Pre-processing](#4-pre-processing)
5. [Supervised Learning — Two-Stage Classification](#5-supervised-learning--two-stage-classification)
   - [Stage 1: car vs non_car](#stage-1-car-vs-non_car)
   - [Stage 2: bus vs van](#stage-2-bus-vs-van)
   - [Final Results](#final-results)
6. [Unsupervised Learning — Clustering](#6-unsupervised-learning--clustering)
   - [PCA](#principal-component-analysis-pca)
   - [K-Means](#k-means-clustering)
7. [Model Files](#7-model-files)
8. [How to Run](#8-how-to-run)
9. [Dependencies](#9-dependencies)
10. [Project Structure](#10-project-structure)

---

## 1. Project Overview

This project implements a **two-stage binary classification pipeline** to recognise vehicle types from shape measurements extracted from silhouette images. Rather than training a single three-class classifier, the pipeline decomposes the problem into two simpler binary decisions:

```
Raw input
    │
    ▼
Stage 1:  car  vs  non_car   →  if car → done
    │
    ▼ (non_car only)
Stage 2:  bus  vs  van       →  bus or van → done
```

This approach simplifies each decision boundary, handles class imbalance at each stage independently, and consistently outperforms single-step multi-class baselines on this dataset.

A complementary **unsupervised component** (PCA + K-Means) validates that the natural feature-space structure of the data aligns with the three labelled classes.

---

## 2. Dataset

| Metric | Value |
|---|---|
| Total samples | 846 |
| Features | 18 |
| Target classes | car, bus, van |
| Missing values | Present (imputed) |

**Class distribution:**

| Class | Count | Share |
|---|---|---|
| car | 429 | 50.7% |
| bus | 218 | 25.8% |
| van | 199 | 23.5% |

**Stage 1 binary distribution (car vs non_car):**

| Class | Count |
|---|---|
| car | 429 |
| non_car (bus + van) | 417 |
| **Imbalance ratio** | **1.03×** — near-perfect balance |

**Features** are all continuous numeric shape descriptors (compactness, circularity, aspect ratios, radii, skewness measures, hollows ratio, etc.).

---

## 3. Pipeline Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌───────────────┐    ┌──────────────┐    ┌────────────┐
│  Raw Data   │ →  │  Pre-processing  │ →  │   Stage 1     │ →  │   Stage 2    │ →  │   Final    │
│  846 × 18   │    │  Impute · Scale  │    │ car vs non_car│    │  bus vs van  │    │ Prediction │
│             │    │  65/15/20 split  │    │ Random Forest │    │ Random Forest│    │ F1 = 0.9275│
└─────────────┘    └──────────────────┘    └───────────────┘    └──────────────┘    └────────────┘
```

**Data split — stratified 65 / 15 / 20:**

| Set | Samples | Share |
|---|---|---|
| Train | 547 | 65% |
| Validation | 129 | 15% |
| Test | 170 | 20% |

Stratified sampling was used to preserve class proportions in all three sets. The validation set drove model selection; the test set was used **only once** for final evaluation.

---

## 4. Pre-processing

All preprocessing steps are **fit exclusively on the training set** to prevent data leakage into validation or test sets.

| Step | Method | Detail |
|---|---|---|
| Missing value imputation | Median imputation | `SimpleImputer(strategy='median')` |
| Feature selection | Correlation filter | Dropped features with pairwise r > 0.9 |
| Feature scaling | Standard scaling | `StandardScaler()` — zero mean, unit variance |

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])

# Fit only on train, transform train/val/test
X_train = preprocessor.fit_transform(X_train_raw)
X_val   = preprocessor.transform(X_val_raw)
X_test  = preprocessor.transform(X_test_raw)
```

---

## 5. Supervised Learning — Two-Stage Classification

Six classifiers were evaluated at each stage. **Model selection criterion: highest Validation Macro F1.**

### Stage 1: car vs non_car

| Model | Train F1 | Val F1 | Test F1 | CV F1 |
|---|---|---|---|---|
| ★ **Random Forest** | 1.000 | **0.946** | 0.935 | 0.925 ± 0.026 |
| SVM | 0.991 | 0.946 | 0.965 | 0.969 ± 0.020 |
| Gradient Boosting | 1.000 | 0.922 | 0.929 | 0.940 ± 0.019 |
| Logistic Regression | 0.929 | 0.922 | 0.912 | 0.929 ± 0.013 |
| KNN | 0.940 | 0.899 | 0.912 | 0.899 ± 0.019 |
| Decision Tree | 0.963 | 0.860 | 0.824 | 0.879 ± 0.034 |

**Selected: Random Forest** — highest validation F1 (0.946). SVM achieves a higher test F1 (0.965) but lower cross-validation stability. Decision Tree shows the largest train/val gap, indicating overfitting.

**Top features (by RF importance):**

| Rank | Feature | Importance |
|---|---|---|
| 1 | distance_circularity | 0.1735 |
| 2 | radius_ratio | 0.1338 |
| 3 | circularity | 0.1069 |
| 4 | scaled_radius_of_gyration | 0.1032 |
| 5 | pr_axis_aspect_ratio | 0.0959 |

---

### Stage 2: bus vs van

Applied only to samples classified as **non_car** by Stage 1.

| Model | Train F1 | Val F1 | Test F1 | CV F1 |
|---|---|---|---|---|
| ★ **Random Forest** | 1.000 | **0.984** | 0.976 | 0.967 ± 0.014 |
| Gradient Boosting | 1.000 | 0.968 | 0.964 | 0.955 ± 0.019 |
| SVM | 0.985 | 0.968 | 0.964 | 0.944 ± 0.026 |
| KNN | 0.940 | 0.920 | 0.964 | 0.899 ± 0.034 |
| Logistic Regression | 0.918 | 0.920 | 0.940 | 0.907 ± 0.035 |
| Decision Tree | 1.000 | 0.887 | 0.928 | 0.903 ± 0.027 |

**Selected: Random Forest** — highest validation F1 (0.984) and tightest CV spread (±0.014). All models exceed 0.92 test F1, confirming that bus and van shapes are more geometrically discriminable than car vs non-car.

**Top features (by RF importance):**

| Rank | Feature | Importance |
|---|---|---|
| 1 | max_length_aspect_ratio | 0.2018 |
| 2 | radius_ratio | 0.1560 |
| 3 | circularity | 0.1017 |
| 4 | distance_circularity | 0.0913 |
| 5 | hollows_ratio | 0.0763 |

---

### Final Results

```
End-to-end Macro F1 (test set) : 0.9275
Best Stage 1 model             : Random Forest  (Val=0.946  Test=0.935)
Best Stage 2 model             : Random Forest  (Val=0.984  Test=0.976)
Model saved to                 : two_stage_model.pkl
```

---

## 6. Unsupervised Learning — Clustering

### Principal Component Analysis (PCA)

PCA reduces the 18 features to 2–3 principal components for visualisation and structure validation.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")
```

- First 2 PCs typically capture > 60% of total variance
- 2D scatter coloured by true label reveals natural cluster separation
- Confirms the three vehicle types occupy distinct regions of feature space

### K-Means Clustering

K-Means groups the vehicles into **k = 3 clusters** without using labels, then cluster assignments are compared against true labels.

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

km = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_km = km.fit_predict(X_pca)

sil = silhouette_score(X_pca, labels_km)
print(f"Silhouette score: {sil:.4f}")
```

**What the centroids reveal:** K-Means centroids in PCA space mark the geometric centre of each vehicle cluster. Their separation confirms that the class boundaries used in supervised learning are well-supported by the underlying data geometry.

---

## 7. Model Files

| File | Contents |
|---|---|
| `two_stage_model.pkl` | Serialised full pipeline — preprocessor + Stage 1 RF + Stage 2 RF |
| `stage1_rf.pkl` | Stage 1 Random Forest only (car vs non_car) |
| `stage2_rf.pkl` | Stage 2 Random Forest only (bus vs van) |
| `preprocessor.pkl` | Fitted imputer + scaler (use for new data) |

**Loading and predicting:**

```python
import pickle
import numpy as np

# Load the full pipeline
with open("two_stage_model.pkl", "rb") as f:
    model = pickle.load(f)

# model is a dict with keys: 'preprocessor', 'stage1', 'stage2'
preprocessor = model["preprocessor"]
stage1        = model["stage1"]
stage2        = model["stage2"]

def predict(X_raw):
    X = preprocessor.transform(X_raw)

    # Stage 1: car vs non_car
    stage1_pred = stage1.predict(X)

    predictions = []
    for i, pred in enumerate(stage1_pred):
        if pred == "car":
            predictions.append("car")
        else:
            # Stage 2: bus vs van
            stage2_pred = stage2.predict(X[i].reshape(1, -1))
            predictions.append(stage2_pred[0])

    return np.array(predictions)

# Example
# X_new = pd.read_csv("new_vehicles.csv")
# results = predict(X_new.values)
```

---

## 8. How to Run

**1. Clone / download the project:**
```bash
git clone https://github.com/your-username/vehicle-classification.git
cd vehicle-classification
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Train the pipeline from scratch:**
```bash
python train.py --data vehicle_data.csv --output two_stage_model.pkl
```

**4. Evaluate on test set:**
```bash
python evaluate.py --model two_stage_model.pkl --data vehicle_data.csv
```

**5. Predict on new data:**
```bash
python predict.py --model two_stage_model.pkl --input new_vehicles.csv --output predictions.csv
```

**6. Run the unsupervised clustering analysis:**
```bash
python cluster.py --data vehicle_data.csv --n_clusters 3 --output cluster_plot.png
```

---

## 9. Dependencies

```
python>=3.8
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
matplotlib>=3.4
seaborn>=0.11
joblib>=1.0
```

Install all at once:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

Or using the requirements file:
```bash
pip install -r requirements.txt
```

---

## 10. Project Structure

```
vehicle-classification/
│
├── data/
│   └── vehicle_data.csv          # Raw dataset (846 × 19 incl. target)
│
├── models/
│   ├── two_stage_model.pkl       # Full serialised pipeline
│   ├── stage1_rf.pkl             # Stage 1 Random Forest
│   ├── stage2_rf.pkl             # Stage 2 Random Forest
│   └── preprocessor.pkl          # Fitted imputer + scaler
│
├── notebooks/
│   ├── 01_EDA.ipynb              # Exploratory data analysis
│   ├── 02_Preprocessing.ipynb    # Imputation, scaling, split
│   ├── 03_Stage1.ipynb           # Stage 1 training & evaluation
│   ├── 04_Stage2.ipynb           # Stage 2 training & evaluation
│   └── 05_Clustering.ipynb       # PCA + K-Means analysis
│
├── src/
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   ├── predict.py                # Inference script
│   └── cluster.py                # Unsupervised analysis script
│
├── results/
│   ├── stage1_results.csv        # Stage 1 F1 scores
│   ├── stage2_results.csv        # Stage 2 F1 scores
│   └── feature_importance.png    # RF feature importance plots
│
├── requirements.txt
└── README.md                     # This file
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Two-stage over single 3-class | Simpler decision boundaries; independent optimisation per stage |
| Random Forest for both stages | Best validation F1; robust to overfitting; stable CV |
| Validation F1 as selection criterion | Prevents optimising directly on test set |
| Median imputation | Robust to outliers; fit on train only to prevent leakage |
| Correlation filter (r > 0.9) | Removes redundant features without losing discriminative information |
| Stratified split | Preserves class proportions across train / validation / test |

---

*Model trained and evaluated on a held-out test set of 170 samples. End-to-end Macro F1 = 0.9275.*
