<div align="center">

# 💓 Classification of Arrhythmia — ECG Data

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20ML%20Repository-blue?style=for-the-badge)](https://archive.ics.uci.edu/ml/datasets/Arrhythmia)
[![Best Recall](https://img.shields.io/badge/Best%20Recall-80.21%25-brightgreen?style=for-the-badge)](#-model-performance)
[![License](https://img.shields.io/badge/License-MIT-1abc9c?style=for-the-badge)](LICENSE)

> Detecting the **presence or absence of cardiac arrhythmia** and classifying ECG signal recordings into **16 clinical groups** using classical ML algorithms, SMOTE oversampling, and PCA dimensionality reduction.

</div>

---

## ⚠️ Medical Disclaimer

> **For educational and research purposes only.** This project is not a validated medical device and must not be used as a substitute for clinical ECG interpretation or professional medical diagnosis.

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [Dataset Overview](#-dataset-overview)
- [Class Distribution](#-class-distribution)
- [Methodology](#-methodology)
- [Model Performance](#-model-performance)
- [Key Findings](#-key-findings)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Tech Stack](#-tech-stack)
- [References](#-references)

---

## 🔬 About the Project

Electrocardiograms (ECGs) capture the electrical activity of the heart and serve as the primary diagnostic tool for detecting irregular heartbeats (**arrhythmias**). While isolated premature beats are often benign, sustained arrhythmias can lead to stroke, heart failure, or sudden cardiac arrest—making early automated screening valuable in preventive cardiology.

Manual ECG review at scale is slow and error-prone. In this project, I benchmark **six classical machine learning classifiers** on the UCI Arrhythmia dataset to distinguish normal sinus rhythm from 15 arrhythmia categories.

**Core Engineering Challenges Addressed:**
- **High Dimensionality ($p \approx n$):** 279 features across only 452 patient records.
- **Severe Class Imbalance:** Over 54% of records belong to a single class (*Normal*), while several arrhythmia subtypes contain fewer than 10 samples (or zero samples).
- **Multicollinearity & Missing Data:** Correlated signal measurements across 12 ECG leads and missing values in key intervals.

---

## 📊 Dataset Overview

| Property | Details |
|----------|---------|
| **Source** | [UCI Machine Learning Repository — Arrhythmia Dataset](https://archive.ics.uci.edu/ml/datasets/Arrhythmia) |
| **Samples** | 452 patient records |
| **Features** | 279 (patient demographics: age, sex, height, weight + 12-lead ECG attributes) |
| **Target Classes** | 16 (1 Normal + 12 Arrhythmia subtypes + 3 Unclassified/Absent groups) |
| **Missing Values** | Present (encoded as `?` in raw data, concentrated in the `J` angle feature) |

---

## 📋 Class Distribution

| Code | Clinical Class | Instances | Share |
|:----:|----------------|:---------:|:-----:|
| 01 | **Normal** | 245 | 54.2% |
| 02 | Ischemic Changes (Coronary Artery Disease) | 44 | 9.7% |
| 03 | Old Anterior Myocardial Infarction | 15 | 3.3% |
| 04 | Old Inferior Myocardial Infarction | 15 | 3.3% |
| 05 | Sinus Tachycardia | 13 | 2.9% |
| 06 | Sinus Bradycardia | 25 | 5.5% |
| 07 | Ventricular Premature Contraction (PVC) | 3 | 0.7% |
| 08 | Supraventricular Premature Contraction | 2 | 0.4% |
| 09 | Left Bundle Branch Block | 9 | 2.0% |
| 10 | Right Bundle Branch Block | 50 | 11.1% |
| 11 | 1° Atrioventricular Block | 0 | 0.0% |
| 12 | 2° Atrioventricular Block | 0 | 0.0% |
| 13 | 3° Atrioventricular Block | 0 | 0.0% |
| 14 | Left Ventricular Hypertrophy | 4 | 0.9% |
| 15 | Atrial Fibrillation or Flutter | 5 | 1.1% |
| 16 | Others (Unclassified) | 22 | 4.9% |
| | **Total** | **452** | **100%** |

> **Note:** Classes `11`, `12`, and `13` have zero recorded instances in the UCI dataset, leaving **13 active classes**. Ultra-rare classes (`07`, `08`, `14`, `15`) contain only 2–5 samples each, requiring low-neighbor oversampling (`k_neighbors=1`) or class grouping during cross-validation.

---

## ⚙️ Methodology

```text
Raw UCI Data (452 × 279)
        │
        ▼
  1. Data Preprocessing
     ├── Replace '?' placeholders & apply median imputation
     ├── Remove zero-variance / constant columns
     └── Standardize numerical features (StandardScaler)
        │
        ▼
  2. Exploratory Data Analysis (EDA)
     ├── Class imbalance & demographic profiling
     └── Inter-lead correlation & missingness inspection
        │
        ▼
  3. Train / Test Split & Resampling (Training Set Only)
     └── SMOTE Oversampling (prevents data leakage into test fold)
        │
        ▼
  4. Dimensionality Reduction
     └── Principal Component Analysis (PCA — 95% explained variance)
        │
        ▼
  5. Model Training & Evaluation
     ├── K-Nearest Neighbors (KNN)
     ├── Logistic Regression
     ├── Decision Tree
     ├── Linear SVC
     ├── Kernelized SVC (RBF)  ◄── Best Model
     └── Random Forest
```

---

## 📈 Model Performance

### Comparison: Baseline vs. SMOTE + PCA Pipeline

| Model | Baseline (No PCA) | With SMOTE + PCA | Key Takeaway |
|-------|:-----------------:|:----------------:|--------------|
| KNN Classifier | ~65.0% | ~72.0% | Major boost from lower-dimensional distance space |
| Logistic Regression | ~70.0% | ~75.0% | Stable linear baseline once multicollinearity is removed |
| Decision Tree | ~63.0% | ~68.0% | High variance; prone to overfitting minority classes |
| Linear SVC | ~72.0% | ~76.0% | Strong on majority classes (`Normal`, `RBBB`) |
| Random Forest | ~73.0% | ~78.0% | Strong ensemble balance across features |
| **Kernelized SVC (RBF)** ✅ | **~74.0%** | **80.21%** | **Best overall recall & generalization** |

> ✅ **Model Selection Rationale:** **Kernelized SVC (RBF) with SMOTE + PCA** achieved the highest recall (**80.21%**). In clinical screening tasks, **recall is prioritized over raw accuracy** to minimize false negatives (missed cardiac abnormalities).

![Model Comparison Results](Image/result.png)

---

## 🔍 Key Findings

1. **PCA Mitigates the Curse of Dimensionality:** Compressing 279 correlated ECG signal attributes into orthogonal principal components retaining 95% variance eliminated multicollinearity and improved every single classifier by **5–7 percentage points**.
2. **Leak-Free SMOTE Is Essential:** Applying SMOTE strictly after the train/test split enabled classifiers to learn decision boundaries for rare subtypes (such as LVH and Atrial Fibrillation) without inflating test metrics through synthetic leakage.
3. **RBF Kernels Outperform Tree Ensembles on Small Tabular Signal Data:** With fewer than 500 samples, RBF-kernel SVMs generalized better in continuous PCA space than axis-aligned splits from Decision Trees and Random Forests.

---

## 📁 Project Structure

```text
Classification-of-Arrhythmia-ECG/
│
├── 📂 Data/
│   ├── arrhythmia.data               # Raw UCI dataset (452 × 279)
│   └── arrhythmia.names              # Attribute & class metadata
│
├── 📂 Preprocessing and EDA/
│   ├── Data preprocessing.ipynb      # Missing value imputation, scaling, cleanup
│   └── EDA.ipynb                     # Class distributions, correlation heatmaps
│
├── 📂 Model/
│   └── oversampled and pca.ipynb     # SMOTE + PCA experiments & model benchmarks
│
├── 📂 Image/
│   └── result.png                    # Model comparison chart
│
├── 📂 1- Reports and presentations/  # Project report, slide deck, reference papers
│
├── final with pca.ipynb              # End-to-end consolidated pipeline (main notebook)
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Arpit-Phutela/arrhythmia-classification-ecg.git
cd arrhythmia-classification-ecg
```

### 2. Create a virtual environment & install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate          # Windows (PowerShell / CMD)

pip install -r requirements.txt
```

### 3. Run the notebooks

```bash
# Option A: Run the full end-to-end pipeline directly
jupyter notebook "final with pca.ipynb"

# Option B: Step through preprocessing and EDA separately
jupyter notebook "Preprocessing and EDA/Data preprocessing.ipynb"
jupyter notebook "Preprocessing and EDA/EDA.ipynb"
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **Data Manipulation** | NumPy, Pandas |
| **Machine Learning** | scikit-learn (SVC, Random Forest, Logistic Regression, KNN, Decision Tree, PCA) |
| **Imbalanced Learning** | imbalanced-learn (`SMOTE`) |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Jupyter Notebook |

---

## 📚 References

- **Dataset:** [UCI Machine Learning Repository — Arrhythmia Dataset](https://archive.ics.uci.edu/ml/datasets/Arrhythmia)
- **Original Paper:** Guvenir, H. A., Acar, B., Demiroz, G., & Cekin, A. (1997). *A Supervised Machine Learning Algorithm for Arrhythmia Analysis.* Computers in Cardiology, 24, 433–436.
- **SMOTE:** Chawla, N. V., et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* Journal of Artificial Intelligence Research, 16, 321–357.
- **Libraries:** [scikit-learn PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca) · [imbalanced-learn SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)

---

<div align="center">

Built by **[Arpit Phutela](https://github.com/Arpit-Phutela)**

⭐ Star this repository if you found it helpful!

</div>
