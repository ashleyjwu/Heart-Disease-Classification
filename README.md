# 🫀 Heart Disease Classification

This project explores multiple machine learning techniques to classify whether a patient has heart disease based on medical attributes. The work is based on the **UCI Heart Disease dataset**, focusing on preprocessing, supervised learning, dimensionality reduction, and clustering methods.

---

## Overview

The goal of this project is to evaluate and compare various machine learning models on a real-world medical dataset. It demonstrates how to design, optimize, and interpret classification and clustering models using **Scikit-learn**.

---

## Dataset

**Source:** UCI Heart Disease Dataset  
**Target Variable:** `sick` (True = Disease, False = No Disease)

**Number of Features:** 14  
**Key Attributes:**
- `age` — Patient age in years  
- `sex` — Male (1) or Female (0)  
- `cp` — Chest pain type (0–3)  
- `trestbps` — Resting blood pressure  
- `chol` — Cholesterol level  
- `fbs` — Fasting blood sugar > 120 mg/dl  
- `restecg` — Resting ECG results  
- `thalach` — Max heart rate achieved  
- `exang` — Exercise-induced angina  
- `oldpeak` — Depression induced by exercise  
- `slope`, `ca`, `thal` — Other heart condition indicators  

---

## ⚙️ Preprocessing

Steps taken:
1. Encoded categorical variables with **LabelEncoder** and **OneHotEncoder**.  
2. Scaled numerical features using **MinMaxScaler** to normalize data ranges.  
3. Split dataset into **65% training / 35% testing** with stratified sampling.  
4. Created a full preprocessing pipeline using `ColumnTransformer`.  

---

## Models and Methods

### 1. Decision Tree Classifier
- Trained a baseline Decision Tree with default parameters.  
- Tuned using **GridSearchCV** with 5-fold cross-validation.  
- Evaluated accuracy, confusion matrix, and feature importance.  
- **Best parameters:**  
  `max_depth=4`, `min_samples_split=16`, `criterion='entropy'`

**Results:**

| Model | Accuracy | Key Observation |
|--------|-----------|------------------|
| Default Decision Tree | 0.71 | Simple, interpretable structure |
| Tuned Decision Tree | 0.785 | Improved accuracy with parameter tuning |

---

### 2. Multi-Layer Perceptron (MLP)
- Architecture: (100, 100) hidden layers  
- `max_iter=1000`, `random_state=66`  
- Compared performance and training time vs. Decision Tree  

**Results:**

| Model | Accuracy | Comment |
|--------|-----------|----------|
| MLP | 0.832 | Captures nonlinear patterns better than Decision Tree |

**Speed comparison:**

| Operation | Decision Tree | MLP |
|------------|----------------|-----|
| Training Time | 0.0067s | 1.38s |
| Prediction Time | 0.0008s | 0.0011s |

---

### 3. Principal Component Analysis (PCA)
- Reduced feature space to 8 principal components.  
- **Top 8 components explained ~80% of total variance.**  
- Tested Decision Tree and MLP on PCA-transformed data.

| Model | Accuracy | Comment |
|--------|-----------|----------|
| PCA + Decision Tree | 0.785 | Similar to tuned tree; less noise |
| PCA + MLP | 0.794 | Slightly worse; nonlinear relationships lost |

---

### 4. K-Means Clustering
- Applied K-Means with different cluster sizes (2–30).  
- Evaluated using **Elbow Method** to find optimal clusters.  
- Optimal cluster range: **5–7**

| Dataset | Best Range | Notes |
|----------|-------------|-------|
| Original | 5–7 | Distinct elbow point |
| PCA-Reduced | 5–7 | Smoother curve, less separable clusters |

---

## Key Insights

- **MLP** achieved the best overall performance (≈83% accuracy).  
- **Decision Trees** were fastest and most interpretable.  
- **PCA** reduced dimensionality effectively but slightly hurt nonlinear model performance.  
- **K-Means** revealed natural groupings aligning with disease presence.

---

## Final Results Summary

| Model | Accuracy | Notes |
|--------|-----------|-------|
| Baseline (Majority Class) | 0.542 | Reference only |
| Decision Tree (Default) | 0.710 | Basic performance |
| Decision Tree (Tuned) | 0.785 | Best tree configuration |
| MLP | 0.832 | Highest performing classifier |
| PCA + Decision Tree | 0.785 | Matches tuned tree |
| PCA + MLP | 0.794 | Slight drop post-PCA |

---

## Reproducibility

### Setup
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
To launch the notebook:

`jupyter notebook notebooks/heart_disease_analysis.ipynb`

### Results

After running the notebook, results will be saved to:

`results/metrics.txt`


Figures such as the Decision Tree plot and K-Means elbow curve will be saved to:

`results/figures/`

## Libraries Used

* Python 3.x
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Seaborn

## Author

Ashley Wu

University of California, Los Angeles

📧 ashleyjwu05@gmail.com
