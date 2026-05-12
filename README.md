# 🎓 Student Dropout Risk Predictor

**Applied Machine Learning — EDGE Series | DUET, Gazipur**  
Project 01 | Tabular | Intermediate | Education Domain

---

## Overview

This project builds an end-to-end machine learning pipeline to predict the probability of student dropout using academic, demographic, and socioeconomic features. An interactive Streamlit GUI allows academic advisors to enter student details and receive a risk score along with the top contributing factors.

## Features

- Full EDA with class distribution, correlation heatmap, and boxplots
- Preprocessing pipeline: median imputation, standard scaling, SMOTE oversampling
- Three classifiers trained and compared: Logistic Regression, Decision Tree, Random Forest
- 5-fold Stratified Cross-Validation with AUC-ROC scoring
- Evaluation: confusion matrix, ROC curves, feature importances
- Interactive Streamlit GUI with risk badge (Low / Medium / High) and factor explanation

## Results Summary

| Model | Accuracy | F1 | AUC-ROC |
|-------|----------|----|---------|
| Logistic Regression | 0.955 | 0.955 | 0.992 |
| Decision Tree | 0.933 | 0.932 | 0.943 |
| **Random Forest** | **0.960** | **0.960** | **0.992** |

**Best Model:** Random Forest (150 trees, balanced class weights, SMOTE)

## Project Structure

```
.
├── app.py                                  # Streamlit GUI
├── pipeline.py                             # Full ML pipeline (run this first)
├── Student_Dropout_Risk_Predictor.ipynb    # Jupyter Notebook
├── artifacts/
│   ├── student_data.csv                    # Dataset
│   ├── rf_model.pkl                        # Trained Random Forest
│   ├── scaler.pkl                          # StandardScaler
│   ├── imputer.pkl                         # SimpleImputer
│   ├── feature_names.json
│   ├── top_features.json
│   └── metrics.json
├── plots/                                  # All saved figures
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── boxplots.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── feature_importances.png
│   └── cv_comparison.png
├── requirements.txt
└── README.md
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the ML pipeline (generates model artifacts and plots)
```bash
python pipeline.py
```

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### 4. (Optional) Open the Jupyter Notebook
```bash
jupyter notebook Student_Dropout_Risk_Predictor.ipynb
```

## Dataset

Dataset mirrors the [UCI Student Dropout Dataset](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success).

Key features: Age at Enrollment, Previous Qualification Grade, Admission Grade, Curricular Units Enrolled/Approved/Grade, Scholarship Holder, Debtor Status, Tuition Fees Up to Date.

**Target:** Binary — Dropout (1) vs Graduate (0)

## Top Risk Factors (Random Forest)

1. Curricular Units Approved — strongest predictor
2. Curricular Units Grade — average grade signal
3. Curricular Units Enrolled — engagement indicator
4. Previous Qualification Grade — prior academic performance
5. Age at Enrollment — demographic risk signal

## Disclaimer

This tool is a decision-support aid only. Predictions are based on statistical patterns and should not replace professional academic counselling.

---

**Instructors:** Prof. Dr. Fazlul Hasan Siddiqui & Md. Rahad Khan | DUET, Gazipur
