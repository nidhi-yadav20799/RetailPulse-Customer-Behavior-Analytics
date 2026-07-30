# RetailPulse – Customer Behavior Analytics & Revenue Intelligence Platform

A Data Science and Machine Learning project that analyzes customer purchasing behavior using the Online Retail II dataset. The project performs Exploratory Data Analysis (EDA), data cleaning, customer feature engineering, RFM-based customer segmentation, churn prediction, and will be extended with revenue forecasting, cohort analysis, and an interactive Streamlit dashboard.

---

## Project Objectives

- Perform Exploratory Data Analysis (EDA)
- Build a reproducible data cleaning pipeline
- Engineer customer-level behavioral features
- Segment customers using RFM Analysis and K-Means Clustering
- Predict customer churn using Machine Learning
- Track experiments with MLflow
- Forecast future revenue (Upcoming)
- Build an interactive Streamlit Dashboard (Upcoming)

---

## Dataset

**Dataset:** Online Retail II

Contains transactional records of a UK-based online retailer including:

- Invoice
- StockCode
- Description
- Quantity
- InvoiceDate
- Price
- Customer ID
- Country

---

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-Learn
- XGBoost
- SHAP
- MLflow
- Streamlit

---

## Project Structure

```
RetailPulse-Customer-Behavior-Analytics/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── reports/
│
├── mlruns/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Completed Tasks

### Day 1
- Project Setup
- Environment Configuration
- Dataset Import
- Data Dictionary

### Day 2
- Exploratory Data Analysis
- Missing Value Analysis
- Distribution Analysis
- Correlation Analysis
- Business Insights

### Day 3
- Data Cleaning
- Duplicate Removal
- Outlier Treatment
- Feature Standardization
- Clean Dataset Generation

### Day 4
- Customer Feature Engineering
- RFM Metrics
- Customer Lifetime Value Approximation
- Behavioral Features

### Day 5
- K-Means Customer Segmentation
- Elbow Method
- Silhouette Score
- Customer Profiling
- MLflow Experiment Tracking

### Day 6
- Customer Churn Prediction
- Random Forest & XGBoost
- ROC-AUC Evaluation
- SHAP Explainability
- Model Serialization

---

## Current Results

### Customer Segmentation
- RFM-based segmentation
- K-Means Clustering
- Silhouette Score Evaluation

### Churn Prediction
- Random Forest Classifier
- ROC-AUC > 0.80
- SHAP Feature Importance
- SHAP Beeswarm Analysis

---

## Generated Artifacts

### Models
- churn_classifier.pkl
- kmeans_segmentation.pkl
- rfm_scaler.pkl
- country_encoder.pkl

### Processed Data
- retail_cleaned.csv
- customer_features.csv
- customer_segments.csv

### Reports
- EDA Visualizations
- Cluster Profiles
- Churn Model Performance
- SHAP Feature Importance
- SHAP Beeswarm

---

## Upcoming Work

- Day 7 – Revenue Forecasting (SARIMA & Prophet)
- Day 8 – Cohort Analysis
- Day 9 – Streamlit Dashboard
- Day 10 – Deployment & Executive Report

---

## Author

**Nidhi Yadav**

B.Sc. Data Science Graduate

GitHub: https://github.com/nidhi-yadav20799