# RetailPulse – Customer Behavior Analytics & Revenue Intelligence Platform

A Data Science and Machine Learning project that analyzes customer purchasing behavior using the Online Retail II dataset. The project performs Exploratory Data Analysis (EDA), data cleaning, customer feature engineering, RFM-based customer segmentation, customer churn prediction, revenue forecasting, cohort analysis, and will be extended with an interactive Streamlit dashboard.

---

# Project Objectives

- Perform Exploratory Data Analysis (EDA)
- Build a reproducible data cleaning pipeline
- Engineer customer-level behavioral features
- Segment customers using RFM Analysis and K-Means Clustering
- Predict customer churn using Machine Learning
- Forecast future revenue using SARIMA & Prophet
- Perform Cohort Retention Analysis
- Track experiments with MLflow
- Build an interactive Streamlit Dashboard
- Generate an Executive Business Report

---

# Dataset

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

# Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-Learn
- XGBoost
- SHAP
- Statsmodels
- Prophet
- MLflow
- Streamlit

---

# Project Structure

```text
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

# Completed Tasks

## Day 1
- Project Setup
- Environment Configuration
- Dataset Import
- Data Dictionary

## Day 2
- Exploratory Data Analysis (EDA)
- Missing Value Analysis
- Distribution Analysis
- Correlation Analysis
- Business Insights

## Day 3
- Data Cleaning Pipeline
- Duplicate Removal
- Missing Value Handling
- Outlier Treatment
- Feature Standardization
- Clean Dataset Generation

## Day 4
- Customer Feature Engineering
- RFM Metrics
- Customer Lifetime Value (CLV) Approximation
- Behavioral Feature Engineering

## Day 5
- K-Means Customer Segmentation
- Elbow Method
- Silhouette Score Evaluation
- Customer Profiling
- MLflow Experiment Tracking

## Day 6
- Customer Churn Prediction
- Random Forest & XGBoost Models
- ROC-AUC Model Evaluation
- SHAP Explainability
- Model Serialization

## Day 7
- Revenue Forecasting
- Monthly Revenue Time Series
- SARIMA Forecasting
- Prophet Forecasting
- Forecast Model Comparison
- Three-Month Revenue Forecast

## Day 8
- Cohort Analysis
- Customer Retention Matrix
- Average Order Value (AOV) Analysis
- Retention Heatmaps
- Cohort Business Insights

---

# Current Results

## Customer Segmentation
- RFM-based Customer Segmentation
- K-Means Clustering
- Silhouette Score Evaluation
- Customer Cluster Profiling

## Customer Churn Prediction
- Random Forest Classifier
- XGBoost Classifier
- ROC-AUC > 0.80
- SHAP Feature Importance
- SHAP Beeswarm Analysis

## Revenue Forecasting
- SARIMA Forecast Model
- Prophet Forecast Model
- Model Performance Comparison
- Three-Month Revenue Forecast

## Cohort Analysis
- Customer Retention Matrix
- Cohort Retention Heatmap
- Average Order Value Heatmap
- Customer Retention Curve

---

# Generated Artifacts

## Models
- churn_classifier.pkl
- kmeans_segmentation.pkl
- rfm_scaler.pkl
- country_encoder.pkl

## Processed Data
- retail_cleaned.csv
- customer_features.csv
- customer_segments.csv
- customer_segments_with_churn.csv
- revenue_forecast_3month.csv
- forecast_model_comparison.csv
- cohort_retention_matrix.csv

## Reports
- EDA Visualizations
- Cluster Profiles
- Churn Model Performance
- SHAP Feature Importance
- SHAP Beeswarm
- Revenue Forecast Plots
- Forecast Comparison
- Cohort Retention Heatmap
- Cohort AOV Heatmap
- Cohort Retention Curve

---

# Project Status

| Day | Status |
|------|--------|
| Day 1 | ✅ Completed |
| Day 2 | ✅ Completed |
| Day 3 | ✅ Completed |
| Day 4 | ✅ Completed |
| Day 5 | ✅ Completed |
| Day 6 | ✅ Completed |
| Day 7 | ✅ Completed |
| Day 8 | ✅ Completed |
| Day 9 | 🚧 Streamlit Dashboard |
| Day 10 | 🚧 Deployment & Executive Report |

---

# Future Work

- Streamlit Interactive Dashboard
- Final Model Deployment
- Executive Business Report
- Project Demonstration

---

# Author

**Nidhi Yadav**

B.Sc. Data Science Graduate

GitHub: https://github.com/nidhi-yadav20799