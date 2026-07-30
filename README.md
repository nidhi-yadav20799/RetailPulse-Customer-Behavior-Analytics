<div align="center">

# 🛍️ RetailPulse
### Customer Behavior Analytics & Revenue Intelligence Platform

*End-to-end data science pipeline for customer segmentation, churn prediction, revenue forecasting, and retention analysis*

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.3-005A9C?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.60-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-3.14-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)

[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=flat-square)](#-project-status)
[![Days Completed](https://img.shields.io/badge/Days%20Completed-8%2F10-brightgreen?style=flat-square)](#-project-status)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](#-license)

<br>

<img src="https://img.shields.io/badge/Segmentation-RFM%20%2B%20K--Means-8A2BE2?style=flat-square"/>
<img src="https://img.shields.io/badge/Churn%20AUC-0.81-success?style=flat-square"/>
<img src="https://img.shields.io/badge/Forecasting-SARIMA%20%2B%20Prophet-orange?style=flat-square"/>
<img src="https://img.shields.io/badge/Explainability-SHAP-red?style=flat-square"/>

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Objectives](#-project-objectives)
- [Dataset](#-dataset)
- [Tech Stack](#️-tech-stack)
- [Project Structure](#-project-structure)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
- [Key Results](#-key-results)
- [Generated Artifacts](#-generated-artifacts)
- [Project Status](#-project-status)
- [Getting Started](#-getting-started)
- [Future Work](#-future-work)
- [Author](#-author)

---

## 🔍 Overview

**RetailPulse** is a full-cycle data science project built on the **Online Retail II** dataset — a UK-based online retailer's transaction history spanning two years. The pipeline takes raw, messy transactional data all the way to executive-ready business intelligence: who the best customers are, who's about to churn, what revenue looks like next quarter, and how well the business retains the customers it acquires.

Every notebook is fully reproducible (`random_state=42` throughout), every cleaning decision is documented and justified, and every model is evaluated on a proper held-out set — no metrics are reported from training data.

---

## 🎯 Project Objectives

| # | Objective |
|---|---|
| 1 | Perform Exploratory Data Analysis (EDA) |
| 2 | Build a reproducible data cleaning pipeline |
| 3 | Engineer customer-level behavioral features |
| 4 | Segment customers using RFM Analysis + K-Means Clustering |
| 5 | Predict customer churn using Machine Learning |
| 6 | Forecast future revenue using SARIMA & Prophet |
| 7 | Perform Cohort Retention Analysis |
| 8 | Track experiments with MLflow |
| 9 | Build an interactive Streamlit Dashboard |
| 10 | Generate an Executive Business Report |

---

## 📦 Dataset

**Source:** [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (UCI Machine Learning Repository)

Transactional records of a UK-based online retailer, Dec 2009 – Dec 2011:

| Column | Description |
|---|---|
| `Invoice` | Invoice/transaction number (prefix `C` = cancellation) |
| `StockCode` | Product/item code |
| `Description` | Product name |
| `Quantity` | Units purchased |
| `InvoiceDate` | Date and time of transaction |
| `Price` | Unit price (GBP) |
| `Customer ID` | Unique customer identifier |
| `Country` | Customer's country |

> 📊 **1,067,371 raw transactions** → **779,414 clean, deduplicated rows** after the Day 3 cleaning pipeline

---

## 🛠️ Tech Stack

<div align="center">

| Category | Tools |
|---|---|
| **Language** | ![Python](https://img.shields.io/badge/-Python%203.12-3776AB?style=flat-square&logo=python&logoColor=white) |
| **Data Wrangling** | ![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white) |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat-square) ![Seaborn](https://img.shields.io/badge/-Seaborn-4C72B0?style=flat-square) ![Plotly](https://img.shields.io/badge/-Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white) |
| **Machine Learning** | ![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) ![XGBoost](https://img.shields.io/badge/-XGBoost-005A9C?style=flat-square) |
| **Explainability** | ![SHAP](https://img.shields.io/badge/-SHAP-FF0051?style=flat-square) |
| **Time Series** | ![Statsmodels](https://img.shields.io/badge/-Statsmodels-8B0000?style=flat-square) ![Prophet](https://img.shields.io/badge/-Prophet-0866FF?style=flat-square) |
| **Experiment Tracking** | ![MLflow](https://img.shields.io/badge/-MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white) |
| **Dashboard** | ![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) |
| **Statistical Testing** | ![SciPy](https://img.shields.io/badge/-SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white) |

</div>

---

## 📁 Project Structure

```text
RetailPulse-Customer-Behavior-Analytics/
│
├── 📂 data/
│   ├── raw/                      # Original Online Retail II dataset
│   └── processed/                # Cleaned + engineered datasets
│
├── 📂 models/                    # Serialized models (.pkl)
│
├── 📂 notebooks/
│   ├── Day1_Setup.ipynb
│   ├── Day2_EDA.ipynb
│   ├── Day3_DataCleaning.ipynb
│   ├── Day4_FeatureEngineering.ipynb
│   ├── Day5_Segmentation.ipynb
│   ├── Day6_ChurnPrediction.ipynb
│   ├── Day7_RevenueForecasting.ipynb
│   └── Day8_CohortAnalysis.ipynb
│
├── 📂 reports/                   # Figures, metrics, business KPIs
│
├── 📂 mlruns/                    # MLflow tracking store (SQLite backend)
│
├── 📄 README.md
├── 📄 requirements.txt
└── 📄 .gitignore
```

---

## 🔄 Pipeline Walkthrough

<div align="center">

| Day | Milestone | Status |
|:---:|---|:---:|
| 1️⃣ | Project Setup, Environment Config, Data Dictionary | ✅ |
| 2️⃣ | EDA — Missing Values, Distributions, Correlations | ✅ |
| 3️⃣ | Data Cleaning — Dedup, Outlier Capping, Type Normalization | ✅ |
| 4️⃣ | Feature Engineering — RFM, CLV Approximation | ✅ |
| 5️⃣ | K-Means Segmentation — Elbow + Silhouette Validated | ✅ |
| 6️⃣ | Churn Prediction — RF/XGBoost + SHAP | ✅ |
| 7️⃣ | Revenue Forecasting — SARIMA vs Prophet | ✅ |
| 8️⃣ | Cohort Retention Analysis | ✅ |
| 9️⃣ | Streamlit Dashboard | 🚧 |
| 🔟 | Deployment + Executive Report | 🚧 |

</div>

---

## 📊 Key Results

### 🎯 Customer Segmentation
- RFM-based segmentation → 4 business-interpretable clusters (**Champions, Loyal Customers, At Risk, Lost/Low-Value**)
- K selected via **elbow method + silhouette score** (silhouette = 0.36 at K=4) — not chosen arbitrarily
- Full customer-level cluster profiling saved to `reports/cluster_profiles.csv`

### ⚠️ Customer Churn Prediction
- Binary churn label (180-day recency threshold, ~41% churn rate)
- Random Forest vs XGBoost compared via 5-fold cross-validation
- **Held-out test ROC-AUC: 0.81** (exceeds the >0.80 requirement)
- Recency-derived features deliberately excluded from the model to prevent label leakage
- SHAP bar + beeswarm plots for full model explainability

### 📈 Revenue Forecasting
- Monthly revenue time series (24 complete months; partial final month excluded)
- **SARIMA(1,1,1)(1,1,1,12)** vs **Prophet** compared on a 3-month held-out window
- SARIMA selected automatically by MAPE (5.4% vs Prophet's 61.5% on this short series)
- 3-month forward forecast with 95% confidence intervals

### 🔁 Cohort Retention Analysis
- Monthly acquisition cohorts tracked across 24 months
- Retention heatmap + average-order-value heatmap by cohort
- **Steepest drop-off consistently occurs between month 0 and month 1** — the single highest-leverage point for a retention campaign

---

## 🗃️ Generated Artifacts

<details>
<summary><strong>🤖 Models</strong></summary>
<br>

| File | Description |
|---|---|
| `churn_classifier.pkl` | Trained churn prediction model |
| `kmeans_segmentation.pkl` | Fitted K-Means clustering model |
| `rfm_scaler.pkl` | StandardScaler for RFM features |
| `country_encoder.pkl` | Label encoder for country feature |

</details>

<details>
<summary><strong>💾 Processed Data</strong></summary>
<br>

| File | Description |
|---|---|
| `retail_cleaned.csv` | Cleaned, deduplicated transaction data |
| `customer_features.csv` | RFM + behavioral features per customer |
| `customer_segments.csv` | Customers with cluster/segment labels |
| `customer_segments_with_churn.csv` | Segments + churn probability |
| `revenue_forecast_3month.csv` | 3-month forward revenue forecast |
| `forecast_model_comparison.csv` | SARIMA vs Prophet performance metrics |
| `cohort_retention_matrix.csv` | Monthly cohort retention percentages |

</details>

<details>
<summary><strong>📈 Reports & Visualizations</strong></summary>
<br>

- EDA distribution plots, correlation heatmap, business KPIs
- Cluster profiles and segment boxplots
- Churn model ROC curve, confusion matrix
- SHAP feature importance (bar + beeswarm)
- Revenue forecast + decomposition plots
- Cohort retention & AOV heatmaps

</details>

---

## 🚦 Project Status

<div align="center">

| Day | Task | Status |
|:---:|---|:---:|
| 1 | Project Setup | ✅ Completed |
| 2 | Exploratory Data Analysis | ✅ Completed |
| 3 | Data Cleaning Pipeline | ✅ Completed |
| 4 | Feature Engineering | ✅ Completed |
| 5 | Customer Segmentation | ✅ Completed |
| 6 | Churn Prediction | ✅ Completed |
| 7 | Revenue Forecasting | ✅ Completed |
| 8 | Cohort Analysis | ✅ Completed |
| 9 | Streamlit Dashboard | 🚧 In Progress |
| 10 | Deployment & Executive Report | 🚧 Upcoming |

**Progress: 8 / 10 days complete**

![Progress](https://progress-bar.xyz/80/?title=Complete&width=400&color=4CAF50)

</div>

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/nidhi-yadav20799/RetailPulse-Customer-Behavior-Analytics.git
cd RetailPulse-Customer-Behavior-Analytics

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Launch JupyterLab
jupyter lab

# View MLflow experiment tracking
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

---

## 🔮 Future Work

- [ ] Interactive Streamlit dashboard (Overview, Segments, Churn, Forecast, Cohorts tabs)
- [ ] Deployment to Streamlit Community Cloud
- [ ] Executive Business Report (PDF) with actionable recommendations
- [ ] Project demonstration video

---

## 👩‍💻 Author

<div align="center">

**Nidhi Yadav**


[![GitHub](https://img.shields.io/badge/GitHub-nidhi--yadav20799-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/nidhi-yadav20799)

</div>

---

<div align="center">

⭐ *If you find this project useful, consider giving it a star on GitHub!* ⭐

</div>