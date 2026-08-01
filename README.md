<div align="center">

# 🛍️ RetailPulse
### Customer Behavior Analytics & Revenue Intelligence Platform

*End-to-end data science pipeline for customer segmentation, churn prediction, revenue forecasting, and retention analysis*

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.3-005A9C?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.60-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-3.14-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)

[![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)](#-project-status)
[![Days Completed](https://img.shields.io/badge/Days%20Completed-10%2F10-brightgreen?style=flat-square)](#-project-status)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](#-license)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://retailpulse-customer-behavior-analytics-cyay3aaqzwvzi6tlgucqga.streamlit.app)
[![Repo](https://img.shields.io/badge/GitHub-Repository-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/nidhi-yadav20799/RetailPulse-Customer-Behavior-Analytics)
[![Live Demo](https://img.shields.io/badge/Live-Demo-success?style=for-the-badge&logo=streamlit)](https://retailpulse-customer-behavior-analytics-cyay3aaqzwvzi6tlgucqga.streamlit.app)

<br>

<img src="https://img.shields.io/badge/Segmentation-RFM%20%2B%20K--Means-8A2BE2?style=flat-square"/>
<img src="https://img.shields.io/badge/Churn%20AUC-0.81-success?style=flat-square"/>
<img src="https://img.shields.io/badge/Forecasting-SARIMA%20%2B%20Prophet-orange?style=flat-square"/>
<img src="https://img.shields.io/badge/Explainability-SHAP-red?style=flat-square"/>

</div>

---

## 🎥 Live Demo

<div align="center">

### 👉 [**Launch the RetailPulse Dashboard**](https://retailpulse-customer-behavior-analytics-cyay3aaqzwvzi6tlgucqga.streamlit.app) 👈

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://retailpulse-customer-behavior-analytics-cyay3aaqzwvzi6tlgucqga.streamlit.app)

*Explore live customer segments, churn risk scores, revenue forecasts, and cohort retention — all in one interactive dashboard.*

</div>

---

## 🌐 Live Demo
**Streamlit Dashboard**
https://retailpulse-customer-behavior-analytics-cyay3aaqzwvzi6tlgucqga.streamlit.app

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Objectives](#-project-objectives)
- [Dataset](#-dataset)
- [Tech Stack](#️-tech-stack)
- [Project Structure](#-project-structure)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
- [Key Results](#-key-results)
- [Dashboard Preview](#-dashboard-preview)
- [Generated Artifacts](#-generated-artifacts)
- [Project Status](#-project-status)
- [Getting Started](#-getting-started)
- [Deployment](#-deployment)
- [Business Impact](#-business-impact)
- [Key Achievements](#-key-achievements)
- [Future Work](#-future-work)
- [Author](#-author)

---

## 🔍 Overview

**RetailPulse** is a full-cycle data science project built on the **Online Retail II** dataset — a UK-based online retailer's transaction history spanning two years. The pipeline takes raw, messy transactional data all the way to executive-ready business intelligence: who the best customers are, who's about to churn, what revenue looks like next quarter, and how well the business retains the customers it acquires.

Every notebook is fully reproducible (`random_state=42` throughout), every cleaning decision is documented and justified, and every model is evaluated on a proper held-out set — no metrics are reported from training data.

The project is now **100% complete** — from raw data to a live, deployed, interactive dashboard and executive business report.

---

## ✨ Features

- 📊 **Interactive Plotly Visualizations** — dynamic, drill-down charts across every module
- 🎨 **Premium Glassmorphism UI** — modern, polished dashboard styling
- 📱 **Responsive Dashboard** — works cleanly across desktop and mobile viewports
- 📥 **Export Filtered CSV** — download filtered customer/segment data on demand
- 🧩 **Customer Segmentation** — RFM + K-Means clustering into business-ready segments
- ⚠️ **Churn Prediction** — ML-driven churn risk scoring per customer
- 🔬 **SHAP Explainability** — transparent, feature-level model interpretability
- 📈 **Revenue Forecasting** — SARIMA vs Prophet, best model auto-selected
- 🔁 **Cohort Analysis** — monthly acquisition cohort retention & AOV heatmaps
- 🧪 **MLflow Tracking** — full experiment tracking and model versioning

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
| 9 | Build a premium interactive Streamlit Dashboard |
| 10 | Deploy to the cloud and deliver an Executive Business Report |

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
| **Deployment** | ![Streamlit Cloud](https://img.shields.io/badge/-Streamlit%20Community%20Cloud-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) |
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
├── 📂 dashboard/
│   └── app.py                    # Premium Streamlit dashboard (deployed)
│
├── 📂 reports/                   # Figures, metrics, business KPIs, executive report
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
| 1️⃣ | Project Setup, Environment Config, Data Dictionary | ✅ Completed |
| 2️⃣ | EDA — Missing Values, Distributions, Correlations | ✅ Completed |
| 3️⃣ | Data Cleaning — Dedup, Outlier Capping, Type Normalization | ✅ Completed |
| 4️⃣ | Feature Engineering — RFM, CLV Approximation | ✅ Completed |
| 5️⃣ | K-Means Segmentation — Elbow + Silhouette Validated | ✅ Completed |
| 6️⃣ | Churn Prediction — RF/XGBoost + SHAP | ✅ Completed |
| 7️⃣ | Revenue Forecasting — SARIMA vs Prophet | ✅ Completed |
| 8️⃣ | Cohort Retention Analysis | ✅ Completed |
| 9️⃣ | Premium Streamlit Dashboard | ✅ Completed |
| 🔟 | Streamlit Cloud Deployment + Executive Business Report | ✅ Completed |

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

## 🖼️ Dashboard Preview

<div align="center">

| Dashboard View 1 | Dashboard View 2 | Dashboard View 3 |
|:---:|:---:|:---:|
| ![Dashboard View 1](assets/Screenshot%202026-08-01%20223940.png) | ![Dashboard View 2](assets/Screenshot%202026-08-01%20225950.png) | ![Dashboard View 3](assets/Screenshot%202026-08-01%20230052.png) |

| Dashboard View 4 | Dashboard View 5 | Dashboard View 6 |
|:---:|:---:|:---:|
| ![Dashboard View 4](assets/Screenshot%202026-08-01%20230113.png) | ![Dashboard View 5](assets/Screenshot%202026-08-01%20230144.png) | ![Dashboard View 6](assets/Screenshot%202026-08-01%20230246.png) |

*Screenshots of the live Streamlit dashboard — segments, churn risk, and forecasts at a glance.*
<!-- TODO: Replace the generic "Dashboard View N" captions above with descriptive titles once you confirm what each screenshot shows. -->

</div>

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

<details>
<summary><strong>🖥️ Application & Deployment</strong></summary>
<br>

| Item | Description |
|---|---|
| `dashboard/app.py` | Premium Streamlit dashboard application |
| Executive Business Report | Stakeholder-facing summary of findings & recommendations |
| Streamlit Dashboard (Live) | Deployed on Streamlit Community Cloud |
| MLflow Tracking | Full experiment history in `mlruns/` |

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
| 9 | Premium Streamlit Dashboard | ✅ Completed |
| 10 | Streamlit Cloud Deployment + Executive Report | ✅ Completed |

**Progress: 10 / 10 days complete — Project 100% Finished 🎉**

![Progress](https://progress-bar.xyz/100/?title=Complete&width=400&color=4CAF50)

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

# Launch JupyterLab (to explore the notebooks)
jupyter lab

# Run the Streamlit dashboard locally
streamlit run dashboard/app.py

# View MLflow experiment tracking
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

---

## ☁️ Deployment

RetailPulse is fully deployed on **Streamlit Community Cloud**, making the entire analytics pipeline accessible without any local setup.

- 🔗 **Live App:** [retailpulse-customer-behavior-analytics-cyay3aaqzwvzi6tlgucqga.streamlit.app](https://retailpulse-customer-behavior-analytics-cyay3aaqzwvzi6tlgucqga.streamlit.app)
- ⚙️ **Entry point:** `dashboard/app.py`
- 🔄 **Auto-redeploy** on every push to the main branch
- 📦 Dependencies managed via `requirements.txt`

---

## 💼 Business Impact

RetailPulse translates raw transactional data into decisions a retail business can act on:

- **Marketing:** Target "Champions" and "Loyal Customers" segments with retention offers; deprioritize spend on "Lost/Low-Value" segments
- **Retention:** Focus intervention budget on the month-0-to-month-1 window, identified as the steepest customer drop-off point
- **Churn Prevention:** Proactively flag high-risk customers (via the 0.81 AUC churn model) for win-back campaigns before they lapse
- **Revenue Planning:** Use the 3-month SARIMA forecast with confidence intervals to inform inventory, staffing, and cash-flow planning
- **Executive Reporting:** The Executive Business Report distills all findings into stakeholder-ready recommendations

---

## 🏆 Key Achievements

- ✅ **End-to-End Data Science Pipeline** — from raw transactional data to deployed product
- ✅ **Machine Learning** — customer segmentation and churn classification
- ✅ **Explainable AI** — SHAP-based model interpretability throughout
- ✅ **Time Series Forecasting** — SARIMA vs Prophet, rigorously benchmarked
- ✅ **Executive Dashboard** — premium, interactive Streamlit application
- ✅ **Production Deployment** — live on Streamlit Community Cloud

---

## 🔮 Future Work

- [ ] Add automated model retraining pipeline (CI/CD)
- [ ] Extend forecasting to per-country / per-segment granularity
- [ ] Add authentication for role-based dashboard access
- [ ] Record a project demonstration video

---

## 👩‍💻 Author

<div align="center">

**Nidhi Yadav**


[![GitHub](https://img.shields.io/badge/GitHub-nidhi--yadav20799-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/nidhi-yadav20799)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://retailpulse-customer-behavior-analytics-cyay3aaqzwvzi6tlgucqga.streamlit.app)

### 🌐 Live Application
https://retailpulse-customer-behavior-analytics-cyay3aaqzwvzi6tlgucqga.streamlit.app

</div>

---

<div align="center">

⭐ *If you find this project useful, consider giving it a star on GitHub!* ⭐

</div>