"""
RetailPulse - Customer Behavior Analytics & Revenue Intelligence Platform
Day 9 - Interactive Streamlit Dashboard

Run from the project root's dashboard/ folder context:
    streamlit run dashboard/app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="RetailPulse | Customer Analytics & Revenue Intelligence",
    page_icon="\U0001F6CD",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Paths (relative to this file, so it works regardless of the CWD Streamlit
# was launched from - "streamlit run dashboard/app.py" from repo root, or
# "streamlit run app.py" from inside dashboard/, both resolve correctly)
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

# --------------------------------------------------------------------------
# Custom CSS - modern, portfolio-quality styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #0e1117;
    }

    /* Hero header */
    .rp-hero {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(37, 117, 252, 0.25);
    }
    .rp-hero h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
    }
    .rp-hero p {
        color: rgba(255,255,255,0.85);
        font-size: 1.05rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(145deg, #1a1f2e, #151925);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        text-align: left;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
        transition: transform 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
    }
    .kpi-label {
        color: #9aa4b2;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        color: #ffffff;
        font-size: 1.85rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .kpi-delta-pos { color: #4ade80; font-size: 0.85rem; font-weight: 600; }
    .kpi-delta-neg { color: #f87171; font-size: 0.85rem; font-weight: 600; }

    /* Segment badges */
    .segment-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        color: white;
    }

    /* Section headers */
    .rp-section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e5e7eb;
        border-left: 5px solid #2575fc;
        padding-left: 0.7rem;
        margin: 1.2rem 0 0.8rem 0;
    }

    .insight-box {
        background: rgba(37, 117, 252, 0.08);
        border: 1px solid rgba(37, 117, 252, 0.25);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        color: #d1d5db;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }

    [data-testid="stSidebar"] {
        background-color: #11151f;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }

    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb { background: #2575fc; border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

SEGMENT_COLORS = {
    "Champions": "#22c55e",
    "Loyal Customers": "#3b82f6",
    "At Risk": "#f59e0b",
    "Lost/Low-Value": "#ef4444",
}


# --------------------------------------------------------------------------
# Cached data loaders
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading transaction data...")
def load_transactions():
    path = os.path.join(DATA_DIR, "retail_cleaned.csv")
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    return df


@st.cache_data(show_spinner="Loading customer segments...")
def load_customer_segments():
    # Prefer the churn-enriched file; fall back to the base segments file if
    # the churn notebook hasn't been run yet in this environment.
    churn_path = os.path.join(DATA_DIR, "customer_segments_with_churn.csv")
    base_path = os.path.join(DATA_DIR, "customer_segments.csv")
    path = churn_path if os.path.exists(churn_path) else base_path
    df = pd.read_csv(path)
    if "Churned" not in df.columns:
        df["Churned"] = np.nan
    if "ChurnProbability" not in df.columns:
        df["ChurnProbability"] = np.nan
    return df


@st.cache_data(show_spinner=False)
def load_customer_features():
    path = os.path.join(DATA_DIR, "customer_features.csv")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_forecast():
    fc_path = os.path.join(REPORTS_DIR, "revenue_forecast_3month.csv")
    cmp_path = os.path.join(REPORTS_DIR, "forecast_model_comparison.csv")
    forecast = pd.read_csv(fc_path, parse_dates=["Month"])
    comparison = pd.read_csv(cmp_path)
    return forecast, comparison


@st.cache_data(show_spinner=False)
def load_cohort_data():
    ret_path = os.path.join(REPORTS_DIR, "cohort_retention_matrix.csv")
    aov_path = os.path.join(REPORTS_DIR, "cohort_aov_matrix.csv")
    retention = pd.read_csv(ret_path, index_col=0)
    retention.columns = [int(c) for c in retention.columns]
    aov = pd.read_csv(aov_path, index_col=0)
    aov.columns = [int(c) for c in aov.columns]
    return retention, aov


@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    try:
        models["churn"] = joblib.load(os.path.join(MODELS_DIR, "churn_classifier.pkl"))
        models["kmeans"] = joblib.load(os.path.join(MODELS_DIR, "kmeans_segmentation.pkl"))
        models["scaler"] = joblib.load(os.path.join(MODELS_DIR, "rfm_scaler.pkl"))
        models["country_encoder"] = joblib.load(os.path.join(MODELS_DIR, "country_encoder.pkl"))
    except FileNotFoundError as e:
        st.warning(f"Some model artifacts were not found ({e}). Precomputed scores from CSVs will be used instead.")
    return models


def img_path(name):
    p = os.path.join(REPORTS_DIR, name)
    return p if os.path.exists(p) else None


def kpi_card(label, value, col):
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------
# Load everything
# --------------------------------------------------------------------------
transactions = load_transactions()
customers = load_customer_segments()
customer_features = load_customer_features()
forecast_df, comparison_df = load_forecast()
retention_matrix, aov_matrix = load_cohort_data()
models = load_models()

# --------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------
st.sidebar.markdown("## \U0001F50D Filters")

all_countries = sorted(transactions["Country"].unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Country", options=all_countries, default=[], placeholder="All countries"
)

all_segments = sorted(customers["Segment"].dropna().unique().tolist())
selected_segments = st.sidebar.multiselect(
    "Customer Segment", options=all_segments, default=[], placeholder="All segments"
)

min_date = transactions["InvoiceDate"].min().date()
max_date = transactions["InvoiceDate"].max().date()
date_range = st.sidebar.date_input(
    "Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Filters apply to the Executive Overview, Segmentation, and Churn tabs. "
    "Forecast and Cohort tabs use the full historical dataset for methodological consistency."
)

# Apply filters
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

filtered_tx = transactions[
    (transactions["InvoiceDate"].dt.date >= start_date)
    & (transactions["InvoiceDate"].dt.date <= end_date)
]
if selected_countries:
    filtered_tx = filtered_tx[filtered_tx["Country"].isin(selected_countries)]

filtered_customers = customers.copy()
if selected_countries:
    filtered_customers = filtered_customers[filtered_customers["PrimaryCountry"].isin(selected_countries)]
if selected_segments:
    filtered_customers = filtered_customers[filtered_customers["Segment"].isin(selected_segments)]

# Sidebar download of filtered customer data
st.sidebar.markdown("## \U0001F4E5 Export")
st.sidebar.download_button(
    label="Download filtered customer data (CSV)",
    data=filtered_customers.to_csv(index=False).encode("utf-8"),
    file_name="retailpulse_filtered_customers.csv",
    mime="text/csv",
    use_container_width=True,
)
st.sidebar.download_button(
    label="Download filtered transactions (CSV)",
    data=filtered_tx.to_csv(index=False).encode("utf-8"),
    file_name="retailpulse_filtered_transactions.csv",
    mime="text/csv",
    use_container_width=True,
)

# --------------------------------------------------------------------------
# Hero header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="rp-hero">
        <h1>\U0001F6CD RetailPulse</h1>
        <p>Customer Behavior Analytics & Revenue Intelligence Platform</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "\U0001F4CA Executive Overview",
        "\U0001F465 Customer Segmentation",
        "\u26A0\uFE0F Churn Prediction",
        "\U0001F4C8 Revenue Forecasting",
        "\U0001F501 Cohort Analysis",
    ]
)

# ==========================================================================
# TAB 1 - EXECUTIVE OVERVIEW
# ==========================================================================
with tabs[0]:
    st.markdown('<div class="rp-section-title">Key Performance Indicators</div>', unsafe_allow_html=True)

    if filtered_tx.empty:
        st.warning("No transactions match the selected filters.")
    else:
        total_revenue = filtered_tx["TotalPrice"].sum()
        total_customers = filtered_tx["Customer ID"].nunique()
        total_orders = filtered_tx["Invoice"].nunique()
        aov = filtered_tx.groupby("Invoice")["TotalPrice"].sum().mean()

        c1, c2, c3, c4 = st.columns(4)
        kpi_card("Total Revenue", f"\u00A3{total_revenue:,.0f}", c1)
        kpi_card("Total Customers", f"{total_customers:,}", c2)
        kpi_card("Total Orders", f"{total_orders:,}", c3)
        kpi_card("Avg Order Value", f"\u00A3{aov:,.2f}", c4)

        st.markdown('<div class="rp-section-title">Revenue Trend</div>', unsafe_allow_html=True)
        monthly = filtered_tx.set_index("InvoiceDate").resample("ME")["TotalPrice"].sum().reset_index()
        fig = px.area(
            monthly, x="InvoiceDate", y="TotalPrice",
            labels={"InvoiceDate": "Month", "TotalPrice": "Revenue (GBP)"},
        )
        fig.update_traces(line_color="#2575fc", fillcolor="rgba(37,117,252,0.15)")
        fig.update_layout(
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="rp-section-title">Revenue by Country (Top 10)</div>', unsafe_allow_html=True)
            top_countries = (
                filtered_tx.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head(10).reset_index()
            )
            fig2 = px.bar(
                top_countries, x="TotalPrice", y="Country", orientation="h",
                labels={"TotalPrice": "Revenue (GBP)"}, color="TotalPrice",
                color_continuous_scale="Blues",
            )
            fig2.update_layout(
                template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(autorange="reversed"), coloraxis_showscale=False, margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            st.markdown('<div class="rp-section-title">Orders by Weekday</div>', unsafe_allow_html=True)
            wd = filtered_tx.copy()
            wd["Weekday"] = wd["InvoiceDate"].dt.day_name()
            order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            wd_counts = wd.groupby("Weekday")["Invoice"].nunique().reindex(order).reset_index()
            wd_counts.columns = ["Weekday", "Orders"]
            fig3 = px.bar(
                wd_counts, x="Weekday", y="Orders", color="Orders", color_continuous_scale="Purp"
            )
            fig3.update_layout(
                template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False, margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown(
            f"""
            <div class="insight-box">
            <strong>\U0001F4A1 Insight:</strong> the selected view covers <strong>{total_customers:,}</strong>
            customers across <strong>{filtered_tx['Country'].nunique()}</strong> countries, generating
            <strong>\u00A3{total_revenue:,.0f}</strong> in revenue from <strong>{total_orders:,}</strong> orders.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==========================================================================
# TAB 2 - CUSTOMER SEGMENTATION
# ==========================================================================
with tabs[1]:
    st.markdown('<div class="rp-section-title">Segment Distribution</div>', unsafe_allow_html=True)

    if filtered_customers.empty:
        st.warning("No customers match the selected filters.")
    else:
        seg_counts = filtered_customers["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Customers"]

        col1, col2 = st.columns([1, 1.4])
        with col1:
            fig = px.pie(
                seg_counts, names="Segment", values="Customers", hole=0.5,
                color="Segment", color_discrete_map=SEGMENT_COLORS,
            )
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            profile = filtered_customers.groupby("Segment").agg(
                Customers=("Customer ID", "count"),
                AvgRecency=("Recency", "mean"),
                AvgFrequency=("Frequency", "mean"),
                AvgMonetary=("Monetary", "mean"),
                AvgCLV=("CLV_Approx", "mean"),
            ).round(1).reset_index()
            st.dataframe(profile, use_container_width=True, hide_index=True)

        st.markdown('<div class="rp-section-title">RFM Distribution by Segment</div>', unsafe_allow_html=True)
        metric_choice = st.radio(
            "Metric", ["Recency", "Frequency", "Monetary", "CLV_Approx"], horizontal=True
        )
        fig4 = px.box(
            filtered_customers, x="Segment", y=metric_choice, color="Segment",
            color_discrete_map=SEGMENT_COLORS,
        )
        fig4.update_layout(
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False, margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig4, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<div class="rp-section-title">Recency vs Monetary</div>', unsafe_allow_html=True)
            fig5 = px.scatter(
                filtered_customers, x="Recency", y="Monetary", color="Segment",
                size="Frequency", opacity=0.65, color_discrete_map=SEGMENT_COLORS,
                hover_data=["Customer ID", "Frequency"],
            )
            fig5.update_layout(
                template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig5, use_container_width=True)

        with col4:
            st.markdown('<div class="rp-section-title">Top Countries by Segment</div>', unsafe_allow_html=True)
            country_seg = (
                filtered_customers.groupby(["PrimaryCountry", "Segment"])["Customer ID"]
                .count()
                .reset_index()
            )
            top5_countries = (
                filtered_customers["PrimaryCountry"].value_counts().head(5).index.tolist()
            )
            country_seg = country_seg[country_seg["PrimaryCountry"].isin(top5_countries)]
            fig6 = px.bar(
                country_seg, x="PrimaryCountry", y="Customer ID", color="Segment",
                color_discrete_map=SEGMENT_COLORS, labels={"Customer ID": "Customers"},
            )
            fig6.update_layout(
                template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig6, use_container_width=True)

        st.markdown(
            """
            <div class="insight-box">
            <strong>\U0001F4A1 Business insight:</strong>
            <b>Champions</b> and <b>Loyal Customers</b> drive the bulk of monetary value despite being a
            minority of the customer base - retention spend should prioritize these segments.
            <b>At Risk</b> customers still carry meaningful historical value and are the best win-back
            campaign target before they fully churn.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==========================================================================
# TAB 3 - CHURN PREDICTION
# ==========================================================================
with tabs[2]:
    st.markdown('<div class="rp-section-title">Model Performance</div>', unsafe_allow_html=True)

    perf_img = img_path("churn_model_performance.png")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        if perf_img:
            st.image(perf_img, caption="ROC Curve & Confusion Matrix - Held-out Test Set", use_container_width=True)
        else:
            st.info("Run Day 6 notebook to generate churn_model_performance.png")
    with col2:
        st.metric("Held-out Test ROC-AUC", "0.813")
        st.metric("Churn Threshold", "180 days inactive")
        valid_churn = filtered_customers["Churned"].dropna()
        if len(valid_churn) > 0:
            st.metric("Churn Rate (filtered view)", f"{valid_churn.mean():.1%}")

    st.markdown('<div class="rp-section-title">Churn Probability Distribution</div>', unsafe_allow_html=True)
    has_proba = filtered_customers["ChurnProbability"].notna().any()
    if has_proba:
        fig7 = px.histogram(
            filtered_customers.dropna(subset=["ChurnProbability"]),
            x="ChurnProbability", color="Segment", nbins=30,
            color_discrete_map=SEGMENT_COLORS, barmode="overlay", opacity=0.75,
        )
        fig7.update_layout(
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig7, use_container_width=True)
    else:
        st.info("Churn probability scores not found in the loaded customer file.")

    st.markdown('<div class="rp-section-title">High-Risk Customers (Top 20 by Churn Probability)</div>', unsafe_allow_html=True)
    if has_proba:
        high_risk = (
            filtered_customers.dropna(subset=["ChurnProbability"])
            .sort_values("ChurnProbability", ascending=False)
            .head(20)[["Customer ID", "Segment", "PrimaryCountry", "Recency", "Frequency", "Monetary", "ChurnProbability"]]
        )
        st.dataframe(
            high_risk.style.format({"Monetary": "\u00A3{:,.2f}", "ChurnProbability": "{:.1%}"}),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No churn probability data available for the current filter selection.")

    st.markdown('<div class="rp-section-title">SHAP Explainability</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        p = img_path("shap_feature_importance_bar.png")
        if p:
            st.image(p, caption="SHAP Feature Importance", use_container_width=True)
        else:
            st.info("Run Day 6 notebook to generate shap_feature_importance_bar.png")
    with col4:
        p = img_path("shap_beeswarm.png")
        if p:
            st.image(p, caption="SHAP Beeswarm Summary", use_container_width=True)
        else:
            st.info("Run Day 6 notebook to generate shap_beeswarm.png")

    st.markdown(
        """
        <div class="insight-box">
        <strong>\U0001F4A1 Insight:</strong> Recency-derived fields were deliberately excluded from the churn
        model's features to avoid label leakage (churn is itself defined by recency). SHAP confirms that
        purchase frequency and monetary behavior are the strongest legitimate predictors of churn risk.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==========================================================================
# TAB 4 - REVENUE FORECASTING
# ==========================================================================
with tabs[3]:
    st.markdown('<div class="rp-section-title">Historical Revenue + 3-Month Forecast</div>', unsafe_allow_html=True)

    historical = (
        transactions[transactions["InvoiceDate"] < "2011-12-01"]
        .set_index("InvoiceDate").resample("ME")["TotalPrice"].sum().reset_index()
    )

    fig8 = go.Figure()
    fig8.add_trace(go.Scatter(
        x=historical["InvoiceDate"], y=historical["TotalPrice"],
        mode="lines+markers", name="Historical Revenue", line=dict(color="#9aa4b2"),
    ))
    fig8.add_trace(go.Scatter(
        x=forecast_df["Month"], y=forecast_df["ForecastRevenue"],
        mode="lines+markers", name="Forecast", line=dict(color="#ef4444"),
    ))
    fig8.add_trace(go.Scatter(
        x=pd.concat([forecast_df["Month"], forecast_df["Month"][::-1]]),
        y=pd.concat([forecast_df["UpperCI"], forecast_df["LowerCI"][::-1]]),
        fill="toself", fillcolor="rgba(239,68,68,0.15)", line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence Interval", showlegend=True,
    ))
    fig8.update_layout(
        template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Month", yaxis_title="Revenue (GBP)", margin=dict(t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig8, use_container_width=True)

    st.markdown('<div class="rp-section-title">SARIMA vs Prophet - Model Comparison</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.dataframe(
            comparison_df.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "MAPE": "{:.1%}"}),
            use_container_width=True, hide_index=True,
        )
        best_model = comparison_df.loc[comparison_df["MAPE"].idxmin(), "Model"]
        st.success(f"Best model selected (by MAPE): **{best_model}**")
    with col2:
        fig9 = px.bar(
            comparison_df, x="Model", y="MAPE", color="Model",
            color_discrete_map={"SARIMA": "#22c55e", "Prophet": "#f59e0b"},
            text_auto=".1%",
        )
        fig9.update_layout(
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False, yaxis_tickformat=".0%", margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig9, use_container_width=True)

    st.markdown('<div class="rp-section-title">3-Month Forecast Detail</div>', unsafe_allow_html=True)
    st.dataframe(
        forecast_df.style.format({"ForecastRevenue": "\u00A3{:,.0f}", "LowerCI": "\u00A3{:,.0f}", "UpperCI": "\u00A3{:,.0f}"}),
        use_container_width=True, hide_index=True,
    )

    st.markdown(
        """
        <div class="insight-box">
        <strong>\U0001F4A1 Insight:</strong> with only ~2 years of history, Prophet's yearly-seasonality terms
        are poorly constrained (it even forecasts a negative revenue month on the held-out test period).
        SARIMA generalizes far better on this short series and is used for the production forecast above.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==========================================================================
# TAB 5 - COHORT ANALYSIS
# ==========================================================================
with tabs[4]:
    st.markdown('<div class="rp-section-title">Monthly Cohort Retention Heatmap</div>', unsafe_allow_html=True)

    fig10 = px.imshow(
        retention_matrix, color_continuous_scale="Blues", aspect="auto",
        labels=dict(x="Months Since First Purchase", y="Cohort Month", color="Retention"),
        text_auto=".0%",
    )
    fig10.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=20, b=20), height=650,
    )
    st.plotly_chart(fig10, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="rp-section-title">Average Order Value by Cohort</div>', unsafe_allow_html=True)
        fig11 = px.imshow(
            aov_matrix, color_continuous_scale="OrRd", aspect="auto",
            labels=dict(x="Months Since First Purchase", y="Cohort Month", color="AOV (GBP)"),
        )
        fig11.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=20, b=20), height=500)
        st.plotly_chart(fig11, use_container_width=True)

    with col2:
        st.markdown('<div class="rp-section-title">Average Retention Curve (All Cohorts)</div>', unsafe_allow_html=True)
        avg_retention = retention_matrix.mean(axis=0).reset_index()
        avg_retention.columns = ["MonthOffset", "Retention"]
        fig12 = px.line(
            avg_retention, x="MonthOffset", y="Retention", markers=True,
        )
        fig12.update_traces(line_color="#2575fc")
        fig12.update_layout(
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis_tickformat=".0%", xaxis_title="Months Since First Purchase", yaxis_title="Retention Rate",
            margin=dict(t=20, b=20), height=500,
        )
        st.plotly_chart(fig12, use_container_width=True)

    st.markdown(
        """
        <div class="insight-box">
        <strong>\U0001F4A1 Insight:</strong> the steepest drop-off consistently occurs between month 0 and
        month 1 - most one-time buyers never return the following month. This is the single highest-leverage
        point for a retention campaign (e.g. a 30-day follow-up offer). Retention figures beyond month ~20
        are based on very few cohorts and should be read with caution.
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.markdown("---")
st.caption("RetailPulse | Built with Streamlit + Plotly | Data: Online Retail II (UCI ML Repository)")