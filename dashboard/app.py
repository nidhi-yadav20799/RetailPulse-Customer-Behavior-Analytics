"""
RetailPulse - Customer Behavior Analytics & Revenue Intelligence Platform
Day 9 - Interactive Streamlit Dashboard (Premium UI)

Run from the project root:
    streamlit run dashboard/app.py

Design tokens
-------------
Background:   #0a0e17 (near-black, slight blue cast)
Surface:      rgba(255,255,255,0.035) glass panels, 1px rgba(255,255,255,0.08) border
Accent A:     #7C5CFC -> #2563EB (violet-to-blue gradient - primary actions, hero)
Accent B:     #10B981 (emerald - positive / champions)
Accent C:     #F59E0B (amber - at risk / caution)
Accent D:     #F43F5E (rose - churn risk / negative)
Display font: Sora (headings, KPI numbers)
Body font:    Inter (labels, tables, captions)
Signature:    a slow-drifting aurora gradient glow behind the top header bar
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
# was launched from)
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

# --------------------------------------------------------------------------
# Design tokens
# --------------------------------------------------------------------------
COLORS = {
    "bg": "#0a0e17",
    "surface": "rgba(255,255,255,0.035)",
    "border": "rgba(255,255,255,0.09)",
    "text": "#e8eaf0",
    "text_muted": "#8b93a7",
    "violet": "#7C5CFC",
    "blue": "#2563EB",
    "emerald": "#10B981",
    "amber": "#F59E0B",
    "rose": "#F43F5E",
}

SEGMENT_COLORS = {
    "Champions": "#10B981",
    "Loyal Customers": "#2563EB",
    "At Risk": "#F59E0B",
    "Lost/Low-Value": "#F43F5E",
}

PLOTLY_FONT = dict(family="Inter, sans-serif", size=13, color=COLORS["text"])
PLOTLY_TEMPLATE_LAYOUT = dict(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=PLOTLY_FONT,
    margin=dict(t=28, b=28, l=10, r=10),
    hoverlabel=dict(
        bgcolor="#161b2c",
        bordercolor=COLORS["violet"],
        font=dict(family="Inter, sans-serif", size=13, color="#ffffff"),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color=COLORS["text_muted"]),
    ),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)"),
)


def style_fig(fig, height=None):
    """Apply the shared premium Plotly theme to any figure - single source of
    truth so every chart across all five tabs looks consistent."""
    fig.update_layout(**PLOTLY_TEMPLATE_LAYOUT)
    if height:
        fig.update_layout(height=height)
    return fig


# --------------------------------------------------------------------------
# Custom CSS - glassmorphism, gradients, animation, full Streamlit override
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --violet: #7C5CFC;
        --blue: #2563EB;
        --emerald: #10B981;
        --amber: #F59E0B;
        --rose: #F43F5E;
        --surface: rgba(255,255,255,0.035);
        --border: rgba(255,255,255,0.09);
        --text: #e8eaf0;
        --text-muted: #8b93a7;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3, .rp-display { font-family: 'Sora', sans-serif; }

    .stApp {
        background:
            radial-gradient(ellipse 1200px 600px at 15% -10%, rgba(124,92,252,0.14), transparent),
            radial-gradient(ellipse 1000px 500px at 85% 0%, rgba(37,99,235,0.10), transparent),
            #0a0e17;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .block-container { animation: fadeInUp 0.45s ease-out; }

    .rp-hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(120deg, rgba(124,92,252,0.16), rgba(37,99,235,0.10) 60%, rgba(16,185,129,0.06));
        border: 1px solid var(--border);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        padding: 1.8rem 2.2rem;
        border-radius: 22px;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.35);
    }
    .rp-hero::before {
        content: "";
        position: absolute; inset: -40% -10%;
        background: radial-gradient(circle at 30% 30%, rgba(124,92,252,0.25), transparent 55%),
                    radial-gradient(circle at 75% 60%, rgba(37,99,235,0.20), transparent 50%);
        animation: aurora 14s ease-in-out infinite alternate;
        pointer-events: none;
    }
    @keyframes aurora {
        0%   { transform: translate(-4%, -2%) scale(1); }
        100% { transform: translate(4%, 3%) scale(1.08); }
    }
    .rp-hero-content { position: relative; z-index: 1; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem; }
    .rp-hero h1 {
        color: #ffffff; font-size: 2rem; font-weight: 800; margin: 0; letter-spacing: -0.02em;
        display: flex; align-items: center; gap: 0.6rem;
    }
    .rp-hero p { color: var(--text-muted); font-size: 0.98rem; margin-top: 0.35rem; margin-bottom: 0; font-weight: 500; }
    .rp-hero-badge {
        background: rgba(16,185,129,0.14); border: 1px solid rgba(16,185,129,0.35);
        color: #34d399; font-size: 0.78rem; font-weight: 700; padding: 0.35rem 0.9rem;
        border-radius: 999px; letter-spacing: 0.02em; white-space: nowrap;
    }

    .kpi-card {
        position: relative;
        background: var(--surface);
        border: 1px solid var(--border);
        backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
        border-radius: 18px;
        padding: 1.35rem 1.5rem;
        box-shadow: 0 6px 22px rgba(0,0,0,0.28);
        transition: transform 0.22s cubic-bezier(.2,.8,.2,1), box-shadow 0.22s ease, border-color 0.22s ease;
        animation: fadeInUp 0.5s ease-out;
        overflow: hidden;
    }
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 14px 34px rgba(124,92,252,0.22);
        border-color: rgba(124,92,252,0.4);
    }
    .kpi-card::after {
        content: ""; position: absolute; top: -60%; right: -30%; width: 140px; height: 140px;
        background: radial-gradient(circle, var(--kpi-glow, rgba(124,92,252,0.25)), transparent 70%);
        pointer-events: none;
    }
    .kpi-icon {
        width: 38px; height: 38px; border-radius: 11px; display: flex; align-items: center; justify-content: center;
        font-size: 1.05rem; margin-bottom: 0.7rem;
        background: linear-gradient(135deg, var(--icon-a, #7C5CFC), var(--icon-b, #2563EB));
        box-shadow: 0 4px 14px rgba(124,92,252,0.35);
    }
    .kpi-label {
        color: var(--text-muted); font-size: 0.76rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.06em; margin-bottom: 0.3rem;
    }
    .kpi-value { color: #ffffff; font-size: 1.9rem; font-weight: 800; line-height: 1.15; font-family: 'Sora', sans-serif; letter-spacing: -0.01em; }
    .kpi-sub { color: var(--text-muted); font-size: 0.8rem; margin-top: 0.25rem; font-weight: 500; }

    .rp-section-title {
        font-size: 1.18rem; font-weight: 700; color: #f3f4f8;
        margin: 1.6rem 0 0.9rem 0; display: flex; align-items: center; gap: 0.55rem;
        font-family: 'Sora', sans-serif; letter-spacing: -0.01em;
    }
    .rp-section-title .bar {
        width: 5px; height: 1.1rem; border-radius: 4px;
        background: linear-gradient(180deg, var(--violet), var(--blue));
        display: inline-block;
    }

    .insight-box {
        position: relative;
        background: linear-gradient(135deg, rgba(124,92,252,0.09), rgba(37,99,235,0.05));
        border: 1px solid rgba(124,92,252,0.28);
        border-radius: 16px;
        padding: 1.1rem 1.4rem;
        color: #d7dae4;
        font-size: 0.94rem; line-height: 1.55;
        margin-top: 0.6rem;
        backdrop-filter: blur(10px);
        animation: fadeInUp 0.5s ease-out;
    }
    .insight-box b { color: #fff; }
    .insight-title {
        font-weight: 700; color: #b9a6ff; font-size: 0.82rem; text-transform: uppercase;
        letter-spacing: 0.05em; margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.4rem;
    }

    .rec-card {
        background: var(--surface); border: 1px solid var(--border); border-left: 3px solid var(--emerald);
        border-radius: 14px; padding: 0.95rem 1.2rem; margin-bottom: 0.6rem;
        transition: transform 0.18s ease, border-color 0.18s ease;
    }
    .rec-card:hover { transform: translateX(4px); border-left-color: var(--violet); }
    .rec-card b { color: #fff; }

    .segment-chip {
        display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.3rem 0.85rem;
        border-radius: 999px; font-size: 0.78rem; font-weight: 700; color: #fff;
        box-shadow: 0 3px 10px rgba(0,0,0,0.25);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1120, #0a0e17);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .rp-sidebar-title {
        font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.02rem; color: #fff;
        margin-bottom: 0.2rem; display: flex; align-items: center; gap: 0.5rem;
    }
    [data-testid="stSidebar"] .rp-sidebar-sub { color: var(--text-muted); font-size: 0.8rem; margin-bottom: 1rem; }

    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
        background: linear-gradient(120deg, var(--violet), var(--blue)) !important;
        border-radius: 8px !important;
    }
    [data-baseweb="select"] > div {
        background-color: rgba(255,255,255,0.04) !important;
        border-color: var(--border) !important;
        border-radius: 10px !important;
    }
    .stDateInput input {
        background-color: rgba(255,255,255,0.04) !important;
        border-radius: 10px !important;
        border-color: var(--border) !important;
    }

    [data-testid="stSidebar"] .stDownloadButton button {
        background: linear-gradient(120deg, var(--violet), var(--blue)) !important;
        color: #fff !important; border: none !important; border-radius: 11px !important;
        font-weight: 600 !important; box-shadow: 0 6px 18px rgba(124,92,252,0.28) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    [data-testid="stSidebar"] .stDownloadButton button:hover {
        transform: translateY(-2px); box-shadow: 0 10px 24px rgba(124,92,252,0.42) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px; background: var(--surface); padding: 6px; border-radius: 14px;
        border: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px; border-radius: 10px; color: var(--text-muted); font-weight: 600;
        font-size: 0.92rem; padding: 0 1.1rem; transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #fff; background: rgba(255,255,255,0.04); }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(120deg, var(--violet), var(--blue)) !important;
        color: #fff !important;
        box-shadow: 0 4px 14px rgba(124,92,252,0.35);
    }

    div[role="radiogroup"] label {
        background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
        padding: 0.35rem 0.9rem !important; margin-right: 0.4rem; transition: all 0.15s ease;
    }
    div[role="radiogroup"] label:hover { border-color: var(--violet); }

    [data-testid="stDataFrame"] {
        border-radius: 14px; overflow: hidden; border: 1px solid var(--border);
    }

    div[data-testid="stMetric"] {
        background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
        padding: 0.9rem 1.1rem; backdrop-filter: blur(12px);
        transition: transform 0.2s ease;
    }
    div[data-testid="stMetric"]:hover { transform: translateY(-3px); }
    div[data-testid="stMetricValue"] { font-size: 1.55rem; font-family: 'Sora', sans-serif; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: var(--text-muted); font-weight: 600; }

    div[data-testid="stAlert"] { border-radius: 14px; backdrop-filter: blur(10px); }

    [data-testid="stImage"] img { border-radius: 14px; border: 1px solid var(--border); }

    ::-webkit-scrollbar { width: 9px; height: 9px; }
    ::-webkit-scrollbar-track { background: #0a0e17; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(180deg, var(--violet), var(--blue)); border-radius: 5px; }

    .rp-footer {
        text-align: center; color: var(--text-muted); font-size: 0.82rem; padding: 1.2rem 0 0.4rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Cached data loaders
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading transaction data...")
def load_transactions():
    path = os.path.join(DATA_DIR, "retail_cleaned.csv")
    return pd.read_csv(path, parse_dates=["InvoiceDate"])


@st.cache_data(show_spinner="Loading customer segments...")
def load_customer_segments():
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


def kpi_card(col, icon, label, value, sub=None, icon_a="#7C5CFC", icon_b="#2563EB", glow="rgba(124,92,252,0.25)"):
    with col:
        sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
        st.markdown(
            f"""
            <div class="kpi-card" style="--icon-a:{icon_a}; --icon-b:{icon_b}; --kpi-glow:{glow};">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                {sub_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


def section_title(text):
    st.markdown(f'<div class="rp-section-title"><span class="bar"></span>{text}</div>', unsafe_allow_html=True)


def insight_box(title, body):
    st.markdown(
        f"""
        <div class="insight-box">
            <div class="insight-title">\u2728 {title}</div>
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def segment_chip(segment):
    color = SEGMENT_COLORS.get(segment, "#6b7280")
    return f'<span class="segment-chip" style="background:{color};">{segment}</span>'


# --------------------------------------------------------------------------
# Load everything
# --------------------------------------------------------------------------
transactions = load_transactions()
customers = load_customer_segments()
forecast_df, comparison_df = load_forecast()
retention_matrix, aov_matrix = load_cohort_data()
models = load_models()

# --------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------
st.sidebar.markdown('<div class="rp-sidebar-title">\U0001F50D Filters</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="rp-sidebar-sub">Refine the view across every tab</div>', unsafe_allow_html=True)

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

st.sidebar.markdown('<div class="rp-sidebar-title" style="margin-top:0.5rem;">\U0001F4E5 Export</div>', unsafe_allow_html=True)
st.sidebar.download_button(
    label="Download filtered customers",
    data=filtered_customers.to_csv(index=False).encode("utf-8"),
    file_name="retailpulse_filtered_customers.csv",
    mime="text/csv",
    width='stretch',
)
st.sidebar.download_button(
    label="Download filtered transactions",
    data=filtered_tx.to_csv(index=False).encode("utf-8"),
    file_name="retailpulse_filtered_transactions.csv",
    mime="text/csv",
    width='stretch',
)

# --------------------------------------------------------------------------
# Hero header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="rp-hero">
        <div class="rp-hero-content">
            <div>
                <h1>\U0001F6CD RetailPulse</h1>
                <p>Customer Behavior Analytics &amp; Revenue Intelligence Platform</p>
            </div>
            <div class="rp-hero-badge">\u25CF Live &middot; Online Retail II</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "\U0001F4CA  Executive Overview",
        "\U0001F465  Customer Segmentation",
        "\u26A0\uFE0F  Churn Prediction",
        "\U0001F4C8  Revenue Forecasting",
        "\U0001F501  Cohort Analysis",
    ]
)

# ==========================================================================
# TAB 1 - EXECUTIVE OVERVIEW
# ==========================================================================
with tabs[0]:
    if filtered_tx.empty:
        st.warning("No transactions match the selected filters.")
    else:
        section_title("Key Performance Indicators")

        total_revenue = filtered_tx["TotalPrice"].sum()
        total_customers = filtered_tx["Customer ID"].nunique()
        total_orders = filtered_tx["Invoice"].nunique()
        aov = filtered_tx.groupby("Invoice")["TotalPrice"].sum().mean()

        c1, c2, c3, c4 = st.columns(4)
        kpi_card(c1, "\U0001F4B0", "Total Revenue", f"\u00A3{total_revenue:,.0f}",
                 sub=f"{filtered_tx['Country'].nunique()} countries", icon_a="#7C5CFC", icon_b="#2563EB", glow="rgba(124,92,252,0.3)")
        kpi_card(c2, "\U0001F465", "Total Customers", f"{total_customers:,}",
                 sub="unique buyers", icon_a="#2563EB", icon_b="#10B981", glow="rgba(37,99,235,0.3)")
        kpi_card(c3, "\U0001F4E6", "Total Orders", f"{total_orders:,}",
                 sub="invoices placed", icon_a="#10B981", icon_b="#0ea5e9", glow="rgba(16,185,129,0.3)")
        kpi_card(c4, "\U0001F4C8", "Avg Order Value", f"\u00A3{aov:,.2f}",
                 sub="per invoice", icon_a="#F59E0B", icon_b="#F43F5E", glow="rgba(245,158,11,0.3)")

        section_title("Revenue Trend")
        monthly = filtered_tx.set_index("InvoiceDate").resample("ME")["TotalPrice"].sum().reset_index()
        fig = px.area(
            monthly, x="InvoiceDate", y="TotalPrice",
            labels={"InvoiceDate": "Month", "TotalPrice": "Revenue (GBP)"},
        )
        fig.update_traces(
            line=dict(color=COLORS["violet"], width=3),
            fillcolor="rgba(124,92,252,0.16)",
            hovertemplate="<b>%{x|%b %Y}</b><br>Revenue: \u00A3%{y:,.0f}<extra></extra>",
        )
        style_fig(fig, height=380)
        st.plotly_chart(fig, width='stretch')

        col_a, col_b = st.columns(2)
        with col_a:
            section_title("Revenue by Country (Top 10)")
            top_countries = (
                filtered_tx.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head(10).reset_index()
            )
            fig2 = px.bar(
                top_countries, x="TotalPrice", y="Country", orientation="h",
                labels={"TotalPrice": "Revenue (GBP)"}, color="TotalPrice",
                color_continuous_scale=[COLORS["blue"], COLORS["violet"]],
            )
            fig2.update_traces(hovertemplate="<b>%{y}</b><br>Revenue: \u00A3%{x:,.0f}<extra></extra>")
            style_fig(fig2, height=380)
            fig2.update_layout(yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.06)"), coloraxis_showscale=False)
            st.plotly_chart(fig2, width='stretch')

        with col_b:
            section_title("Orders by Weekday")
            wd = filtered_tx.copy()
            wd["Weekday"] = wd["InvoiceDate"].dt.day_name()
            order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            wd_counts = wd.groupby("Weekday")["Invoice"].nunique().reindex(order).reset_index()
            wd_counts.columns = ["Weekday", "Orders"]
            fig3 = px.bar(
                wd_counts, x="Weekday", y="Orders", color="Orders",
                color_continuous_scale=[COLORS["emerald"], COLORS["blue"], COLORS["violet"]],
            )
            fig3.update_traces(hovertemplate="<b>%{x}</b><br>Orders: %{y:,}<extra></extra>")
            style_fig(fig3, height=380)
            fig3.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig3, width='stretch')

        section_title("Insights & Recommendations")
        col_i, col_r = st.columns(2)
        with col_i:
            insight_box(
                "Insight",
                f"The selected view covers <b>{total_customers:,}</b> customers across "
                f"<b>{filtered_tx['Country'].nunique()}</b> countries, generating "
                f"<b>\u00A3{total_revenue:,.0f}</b> in revenue from <b>{total_orders:,}</b> orders "
                f"at an average order value of <b>\u00A3{aov:,.2f}</b>.",
            )
        with col_r:
            st.markdown(
                """
                <div class="rec-card">\U0001F3AF <b>Prioritize UK retention</b> - it drives the majority of revenue; even small churn reductions there compound quickly.</div>
                <div class="rec-card">\U0001F4C5 <b>Plan Q4 inventory early</b> - Sep-Nov consistently shows a seasonal demand spike.</div>
                """,
                unsafe_allow_html=True,
            )

# ==========================================================================
# TAB 2 - CUSTOMER SEGMENTATION
# ==========================================================================
with tabs[1]:
    if filtered_customers.empty:
        st.warning("No customers match the selected filters.")
    else:
        section_title("Segment Distribution")
        seg_counts = filtered_customers["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Customers"]

        chip_row = " ".join(
            segment_chip(s) + f' <span style="color:#8b93a7; font-size:0.82rem; margin-right:0.9rem;">{c}</span>'
            for s, c in zip(seg_counts["Segment"], seg_counts["Customers"])
        )
        st.markdown(f"<div style='margin-bottom:0.8rem;'>{chip_row}</div>", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1.4])
        with col1:
            fig = px.pie(
                seg_counts, names="Segment", values="Customers", hole=0.6,
                color="Segment", color_discrete_map=SEGMENT_COLORS,
            )
            fig.update_traces(
                textfont=dict(family="Inter, sans-serif", size=12, color="#fff"),
                hovertemplate="<b>%{label}</b><br>%{value:,} customers (%{percent})<extra></extra>",
                marker=dict(line=dict(color="#0a0e17", width=2)),
            )
            style_fig(fig, height=360)
            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.15))
            st.plotly_chart(fig, width='stretch')

        with col2:
            profile = filtered_customers.groupby("Segment").agg(
                Customers=("Customer ID", "count"),
                AvgRecency=("Recency", "mean"),
                AvgFrequency=("Frequency", "mean"),
                AvgMonetary=("Monetary", "mean"),
                AvgCLV=("CLV_Approx", "mean"),
            ).round(1).reset_index()
            st.dataframe(
                profile.style.format({
                    "AvgRecency": "{:.0f}", "AvgFrequency": "{:.1f}",
                    "AvgMonetary": "\u00A3{:,.0f}", "AvgCLV": "\u00A3{:,.0f}",
                }).background_gradient(subset=["AvgMonetary"], cmap="Purples"),
                width='stretch', hide_index=True, height=360,
            )

        section_title("RFM Distribution by Segment")
        metric_choice = st.radio(
            "Metric", ["Recency", "Frequency", "Monetary", "CLV_Approx"], horizontal=True, label_visibility="collapsed"
        )
        fig4 = px.box(
            filtered_customers, x="Segment", y=metric_choice, color="Segment",
            color_discrete_map=SEGMENT_COLORS, points=False,
        )
        fig4.update_traces(hovertemplate="<b>%{x}</b><br>" + metric_choice + ": %{y:,.1f}<extra></extra>")
        style_fig(fig4, height=380)
        fig4.update_layout(showlegend=False)
        st.plotly_chart(fig4, width='stretch')

        col3, col4 = st.columns(2)
        with col3:
            section_title("Recency vs Monetary")
            fig5 = px.scatter(
                filtered_customers, x="Recency", y="Monetary", color="Segment",
                size="Frequency", opacity=0.7, color_discrete_map=SEGMENT_COLORS,
                hover_data=["Customer ID", "Frequency"],
            )
            style_fig(fig5, height=380)
            st.plotly_chart(fig5, width='stretch')

        with col4:
            section_title("Top Countries by Segment")
            country_seg = (
                filtered_customers.groupby(["PrimaryCountry", "Segment"])["Customer ID"].count().reset_index()
            )
            top5_countries = filtered_customers["PrimaryCountry"].value_counts().head(5).index.tolist()
            country_seg = country_seg[country_seg["PrimaryCountry"].isin(top5_countries)]
            fig6 = px.bar(
                country_seg, x="PrimaryCountry", y="Customer ID", color="Segment",
                color_discrete_map=SEGMENT_COLORS, labels={"Customer ID": "Customers"},
            )
            style_fig(fig6, height=380)
            st.plotly_chart(fig6, width='stretch')

        section_title("Insights & Recommendations")
        col_i, col_r = st.columns(2)
        with col_i:
            insight_box(
                "Business insight",
                "<b>Champions</b> and <b>Loyal Customers</b> drive the bulk of monetary value despite being "
                "a minority of the customer base. <b>At Risk</b> customers still carry meaningful historical "
                "value and are the best win-back target before they fully churn.",
            )
        with col_r:
            st.markdown(
                """
                <div class="rec-card">\U0001F451 <b>Reward Champions</b> - early access and loyalty perks protect your highest-value cohort.</div>
                <div class="rec-card">\U0001F4E9 <b>Win back At Risk</b> - a personalized offer before full churn is far cheaper than reacquisition.</div>
                """,
                unsafe_allow_html=True,
            )

# ==========================================================================
# TAB 3 - CHURN PREDICTION
# ==========================================================================
with tabs[2]:
    section_title("Model Performance")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Held-out Test ROC-AUC", "0.813", help="Evaluated on a held-out test set, not training data")
    with c2:
        st.metric("Churn Definition", "180 days inactive")
    with c3:
        valid_churn = filtered_customers["Churned"].dropna()
        churn_display = f"{valid_churn.mean():.1%}" if len(valid_churn) > 0 else "N/A"
        st.metric("Churn Rate (filtered view)", churn_display)

    perf_img = img_path("churn_model_performance.png")
    if perf_img:
        st.image(perf_img, caption="ROC Curve & Confusion Matrix - Held-out Test Set", width='stretch')
    else:
        st.info("Run Day 6 notebook to generate churn_model_performance.png")

    section_title("Churn Probability Distribution")
    has_proba = filtered_customers["ChurnProbability"].notna().any()
    if has_proba:
        fig7 = px.histogram(
            filtered_customers.dropna(subset=["ChurnProbability"]),
            x="ChurnProbability", color="Segment", nbins=30,
            color_discrete_map=SEGMENT_COLORS, barmode="overlay", opacity=0.78,
        )
        fig7.update_traces(hovertemplate="Probability: %{x:.0%}<br>Customers: %{y}<extra></extra>")
        style_fig(fig7, height=360)
        fig7.update_layout(xaxis_tickformat=".0%")
        st.plotly_chart(fig7, width='stretch')
    else:
        st.info("Churn probability scores not found in the loaded customer file.")

    section_title("High-Risk Customers (Top 20 by Churn Probability)")
    if has_proba:
        high_risk = (
            filtered_customers.dropna(subset=["ChurnProbability"])
            .sort_values("ChurnProbability", ascending=False)
            .head(20)[["Customer ID", "Segment", "PrimaryCountry", "Recency", "Frequency", "Monetary", "ChurnProbability"]]
        )
        st.dataframe(
            high_risk.style.format({"Monetary": "\u00A3{:,.2f}", "ChurnProbability": "{:.1%}"})
            .background_gradient(subset=["ChurnProbability"], cmap="Reds"),
            width='stretch', hide_index=True,
        )
    else:
        st.info("No churn probability data available for the current filter selection.")

    section_title("SHAP Explainability")
    col3, col4 = st.columns(2)
    with col3:
        p = img_path("shap_feature_importance_bar.png")
        if p:
            st.image(p, caption="SHAP Feature Importance", width='stretch')
        else:
            st.info("Run Day 6 notebook to generate shap_feature_importance_bar.png")
    with col4:
        p = img_path("shap_beeswarm.png")
        if p:
            st.image(p, caption="SHAP Beeswarm Summary", width='stretch')
        else:
            st.info("Run Day 6 notebook to generate shap_beeswarm.png")

    insight_box(
        "Insight",
        "Recency-derived fields were deliberately excluded from the churn model's features to avoid label "
        "leakage (churn is itself defined by recency). SHAP confirms that purchase frequency and monetary "
        "behavior are the strongest legitimate predictors of churn risk.",
    )

# ==========================================================================
# TAB 4 - REVENUE FORECASTING
# ==========================================================================
with tabs[3]:
    section_title("Historical Revenue + 3-Month Forecast")

    historical = (
        transactions[transactions["InvoiceDate"] < "2011-12-01"]
        .set_index("InvoiceDate").resample("ME")["TotalPrice"].sum().reset_index()
    )

    fig8 = go.Figure()
    fig8.add_trace(go.Scatter(
        x=historical["InvoiceDate"], y=historical["TotalPrice"],
        mode="lines+markers", name="Historical Revenue",
        line=dict(color=COLORS["text_muted"], width=2.5), marker=dict(size=5),
        hovertemplate="<b>%{x|%b %Y}</b><br>Revenue: \u00A3%{y:,.0f}<extra></extra>",
    ))
    fig8.add_trace(go.Scatter(
        x=forecast_df["Month"], y=forecast_df["ForecastRevenue"],
        mode="lines+markers", name="Forecast (SARIMA)",
        line=dict(color=COLORS["rose"], width=3, dash="dot"), marker=dict(size=8, symbol="diamond"),
        hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: \u00A3%{y:,.0f}<extra></extra>",
    ))
    fig8.add_trace(go.Scatter(
        x=pd.concat([forecast_df["Month"], forecast_df["Month"][::-1]]),
        y=pd.concat([forecast_df["UpperCI"], forecast_df["LowerCI"][::-1]]),
        fill="toself", fillcolor="rgba(244,63,94,0.14)", line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence Interval", hoverinfo="skip",
    ))
    style_fig(fig8, height=420)
    fig8.update_layout(
        xaxis_title="Month", yaxis_title="Revenue (GBP)",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0),
    )
    st.plotly_chart(fig8, width='stretch')

    section_title("Forecast Summary")
    fc1, fc2, fc3 = st.columns(3)
    for col, (_, row) in zip([fc1, fc2, fc3], forecast_df.iterrows()):
        with col:
            kpi_card(
                col, "\U0001F4C5", row["Month"].strftime("%B %Y"), f"\u00A3{row['ForecastRevenue']:,.0f}",
                sub=f"CI: \u00A3{row['LowerCI']:,.0f} - \u00A3{row['UpperCI']:,.0f}",
                icon_a="#F43F5E", icon_b="#F59E0B", glow="rgba(244,63,94,0.25)",
            )

    section_title("SARIMA vs Prophet - Model Comparison")
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.dataframe(
            comparison_df.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "MAPE": "{:.1%}"})
            .background_gradient(subset=["MAPE"], cmap="RdYlGn_r"),
            width='stretch', hide_index=True,
        )
        best_model = comparison_df.loc[comparison_df["MAPE"].idxmin(), "Model"]
        st.success(f"\u2705 Best model selected (by MAPE): **{best_model}**")
    with col2:
        fig9 = px.bar(
            comparison_df, x="Model", y="MAPE", color="Model",
            color_discrete_map={"SARIMA": COLORS["emerald"], "Prophet": COLORS["amber"]},
            text_auto=".1%",
        )
        fig9.update_traces(hovertemplate="<b>%{x}</b><br>MAPE: %{y:.1%}<extra></extra>")
        style_fig(fig9, height=320)
        fig9.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig9, width='stretch')

    insight_box(
        "Insight",
        "With only ~2 years of history, Prophet's yearly-seasonality terms are poorly constrained (it even "
        "forecasts a negative revenue month on the held-out test period). SARIMA generalizes far better on "
        "this short series and is used for the production forecast above.",
    )

# ==========================================================================
# TAB 5 - COHORT ANALYSIS
# ==========================================================================
with tabs[4]:
    section_title("Monthly Cohort Retention Heatmap")

    fig10 = px.imshow(
        retention_matrix,
        color_continuous_scale=[[0, "#0a0e17"], [0.35, "#2563EB"], [0.7, "#7C5CFC"], [1, "#10B981"]],
        aspect="auto",
        labels=dict(x="Months Since First Purchase", y="Cohort Month", color="Retention"),
    )
    fig10.update_traces(hovertemplate="Cohort: %{y}<br>Month +%{x}<br>Retention: %{z:.1%}<extra></extra>")
    style_fig(fig10, height=640)
    fig10.update_layout(coloraxis_colorbar=dict(tickformat=".0%", title=""))
    st.plotly_chart(fig10, width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        section_title("Average Order Value by Cohort")
        fig11 = px.imshow(
            aov_matrix,
            color_continuous_scale=[[0, "#0a0e17"], [0.5, "#F59E0B"], [1, "#F43F5E"]],
            aspect="auto",
            labels=dict(x="Months Since First Purchase", y="Cohort Month", color="AOV (GBP)"),
        )
        fig11.update_traces(hovertemplate="Cohort: %{y}<br>Month +%{x}<br>AOV: \u00A3%{z:,.0f}<extra></extra>")
        style_fig(fig11, height=480)
        st.plotly_chart(fig11, width='stretch')

    with col2:
        section_title("Average Retention Curve (All Cohorts)")
        avg_retention = retention_matrix.mean(axis=0).reset_index()
        avg_retention.columns = ["MonthOffset", "Retention"]
        fig12 = px.area(avg_retention, x="MonthOffset", y="Retention")
        fig12.update_traces(
            line=dict(color=COLORS["emerald"], width=3), fillcolor="rgba(16,185,129,0.14)",
            hovertemplate="Month +%{x}<br>Retention: %{y:.1%}<extra></extra>",
        )
        style_fig(fig12, height=480)
        fig12.update_layout(
            yaxis_tickformat=".0%", xaxis_title="Months Since First Purchase", yaxis_title="Retention Rate",
        )
        st.plotly_chart(fig12, width='stretch')

    insight_box(
        "Insight",
        "The steepest drop-off consistently occurs between month 0 and month 1 - most one-time buyers never "
        "return the following month. This is the single highest-leverage point for a retention campaign (e.g. "
        "a 30-day follow-up offer). Retention figures beyond month ~20 are based on very few cohorts and "
        "should be read with caution.",
    )

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.markdown(
    '<div class="rp-footer">RetailPulse &middot; Built with Streamlit + Plotly &middot; '
    'Data: Online Retail II (UCI ML Repository)</div>',
    unsafe_allow_html=True,
)