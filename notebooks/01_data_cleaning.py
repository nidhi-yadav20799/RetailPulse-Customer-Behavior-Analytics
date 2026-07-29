"""
RetailPulse — Data Cleaning Pipeline
Day 1-3 deliverable: loads raw transaction data, applies documented cleaning
rules, and writes a reproducible cleaned dataset + cleaning log.

Run: python 01_data_cleaning.py
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

RANDOM_STATE = 42
RAW_PATH = Path("data/raw/online_retail_II.csv")
OUT_PATH = Path("data/processed/retail_cleaned.csv")
CANCEL_PATH = Path("data/processed/cancellations.csv")
LOG_PATH = Path("reports/cleaning_log.json")


def load_raw(path: Path) -> pd.DataFrame:
    """Load raw CSV with correct encoding and dtypes."""
    df = pd.read_csv(path, encoding="latin1", dtype={"Invoice": str, "StockCode": str})
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def clean_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Apply documented cleaning steps in sequence, logging row counts removed
    at each stage for reproducibility. Returns (cleaned_df, cancellations_df, log).
    """
    log = {"start_rows": len(df)}

    # Step 1: separate cancellations (Invoice starting with 'C') before other filtering
    is_cancel = df["Invoice"].str.startswith("C", na=False)
    cancellations = df[is_cancel].copy()
    df = df[~is_cancel].copy()
    log["removed_cancellations"] = int(is_cancel.sum())

    # Step 2: drop null Customer ID (anonymous transactions — can't attribute to a customer)
    n_before = len(df)
    df = df.dropna(subset=["Customer ID"])
    log["removed_null_customer_id"] = n_before - len(df)

    # Step 3: drop null Description
    n_before = len(df)
    df = df.dropna(subset=["Description"])
    log["removed_null_description"] = n_before - len(df)

    # Step 4: remove non-positive Price
    n_before = len(df)
    df = df[df["Price"] > 0]
    log["removed_nonpositive_price"] = n_before - len(df)

    # Step 5: remove non-positive Quantity (remaining, post-cancellation-removal)
    n_before = len(df)
    df = df[df["Quantity"] > 0]
    log["removed_nonpositive_quantity"] = n_before - len(df)

    # Step 6: Winsorize extreme outliers in Quantity and Price at 0.5/99.5 percentile
    for col in ["Quantity", "Price"]:
        lo, hi = df[col].quantile([0.005, 0.995])
        n_capped = int(((df[col] < lo) | (df[col] > hi)).sum())
        df[col] = df[col].clip(lower=lo, upper=hi)
        log[f"capped_{col}_outliers"] = n_capped

    df["Customer ID"] = df["Customer ID"].astype(int)
    df["TotalPrice"] = df["Quantity"] * df["Price"]

    log["final_rows"] = len(df)
    log["pct_retained"] = round(len(df) / log["start_rows"] * 100, 2)

    return df.reset_index(drop=True), cancellations.reset_index(drop=True), log


def main():
    np.random.seed(RANDOM_STATE)
    df_raw = load_raw(RAW_PATH)
    df_clean, df_cancel, log = clean_pipeline(df_raw)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUT_PATH, index=False)
    df_cancel.to_csv(CANCEL_PATH, index=False)
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    print("Cleaning complete.")
    print(json.dumps(log, indent=2))
    print(f"\nCleaned dataset: {OUT_PATH} ({len(df_clean):,} rows)")


if __name__ == "__main__":
    main()
