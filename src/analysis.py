"""Fraud-analysis statistics without Streamlit dependencies.

The provided Time is elapsed seconds since the first observation; it is *not*
a clock time, timezone or date. Amount has no stated currency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
REQUIRED = FEATURES + ["Class"]
AMOUNT_BINS = [-0.001, 0, 10, 50, 100, 500, 1000, float("inf")]
AMOUNT_LABELS = ["0", "(0, 10]", "(10, 50]", "(50, 100]", "(100, 500]", "(500, 1000]", ">1000"]


def validate_data(df: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {', '.join(missing)}")
    if not df[REQUIRED].apply(pd.api.types.is_numeric_dtype).all():
        raise ValueError("Todas as variáveis devem ser numéricas")
    if df[REQUIRED].isna().any().any():
        raise ValueError("Existem valores ausentes; verifique o arquivo antes da análise")
    if not df["Class"].isin([0, 1]).all():
        raise ValueError("A coluna Class deve conter apenas 0 e 1")
    if (df["Amount"] < 0).any():
        raise ValueError("A coluna Amount contém valores negativos")


def summarize_data(df: pd.DataFrame) -> dict:
    """Generate dataset-level summary and aggregates, safe to publish without raw CSV."""
    validate_data(df)
    n = len(df)
    fraud = int(df["Class"].sum())
    by_class = {}
    for label in (0, 1):
        subset = df.loc[df["Class"] == label, "Amount"]
        by_class[str(label)] = {
            "count": int(len(subset)),
            "amount_mean": float(subset.mean()),
            "amount_median": float(subset.median()),
            "amount_p90": float(subset.quantile(0.90)),
        }

    # Relative-hour buckets; not real local time-of-day.
    relative_hour = ((df["Time"] // 3600) % 24).astype(int)
    hours = (
        df.assign(relative_hour=relative_hour)
        .groupby("relative_hour")["Class"]
        .agg(["sum", "count"])
        .reindex(range(24), fill_value=0)
    )
    hourly = [
        {
            "hour": int(hour),
            "fraud_count": int(row["sum"]),
            "count": int(row["count"]),
            "fraud_rate_pct": float(row["sum"] / row["count"] * 100) if row["count"] else 0.0,
        }
        for hour, row in hours.iterrows()
    ]

    amount_bucket = pd.cut(df["Amount"], bins=AMOUNT_BINS, labels=AMOUNT_LABELS, include_lowest=True)
    amounts = df.assign(amount_bucket=amount_bucket).groupby("amount_bucket", observed=False)["Class"].agg(["sum", "count"])
    amount_groups = [
        {
            "range": str(label),
            "fraud_count": int(amounts.loc[label, "sum"]),
            "count": int(amounts.loc[label, "count"]),
            "fraud_rate_pct": float(amounts.loc[label, "sum"] / amounts.loc[label, "count"] * 100)
            if amounts.loc[label, "count"] else 0.0,
        }
        for label in AMOUNT_LABELS
    ]

    return {
        "dataset": "Credit Card Fraud Detection (anonymized)",
        "source_note": "Time é relativo ao início da coleta. Amount não informa moeda. Dados históricos, não de produção.",
        "n_rows": n,
        "n_columns": len(df.columns),
        "fraud_count": fraud,
        "legitimate_count": n - fraud,
        "fraud_rate_pct": float(fraud / n * 100),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
        "time_max_seconds": float(df["Time"].max()),
        "amount_mean": float(df["Amount"].mean()),
        "amount_median": float(df["Amount"].median()),
        "by_class": by_class,
        "hourly": hourly,
        "amount_groups": amount_groups,
    }
