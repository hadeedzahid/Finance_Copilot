"""
Anomaly Detection Agent
------------------------
Detects unusual spending patterns and flags potential anomalies
using statistical methods (z-score, IQR) on transaction data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict


def detect_anomalies_zscore(df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    """
    Detect anomalous transactions using z-score method.

    Args:
        df: DataFrame with transaction data (must include 'amount' and 'type').
        threshold: Z-score threshold above which a transaction is flagged.

    Returns:
        DataFrame of flagged anomalous transactions.
    """
    debits = df[df["type"] == "debit"].copy()

    if debits.empty:
        return pd.DataFrame()

    mean_amount = debits["amount"].mean()
    std_amount = debits["amount"].std()

    if std_amount == 0:
        return pd.DataFrame()

    debits["z_score"] = ((debits["amount"] - mean_amount) / std_amount).round(2)
    anomalies = debits[debits["z_score"].abs() > threshold].copy()
    anomalies["reason"] = anomalies.apply(
        lambda row: f"Unusually {'high' if row['z_score'] > 0 else 'low'} "
                     f"transaction (z-score: {row['z_score']})",
        axis=1,
    )

    return anomalies


def detect_duplicate_charges(df: pd.DataFrame, window_days: int = 3) -> pd.DataFrame:
    """
    Detect potential duplicate charges (same merchant, same amount within N days).

    Args:
        df: DataFrame with transaction data.
        window_days: Number of days to look for duplicates.

    Returns:
        DataFrame of potential duplicate transactions.
    """
    debits = df[df["type"] == "debit"].copy()
    debits = debits.sort_values("date")
    duplicates = []

    for _, group in debits.groupby(["merchant", "amount"]):
        if len(group) < 2:
            continue
        dates = group["date"].sort_values()
        for i in range(1, len(dates)):
            diff = (dates.iloc[i] - dates.iloc[i - 1]).days
            if diff <= window_days:
                duplicates.append(group.iloc[i].name)

    if duplicates:
        result = debits.loc[duplicates].copy()
        result["reason"] = "Potential duplicate charge (same merchant & amount within "  \
                           f"{window_days} days)"
        return result

    return pd.DataFrame()


def get_anomaly_report(df: pd.DataFrame) -> Dict:
    """
    Generate a comprehensive anomaly report.

    Args:
        df: Full transactions DataFrame.

    Returns:
        Dictionary containing anomaly findings.
    """
    zscore_anomalies = detect_anomalies_zscore(df)
    duplicate_anomalies = detect_duplicate_charges(df)

    return {
        "high_value_anomalies": len(zscore_anomalies),
        "potential_duplicates": len(duplicate_anomalies),
        "zscore_flags": zscore_anomalies[["date", "merchant", "amount", "reason"]].to_dict("records")
        if not zscore_anomalies.empty else [],
        "duplicate_flags": duplicate_anomalies[["date", "merchant", "amount", "reason"]].to_dict("records")
        if not duplicate_anomalies.empty else [],
    }


if __name__ == "__main__":
    from ingestion_agent import load_transactions

    transactions = load_transactions()
    report = get_anomaly_report(transactions)
    print("=== Anomaly Report ===")
    print(f"High-value anomalies: {report['high_value_anomalies']}")
    print(f"Potential duplicates: {report['potential_duplicates']}")
    for flag in report["zscore_flags"]:
        print(f"  ⚠ {flag['merchant']} — ${flag['amount']} — {flag['reason']}")
    for flag in report["duplicate_flags"]:
        print(f"  🔁 {flag['merchant']} — ${flag['amount']} — {flag['reason']}")
