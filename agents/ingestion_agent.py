"""
Ingestion Agent
----------------
Responsible for loading and parsing raw transaction data from CSV files.
Validates data integrity and prepares it for downstream agents.
"""

import pandas as pd
from typing import Optional


def load_transactions(filepath: str = "data/sample_transactions.csv") -> pd.DataFrame:
    """
    Load transaction data from a CSV file.

    Args:
        filepath: Path to the CSV file containing transaction data.

    Returns:
        A cleaned pandas DataFrame with parsed dates and validated columns.
    """
    required_columns = {"date", "merchant", "amount", "type"}

    df = pd.read_csv(filepath)

    # Validate required columns
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse dates
    df["date"] = pd.to_datetime(df["date"])

    # Ensure amount is numeric
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Normalize type column
    df["type"] = df["type"].str.strip().str.lower()

    # Drop rows with critical missing values
    df.dropna(subset=["date", "merchant", "amount", "type"], inplace=True)

    # Sort by date
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def get_summary(df: pd.DataFrame) -> dict:
    """
    Generate a quick summary of the loaded transaction data.

    Args:
        df: DataFrame of transactions.

    Returns:
        Dictionary with key stats about the dataset.
    """
    return {
        "total_transactions": len(df),
        "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}",
        "total_debits": round(df[df["type"] == "debit"]["amount"].sum(), 2),
        "total_credits": round(df[df["type"] == "credit"]["amount"].sum(), 2),
        "unique_merchants": df["merchant"].nunique(),
    }


if __name__ == "__main__":
    transactions = load_transactions()
    print(transactions.head())
    print("\nSummary:")
    for key, value in get_summary(transactions).items():
        print(f"  {key}: {value}")
