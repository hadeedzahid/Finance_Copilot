"""
Categorization Agent
---------------------
Classifies each transaction into spending categories using rule-based
matching and optionally an LLM for ambiguous merchants.
"""

import pandas as pd
from typing import Optional

# Rule-based category mapping
CATEGORY_RULES = {
    "Food & Dining": [
        "starbucks", "mcdonald", "chipotle", "panera", "chick-fil-a",
        "subway", "domino", "grubhub", "doordash", "whole foods",
        "trader joe", "kroger",
    ],
    "Transport": [
        "uber", "lyft", "shell", "bp gas", "chevron", "exxon",
    ],
    "Shopping": [
        "amazon", "target", "costco", "walmart", "zara", "h&m",
        "nike", "best buy",
    ],
    "Subscriptions": [
        "netflix", "spotify", "apple icloud", "adobe", "youtube premium",
        "chatgpt plus",
    ],
    "Income": [
        "salary", "freelance", "interest payment", "refund",
    ],
}


def _match_category(merchant: str) -> str:
    """Match a merchant name to a category using keyword rules."""
    merchant_lower = merchant.lower()
    for category, keywords in CATEGORY_RULES.items():
        for keyword in keywords:
            if keyword in merchant_lower:
                return category
    return "Other"


def categorize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'category' column to the transactions DataFrame.

    Args:
        df: DataFrame with at least a 'merchant' column.

    Returns:
        DataFrame with an added 'category' column.
    """
    df = df.copy()
    df["category"] = df["merchant"].apply(_match_category)
    return df


def get_category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get spending breakdown by category.

    Args:
        df: Categorized transactions DataFrame.

    Returns:
        DataFrame with total amount per category, sorted descending.
    """
    breakdown = (
        df[df["type"] == "debit"]
        .groupby("category")["amount"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "total_spent", "count": "num_transactions"})
        .sort_values("total_spent", ascending=False)
        .round(2)
    )
    return breakdown


if __name__ == "__main__":
    from ingestion_agent import load_transactions

    transactions = load_transactions()
    categorized = categorize_transactions(transactions)
    print(categorized[["date", "merchant", "amount", "type", "category"]].head(15))
    print("\nCategory Breakdown:")
    print(get_category_breakdown(categorized))
