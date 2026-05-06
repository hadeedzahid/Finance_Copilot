"""
Insights Agent
---------------
Generates natural-language financial insights and recommendations
by combining transaction analytics with LLM-powered analysis.
"""

import os
import pandas as pd
from typing import Dict, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

INSIGHTS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert personal finance advisor. Analyze the user's "
        "transaction data and provide actionable insights. Be specific, "
        "reference actual numbers, and give practical recommendations."
    )),
    ("human", (
        "Here is my financial summary for the period:\n\n"
        "{financial_summary}\n\n"
        "Please provide:\n"
        "1. A brief overview of my spending habits\n"
        "2. Top 3 areas where I could save money\n"
        "3. Any concerning patterns you notice\n"
        "4. A personalized budgeting tip"
    )),
])


def build_financial_summary(
    summary: Dict,
    category_breakdown: pd.DataFrame,
    anomaly_report: Dict,
) -> str:
    """
    Build a formatted financial summary string for the LLM.

    Args:
        summary: Output from ingestion_agent.get_summary().
        category_breakdown: Output from categorization_agent.get_category_breakdown().
        anomaly_report: Output from anomaly_agent.get_anomaly_report().

    Returns:
        Formatted string summarizing the user's finances.
    """
    lines = [
        f"📊 Transaction Overview:",
        f"  - Total transactions: {summary['total_transactions']}",
        f"  - Date range: {summary['date_range']}",
        f"  - Total debits (spending): ${summary['total_debits']:,.2f}",
        f"  - Total credits (income): ${summary['total_credits']:,.2f}",
        f"  - Net cash flow: ${summary['total_credits'] - summary['total_debits']:,.2f}",
        f"  - Unique merchants: {summary['unique_merchants']}",
        "",
        "📂 Spending by Category:",
    ]

    for category, row in category_breakdown.iterrows():
        lines.append(
            f"  - {category}: ${row['total_spent']:,.2f} "
            f"({int(row['num_transactions'])} transactions)"
        )

    if anomaly_report["high_value_anomalies"] > 0:
        lines.append("")
        lines.append("⚠️ Anomalies Detected:")
        for flag in anomaly_report["zscore_flags"]:
            lines.append(f"  - {flag['merchant']}: ${flag['amount']} — {flag['reason']}")

    if anomaly_report["potential_duplicates"] > 0:
        lines.append("")
        lines.append("🔁 Potential Duplicates:")
        for flag in anomaly_report["duplicate_flags"]:
            lines.append(f"  - {flag['merchant']}: ${flag['amount']} — {flag['reason']}")

    return "\n".join(lines)


def generate_insights(financial_summary: str, model: str = "llama-3.3-70b-versatile") -> str:
    """
    Use an LLM to generate personalized financial insights.

    Args:
        financial_summary: Formatted summary of the user's finances.
        model: Groq model to use.

    Returns:
        LLM-generated insights string.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return (
            "⚠️ GROQ_API_KEY not set. Here's your raw financial summary:\n\n"
            + financial_summary
        )

    llm = ChatGroq(model=model, temperature=0.7, groq_api_key=api_key)
    chain = INSIGHTS_PROMPT | llm
    response = chain.invoke({"financial_summary": financial_summary})
    return response.content


QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert personal finance advisor. Answer the user's question "
        "based on their transaction data. Be specific, reference actual numbers "
        "from the data, and keep your answer concise but helpful."
    )),
    ("human", (
        "Here is my transaction data summary:\n\n"
        "{transaction_context}\n\n"
        "My question: {question}"
    )),
])


def answer_finance_question(
    question: str,
    categorized_df: pd.DataFrame,
    model: str = "llama-3.3-70b-versatile",
) -> str:
    """
    Answer a user's finance question using categorized transaction data.

    Args:
        question: The user's natural-language question.
        categorized_df: DataFrame with categorized transactions.
        model: Groq model to use.

    Returns:
        LLM-generated answer string.
    """
    # Build context from the categorized data
    debits = categorized_df[categorized_df["type"] == "debit"]
    category_totals = (
        debits.groupby("category")["amount"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "total_spent", "count": "num_transactions"})
        .sort_values("total_spent", ascending=False)
        .round(2)
    )

    total_spent = round(debits["amount"].sum(), 2)
    total_income = round(
        categorized_df[categorized_df["type"] == "credit"]["amount"].sum(), 2
    )

    context_lines = [
        f"Total spending: ${total_spent:,.2f}",
        f"Total income: ${total_income:,.2f}",
        f"Net cash flow: ${total_income - total_spent:,.2f}",
        f"Total transactions: {len(categorized_df)}",
        "",
        "Spending by category:",
    ]
    for category, row in category_totals.iterrows():
        pct = round(row["total_spent"] / total_spent * 100, 1) if total_spent else 0
        context_lines.append(
            f"  - {category}: ${row['total_spent']:,.2f} "
            f"({int(row['num_transactions'])} txns, {pct}%)"
        )

    context_lines.append("")
    context_lines.append("Recent transactions:")
    for _, row in categorized_df.tail(10).iterrows():
        context_lines.append(
            f"  - {row['date'].strftime('%Y-%m-%d')} | {row['merchant']} | "
            f"${row['amount']:.2f} | {row['type']} | {row['category']}"
        )

    transaction_context = "\n".join(context_lines)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return (
            f"⚠️ GROQ_API_KEY not set. Cannot answer: '{question}'\n\n"
            f"Here's your data context:\n{transaction_context}"
        )

    llm = ChatGroq(model=model, temperature=0.3, groq_api_key=api_key)
    chain = QA_PROMPT | llm
    response = chain.invoke({
        "transaction_context": transaction_context,
        "question": question,
    })
    return response.content


if __name__ == "__main__":
    from ingestion_agent import load_transactions, get_summary
    from categorization_agent import categorize_transactions, get_category_breakdown
    from anomaly_agent import get_anomaly_report

    df = load_transactions()
    summary = get_summary(df)
    categorized = categorize_transactions(df)
    breakdown = get_category_breakdown(categorized)
    anomalies = get_anomaly_report(df)

    fin_summary = build_financial_summary(summary, breakdown, anomalies)
    print(fin_summary)
    print("\n" + "=" * 60)
    print("🤖 AI Insights:")
    print("=" * 60)
    print(generate_insights(fin_summary))
