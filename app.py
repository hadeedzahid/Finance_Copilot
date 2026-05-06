"""
Finance Copilot — Streamlit Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from agents.ingestion_agent import load_transactions, get_summary
from agents.categorization_agent import categorize_transactions, get_category_breakdown
from agents.anomaly_agent import get_anomaly_report
from agents.insights_agent import build_financial_summary, generate_insights

st.set_page_config(page_title="Finance Copilot", page_icon="💰", layout="wide")
st.title("💰 Finance Copilot")
st.caption("AI-powered personal finance analysis")

# Load and process data
@st.cache_data
def load_and_process():
    df = load_transactions()
    summary = get_summary(df)
    categorized = categorize_transactions(df)
    breakdown = get_category_breakdown(categorized)
    anomalies = get_anomaly_report(categorized)
    return df, summary, categorized, breakdown, anomalies

df, summary, categorized, breakdown, anomalies = load_and_process()

# --- Metrics Row ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", summary["total_transactions"])
col2.metric("Total Spending", f"${summary['total_debits']:,.2f}")
col3.metric("Total Income", f"${summary['total_credits']:,.2f}")
net = summary["total_credits"] - summary["total_debits"]
col4.metric("Net Cash Flow", f"${net:,.2f}", delta=f"${net:,.2f}")

st.divider()

# --- Charts ---
left, right = st.columns(2)

with left:
    st.subheader("Spending by Category")
    fig_pie = px.pie(
        breakdown.reset_index(),
        values="total_spent",
        names="category",
        hole=0.4,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with right:
    st.subheader("Daily Spending Trend")
    daily = categorized[categorized["type"] == "debit"].groupby(
        categorized["date"].dt.date
    )["amount"].sum().reset_index()
    daily.columns = ["date", "amount"]
    fig_line = px.area(daily, x="date", y="amount")
    st.plotly_chart(fig_line, use_container_width=True)

st.divider()

# --- Anomalies ---
st.subheader("⚠️ Anomaly Detection")
if anomalies["high_value_anomalies"] > 0 or anomalies["potential_duplicates"] > 0:
    for flag in anomalies["zscore_flags"]:
        st.warning(f"**{flag['merchant']}** — ${flag['amount']} — {flag['reason']}")
    for flag in anomalies["duplicate_flags"]:
        st.info(f"🔁 **{flag['merchant']}** — ${flag['amount']} — {flag['reason']}")
else:
    st.success("No anomalies detected!")

st.divider()

# --- AI Insights ---
st.subheader("🤖 AI Financial Insights")
if st.button("Generate Insights", type="primary"):
    with st.spinner("Analyzing your finances..."):
        fin_summary = build_financial_summary(summary, breakdown, anomalies)
        insights = generate_insights(fin_summary)
        st.markdown(insights)

st.divider()

# --- Raw Data ---
with st.expander("📄 View Raw Transactions"):
    st.dataframe(categorized, use_container_width=True)
