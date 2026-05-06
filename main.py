"""
Finance Copilot — LangGraph Multi-Agent Pipeline
==================================================
Wires all 4 agents (ingest, categorize, detect, answer) into a
sequential StateGraph that runs end-to-end.
"""

from typing import TypedDict, Optional, Dict, List, Any
import pandas as pd

from langgraph.graph import StateGraph, END

from agents.ingestion_agent import load_transactions
from agents.categorization_agent import categorize_transactions
from agents.anomaly_agent import get_anomaly_report
from agents.insights_agent import answer_finance_question


# ── State Schema ──────────────────────────────────────────────

class PipelineState(TypedDict):
    csv_path: str
    transactions: Optional[pd.DataFrame]
    categorized_transactions: Optional[pd.DataFrame]
    anomalies: Optional[Dict[str, Any]]
    user_question: str
    answer: Optional[str]


# ── Node Functions ────────────────────────────────────────────

def ingest(state: PipelineState) -> dict:
    """Node 1: Load and validate transactions from CSV."""
    print("📥 [ingest] Loading transactions...")
    df = load_transactions(state["csv_path"])
    print(f"   ✅ Loaded {len(df)} transactions")
    return {"transactions": df}


def categorize(state: PipelineState) -> dict:
    """Node 2: Categorize each transaction by merchant."""
    print("🏷️  [categorize] Categorizing transactions...")
    categorized = categorize_transactions(state["transactions"])
    categories = categorized["category"].nunique()
    print(f"   ✅ Assigned {categories} categories")
    return {"categorized_transactions": categorized}


def detect(state: PipelineState) -> dict:
    """Node 3: Detect anomalies in categorized transactions."""
    print("🔍 [detect] Scanning for anomalies...")
    anomalies = get_anomaly_report(state["categorized_transactions"])
    print(f"   ✅ Found {anomalies['high_value_anomalies']} high-value anomalies")
    print(f"   ✅ Found {anomalies['potential_duplicates']} potential duplicates")
    return {"anomalies": anomalies}


def answer(state: PipelineState) -> dict:
    """Node 4: Answer user's finance question using LLM + transaction context."""
    print("🤖 [answer] Generating answer...")
    result = answer_finance_question(
        question=state["user_question"],
        categorized_df=state["categorized_transactions"],
    )
    print("   ✅ Answer generated")
    return {"answer": result}


# ── Build the Graph ───────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct the LangGraph StateGraph with 4 nodes connected
    in sequence: ingest → categorize → detect → answer → END
    """
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("ingest", ingest)
    graph.add_node("categorize", categorize)
    graph.add_node("detect", detect)
    graph.add_node("answer", answer)

    # Set entry point
    graph.set_entry_point("ingest")

    # Connect nodes in sequence
    graph.add_edge("ingest", "categorize")
    graph.add_edge("categorize", "detect")
    graph.add_edge("detect", "answer")
    graph.add_edge("answer", END)

    return graph


def compile_and_run(csv_path: str, question: str) -> PipelineState:
    """
    Compile the graph and execute the full pipeline.

    Args:
        csv_path: Path to the transactions CSV file.
        question: User's finance question.

    Returns:
        Final pipeline state with all results.
    """
    graph = build_graph()
    app = graph.compile()

    initial_state: PipelineState = {
        "csv_path": csv_path,
        "transactions": None,
        "categorized_transactions": None,
        "anomalies": None,
        "user_question": question,
        "answer": None,
    }

    print("🚀 Finance Copilot — LangGraph Pipeline Starting\n")
    print("=" * 60)

    final_state = app.invoke(initial_state)

    print("=" * 60)
    print("✅ Pipeline complete!\n")

    return final_state


# ── Test Run ──────────────────────────────────────────────────

if __name__ == "__main__":
    result = compile_and_run(
        csv_path="data/sample_transactions.csv",
        question="What is my biggest spending category?",
    )

    # Print anomalies
    print("\n" + "=" * 60)
    print("⚠️  ANOMALIES DETECTED")
    print("=" * 60)
    anomalies = result["anomalies"]
    if anomalies["high_value_anomalies"] > 0:
        for flag in anomalies["zscore_flags"]:
            print(f"  🔴 {flag['merchant']} — ${flag['amount']} — {flag['reason']}")
    else:
        print("  ✅ No high-value anomalies found")

    if anomalies["potential_duplicates"] > 0:
        for flag in anomalies["duplicate_flags"]:
            print(f"  🔁 {flag['merchant']} — ${flag['amount']} — {flag['reason']}")
    else:
        print("  ✅ No duplicate charges found")

    # Print final answer
    print("\n" + "=" * 60)
    print("💡 ANSWER")
    print("=" * 60)
    print(result["answer"])
    print("=" * 60)
