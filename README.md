# 💰 Finance Copilot

**An AI-powered personal finance assistant that analyzes your bank transactions using a multi-agent LangGraph pipeline to categorize spending, detect anomalies, and deliver actionable insights.**

---

## 📌 How It Works

Finance Copilot uses **4 specialized AI agents** orchestrated through a **LangGraph state machine**. Each agent handles one step of the analysis pipeline, passing its results forward through a shared state:

1. **Ingestion Agent** — Loads raw transaction data from a CSV file, validates the schema, parses dates, and cleans the data for downstream processing.

2. **Categorization Agent** — Classifies each transaction into spending categories (Food & Dining, Transport, Shopping, Subscriptions, Income) using intelligent keyword matching.

3. **Anomaly Detection Agent** — Scans for unusual spending patterns using statistical z-score analysis and flags potential duplicate charges within configurable time windows.

4. **Insights Agent** — Leverages an LLM (via Groq) to answer natural-language questions about your finances, referencing real numbers and category breakdowns from your data.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph StateGraph                        │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌───────────┐    ┌───────┐│
│  │          │    │              │    │           │    │       ││
│  │  Ingest  │───▶│  Categorize  │───▶│  Detect   │───▶│Answer ││
│  │          │    │              │    │           │    │       ││
│  └──────────┘    └──────────────┘    └───────────┘    └───────┘│
│       ▲                                                   │    │
│       │                                                   ▼    │
│   CSV File                                          LLM Answer │
└─────────────────────────────────────────────────────────────────┘

State: {csv_path, transactions, categorized_transactions, anomalies, user_question, answer}
```

**Data Flow:**
```
CSV File ──▶ Ingestion ──▶ Categorization ──▶ Anomaly Detection ──▶ AI Insights
  (raw)       (clean)       (labeled)          (flagged)             (answer)
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python** | Core language |
| **LangChain** | LLM framework & prompt engineering |
| **LangGraph** | Multi-agent orchestration & state machines |
| **Groq (Llama 3.3 70B)** | Fast LLM inference for financial Q&A |
| **Pandas** | Data manipulation & analysis |
| **Streamlit** | Interactive web dashboard |
| **Plotly** | Dynamic charts & visualizations |

---

## 🚀 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/hadeedzahid/finance-copilot.git
cd finance-copilot
```

### 2. Create your environment file
```bash
cp .env.example .env
```
Edit `.env` and add your API key:
```
GROQ_API_KEY=your_groq_api_key_here
```
> Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app

**Streamlit Dashboard (recommended):**
```bash
streamlit run app.py
```

**CLI Pipeline:**
```bash
python main.py
```

---

## 📊 Sample Output

### Anomaly Detection
```
⚠️  ANOMALIES DETECTED
════════════════════════════════════════════════════════
  🔴 Costco — $215.60 — Unusually high transaction (z-score: 2.69)
  🔴 Best Buy — $249.99 — Unusually high transaction (z-score: 3.26)
  ✅ No duplicate charges found
```

### Q&A Example
```
❓ Question: "What is my biggest spending category?"

💡 Answer: Your biggest spending category is Shopping, with a total
   of $1,180.60 spent across 9 transactions, accounting for
   approximately 62.7% of your total spending.
```

### Category Breakdown
```
📂 Spending by Category:
  - Shopping:       $1,180.60  (9 transactions)
  - Food & Dining:    $585.29  (14 transactions)
  - Transport:        $256.40  (7 transactions)
  - Subscriptions:    $117.95  (6 transactions)
```

---

## 🎯 Skills Demonstrated

- **Multi-Agent Orchestration** — Designed and implemented 4 specialized agents that collaborate through a shared pipeline
- **LangGraph State Machines** — Built a `StateGraph` with typed state, sequential edges, and compiled execution
- **Prompt Engineering** — Crafted system/human prompt templates for financial analysis and Q&A tasks
- **NLP on Financial Data** — Applied LLM reasoning to real-world transaction categorization and anomaly detection
- **Streamlit Deployment** — Created an interactive dashboard with real-time charts, metrics, and AI-powered insights

---

## 📁 Project Structure

```
finance-copilot/
├── agents/
│   ├── ingestion_agent.py        # CSV loading & validation
│   ├── categorization_agent.py   # Transaction classification
│   ├── anomaly_agent.py          # Z-score anomaly detection
│   └── insights_agent.py         # LLM-powered Q&A & insights
├── data/
│   └── sample_transactions.csv   # 40 realistic bank transactions
├── app.py                        # Streamlit dashboard
├── main.py                       # LangGraph pipeline entry point
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
└── README.md
```

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/hadeedzahid">Hadeed Zahid</a>
</p>
