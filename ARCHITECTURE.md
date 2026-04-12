# System Architecture

This document briefly explains how the different modules of the Financial Assistant piece together. 

## High-Level Flow
The agent is powered by a `LangGraph` ReAct loop. When a user asks a question via the CLI, the graph parses the prompt against `AGENT_SYSTEM_PROMPT` to decide which external tools it needs to fulfill the request.

If it needs fundamental SEC data -> calls `sec_10k_filings_search`
If it needs current pricing -> calls `get_live_stock_data`
If it needs predictive outlooks -> calls `stock_price_predictor`
If it needs multi-asset evaluation -> calls `portfolio_risk_analyzer`

Once the tools return their raw data (dataframes, dicts, strings), the LLM synthesizes everything into a "Convergence Analysis" and outputs it to the user.

## Directory Structure

```text
financial_assistant/
│
├── data/                       # Local DB and logging
│   ├── chroma_db/              # ChromaDB vector store files (ignored in git)
│   ├── docs/                   # Raw downloaded sec filings (ignored in git)
│   └── audit.log               # Append-only interaction logging
│
├── scripts/
│   └── ingest.py               # Runner to pull and embed a ticker's latest 10-K
│
├── src/
│   ├── ingestion/              # Data pipeline code
│   │   ├── sec_fetcher.py      # SEC EDGAR wrapper
│   │   ├── loader.py           # BeautifulSoup HTML cleaning and chunking
│   │   └── embedder.py         # Pushes chunks to ChromaDB
│   │
│   ├── retrieval/
│   │   └── retriever.py        # Connects agent to the local ChromaDB index
│   │
│   ├── tools/
│   │   ├── forecasting.py      # XGBoost classification logic
│   │   ├── market_data.py      # Pandas-based RSI and SMA calculators
│   │   ├── portfolio_risk.py   # PyPortfolioOpt integration for VaR/Sharpe
│   │   └── sec_retriever.py    # Wraps retrieval/retriever.py into a LangChain tool
│   │
│   ├── agent/
│   │   ├── chain.py            # The core LangGraph definition and System Prompt restrictors
│   │   └── prompt.py           # (Legacy) static prompt definitions
│   │
│   └── utils/
│       └── guardrails.py       # Intercepts agent outputs to enforce disclaimers and logging
│
├── main.py                     # Entry point for the interactive CLI loop
├── requirements.txt            # Package dependencies
└── .env                        # API keys
```

## Guardrails
To prevent the LLM from acting recklessly, three distinct layers of guardrails were implemented:

1. **Input / Behavioral Guardrail (System Prompting)**
   The agent's state modifier explicitly forbids it from generating its own market forecasts. It must rely entirely on the `xgboost` tool output. It is also linguistically banned from using definitive commands like "buy" or "sell".

2. **Output Guardrail (Compliance Interceptor)**
   Before any text hits the console, `guardrails.py` dynamically appends a hardcoded legal disclaimer to the output block.

3. **Audit Guardrail (Logging)**
   Every single transaction is dumped into `audit.log` alongside the EXACT chunks it retrieved from the DB. This prevents "black box" hallucination debugging.
