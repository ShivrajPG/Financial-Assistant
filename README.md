# Financial RAG Agent

A locally-running financial assistant agent that uses LangGraph, LangChain, and Google Gemini to analyze market data, pull SEC filings, and run predictive ML models. 

The goal of this project is to build an AI agent that doesn't just hallucinate stock prices. Instead, it relies on a strict Tool-Calling framework to retrieve actual numbers before generating its analysis, guarded by multiple compliance guardrails.

## Features
- **RAG for SEC Filings**: Downloads full 10-K filings from the SEC EDGAR API, chunks them using BeautifulSoup, and stores them in a local ChromaDB instance to answer qualitative risk questions.
- **Real-Time Technicals**: Uses `yfinance` to grab daily closes and computes indicators like the 20-Day SMA and 14-Day RSI.
- **Predictive ML engine**: Automatically trains a local `xgboost` classifier on the last 2 years of daily returns and volatility to spit out a probability interval for tomorrow's price direction.
- **Portfolio Analytics**: Built with `PyPortfolioOpt` to compute Historical VaR (95%) and Sharpe Ratios for multi-stock portfolios.
- **Strict Guardrails**: Prevents the agent from issuing "Buy/Sell/Hold" advice. It is forced to synthesize data into a "Convergence Analysis" instead.

## Setup Instructions

### 1. Environment
You'll need Python 3.11+. I recommend setting up a virtual environment:
```bash
python -m venv venv
source venv/Scripts/activate # on Windows

or

.\venv\Scripts\python.exe main.py
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Note for Windows users: if PyPortfolioOpt gives you cvxpy/C++ build errors, make sure you have the MSVC build tools installed, though the pre-built wheels should work on newer pythons).*

### 3. API Keys
Create a `.env` file in the root directory:
```
GEMINI_API_KEY="your_api_key_here"
```

## Running the Project

First, you need to embed at least one SEC filing into the local vector database so the agent has something to read.
Run the ingest script (e.g., for Apple):
```bash
python scripts/ingest.py AAPL
```
*(Warning: The ingestion truncates after 50 chunks to avoid hitting free-tier Gemini embedding API rate limits. If you have a paid tier, you can remove the truncation in `loader.py`)*

After ingestion is done, start the main CLI chat interface:
```bash
python main.py
```

## Audit Logging
To ensure we can reconstruct how the agent arrived at an answer, all interactions, retrieved sources, and model outputs are written to `data/audit.log`.

## Tools Used
- LangChain / LangGraph (Routing logic)
- ChromaDB (Local vector storage)
- Google Gemini 2.5 Flash / Embeddings
- XGBoost & Scikit-Learn
- yfinance & PyPortfolioOpt
#
