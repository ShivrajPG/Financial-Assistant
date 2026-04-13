# Financial RAG Agent

A locally-running financial assistant agent that uses LangGraph, LangChain, and Google Gemini to analyze market data, pull SEC filings, and run predictive ML models.

The goal of this project is to build an AI agent that doesn't just hallucinate stock prices. Instead, it relies on a strict Tool-Calling framework to retrieve actual numbers before generating its analysis, guarded by multiple compliance guardrails.

## Features
- **RAG for SEC Filings**: Downloads full 10-K filings from the SEC EDGAR API, chunks them using BeautifulSoup, and stores them in a local ChromaDB instance to answer qualitative risk questions.
- **Real-Time Technicals**: Uses yfinance to grab daily closes and computes indicators like the 20-Day SMA and 14-Day RSI.
- **Predictive ML engine**: Automatically trains a local xgboost classifier on the last 2 years of daily returns and volatility to spit out a probability interval for tomorrow's price direction.
- **Portfolio Analytics**: Built with PyPortfolioOpt to compute Historical VaR (95%) and Sharpe Ratios for multi-stock portfolios.
- **Strict Guardrails**: Prevents the agent from issuing "Buy/Sell/Hold" advice. It is forced to synthesize data into a "Convergence Analysis" instead.

## Setup Instructions

### Prerequisites
- Python 3.11 or higher (Python 3.14 may work but has compatibility warnings)
- A Google Gemini API key (get one from [Google AI Studio](https://aistudio.google.com/app/apikey))

### 1. Clone or Download the Repository
Ensure you have the project files in a directory.

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
**Note for Windows users**: If you encounter issues with PyPortfolioOpt or cvxpy, ensure you have the Microsoft Visual C++ Build Tools installed. However, pre-built wheels should work on recent Python versions.

### 4. Configure API Keys
Create a .env file in the root directory of the project:
```bash
GEMINI_API_KEY=your_actual_api_key_here
```
Replace your_actual_api_key_here with your real Google Gemini API key.

### 5. Ingest SEC Filings Data
Before running the agent, you need to populate the local vector database with SEC filings data. Run the ingestion script for one or more tickers (e.g., Apple):
```bash
python scripts/ingest.py AAPL
```
- This will download the latest 10-K filing for AAPL, chunk it, and embed it into ChromaDB.
- You can run it for multiple tickers: python scripts/ingest.py AAPL MSFT GOOGL
- **Warning**: The script truncates to 50 chunks to avoid hitting free-tier Gemini embedding API rate limits. If you have a paid API tier, you can modify src/ingestion/loader.py to remove the truncation.
- Ingestion may take several minutes depending on the filing size and your internet connection.

### 6. Run the Agent
Once ingestion is complete, start the interactive CLI:
```bash
python main.py
```
- Type your financial questions (e.g., "What are Apple's main risks?" or "Analyze AAPL's portfolio with MSFT").
- Type exit or quit to close the application.
## Sample Output

Here are examples of the agent's output to demonstrate its functionality:

**Agent Startup:**
```
╭───────────────────────────────────╮
│ Financial Assistant CLI - Phase 1 │
│ Type 'exit' or 'quit' to close.   │
╰───────────────────────────────────╯
```

**Sample Interaction - SEC Filing Query:**
```
You: What are Apple's main risks?

Assistant: Based on Apple's latest SEC 10-K filing, the main risks include supply chain disruptions, regulatory changes in key markets, competition in the smartphone and services sectors, and dependency on new product innovation. The filing highlights geopolitical tensions affecting manufacturing and potential tariffs impacting profitability.
```

**Sample Interaction - Technical Analysis:**
```
You: Analyze AAPL stock with current technical indicators.

Assistant: For AAPL, the current RSI is 65.2 (neutral to bullish), 20-day SMA is $182.45, and the latest close is $185.30. The ML model predicts a 58% probability of upward movement tomorrow based on historical patterns.
```

**Sample Interaction - Portfolio Analysis:**
```
You: Analyze a portfolio with AAPL and MSFT.

Assistant: Portfolio Analysis for AAPL and MSFT:
- Historical VaR (95%): 12.3%
- Sharpe Ratio: 1.85
- Optimal Weights: AAPL 45%, MSFT 55%
- Correlation: 0.72

This portfolio shows moderate risk with good diversification.
```
## Troubleshooting
- **Agent fails to initialize**: Ensure you've run the ingestion script and that your .env file is correctly set up.
- **API errors**: Verify your GEMINI_API_KEY is valid and has sufficient quota.
- **Import errors**: Make sure all dependencies are installed via pip install -r requirements.txt.
- **Python version issues**: Use Python 3.11-3.13 for best compatibility.

## Audit Logging
To ensure we can reconstruct how the agent arrived at an answer, all interactions, retrieved sources, and model outputs are written to data/audit.log.

## Tools Used
- LangChain / LangGraph (Routing logic)
- ChromaDB (Local vector storage)
- Google Gemini 2.5 Flash / Embeddings
- XGBoost & Scikit-Learn
- yfinance & PyPortfolioOpt
