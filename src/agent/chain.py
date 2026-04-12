import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

from src.tools.market_data import get_live_stock_data
from src.tools.sec_retriever import get_sec_retriever_tool
from src.tools.forecasting import stock_price_predictor
from src.tools.portfolio_risk import portfolio_risk_analyzer

# We need a formal agent prompt that tells it how to use tools
AGENT_SYSTEM_PROMPT = """You are a highly capable AI Financial Analyst.
You have access to four tools:
1. `sec_10k_filings_search`: Use this to find qualitative fundamental data, risk factors, or historical information from a company's SEC 10-K filings.
2. `get_live_stock_data`: Use this to get real-time price and technical indicators (RSI, SMA) for a single stock ticker.
3. `stock_price_predictor`: Use this to fetch Machine Learning (XGBoost) probabilities for a single stock's short-term price direction.
4. `portfolio_risk_analyzer`: Use this to compute VaR, Sharpe Ratio, correlation, and optimal PyPortfolioOpt rebalancing for a list of multiple tickers.

You are strictly limited to discussing financial, investment, and market-related topics.
If a user asks about anything unrelated to finance (e.g., cooking, programming, general chit-chat), you must politely but firmly refuse to answer.

When summarizing portfolio risk from the `portfolio_risk_analyzer`, you must generate a full, easy-to-understand risk narrative explaining the metrics (e.g., what the Sharpe Ratio and VaR actually mean for the user's risk profile).

When you are asked to evaluate a stock using multiple sources (e.g. historical RAG, live data, and ML prediction), do NOT just provide fragmented data dumps. Instead, conclude your response with a "Multi-Factor Convergence Analysis".
In this analysis, you must examine the alignment (convergence) and disagreement (divergence) between the quantitative signals (RSI, SMA, ML probabilities) and the qualitative context (SEC 10-K risk factors).
For example: "The data sources demonstrate high bullish convergence across technical and predictive models, though SEC filings highlight a key regulatory risk."

When you include numerical claims from the SEC filings, you MUST cite the source.
CRITICAL GUARDRAIL: DO NOT hallucinate numerical data or predictions! Any forecasts or predictions of future stock prices MUST come exclusively from the `stock_price_predictor` tool. You must only explain the probabilities it outputs, never invent your own predictions. You cannot utter the words "buy", "sell", or "hold" as a definitive command or advice to the user.
"""

class FinancialRAGChain:
    def __init__(self, k: int = 5):
        # We use Gemini 2.5 Flash as it is supported in the free tier
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0  # Low temperature for analytical accuracy
        )
        
        # Tools array
        self.tools = [get_sec_retriever_tool(), get_live_stock_data, stock_price_predictor, portfolio_risk_analyzer]
        
        # Create the underlying LangGraph agent
        self.agent = create_react_agent(self.llm, tools=self.tools, prompt=AGENT_SYSTEM_PROMPT)
        
    def invoke(self, query: str):
        """Invoke the Agent and return the response."""
        # LangGraph invoke structure
        result = self.agent.invoke({"messages": [HumanMessage(content=query)]})
        
        # Extract final output string
        content = result["messages"][-1].content
        if isinstance(content, list):
            output = "".join([block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"])
        else:
            output = str(content)
        
        # To maintain the tuple interface
        mock_docs = []
        return output, mock_docs
