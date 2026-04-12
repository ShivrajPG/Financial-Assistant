import yfinance as yf
import pandas as pd
from langchain_core.tools import tool

def compute_rsi(data: pd.Series, periods: int = 14) -> pd.Series:
    """Calculates Relative Strength Index."""
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=periods, min_periods=periods).mean()
    avg_loss = loss.rolling(window=periods, min_periods=periods).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

@tool
def get_live_stock_data(ticker: str) -> str:
    """
    Fetches the current stock price and key real-time indicators (RSI) for a given ticker symbol.
    Call this tool when the user asks about the current price, recent performance, or live market data for a stock.
    """
    try:
        # get 3 months of data to ensure rolling windows compute right
        stock = yf.Ticker(ticker)
        df = stock.history(period="3mo")
        
        if df.empty:
            return f"No live market data found for ticker '{ticker}'."
            
        current_price = df['Close'].iloc[-1]
        
        # calculate rsi manaually because pandas-ta broke on windows
        df['RSI'] = compute_rsi(df['Close'])
        
        # simple moving average fallback for now
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        latest = df.iloc[-1]
        
        # Formatting the response
        rsi_val = latest.get('RSI', 'N/A')
        sma_val = latest.get('SMA_20', 'N/A')
        
        report = (
            f"Live Market Data for {ticker.upper()}:\n"
            f"- Current Price (Close): ${current_price:.2f}\n"
            f"- RSI (14): {rsi_val if isinstance(rsi_val, str) else f'{rsi_val:.2f}'}\n"
            f"- 20-Day SMA: ${sma_val if isinstance(sma_val, str) else f'{sma_val:.2f}'}\n"
        )
        return report
    except Exception as e:
        return f"Error fetching market data for {ticker}: {str(e)}"
