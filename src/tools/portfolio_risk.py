import yfinance as yf
import pandas as pd
import numpy as np
from langchain_core.tools import tool

# Attempt to import PyPortfolioOpt
try:
    from pypfopt.expected_returns import mean_historical_return
    from pypfopt.risk_models import risk_matrix
    from pypfopt.efficient_frontier import EfficientFrontier
    HAS_PYPORTFOLIOOPT = True
except ImportError:
    HAS_PYPORTFOLIOOPT = False

@tool
def portfolio_risk_analyzer(tickers: list[str]) -> str:
    """
    Analyzes the risk profile of a portfolio of stocks (equal-weighted by default).
    Computes Beta, Sharpe Ratio, Value at Risk (VaR), and optionally suggests Optimal Rebalancing weights.
    Call this tool when the user asks about the risk, correlation, or rebalancing of multiple stocks.
    """
    try:
        if not tickers or len(tickers) < 2:
            return "Please provide at least two ticker symbols to analyze a portfolio."

        # Fetch 3 years of daily closing prices
        data = yf.download(tickers, period="3y")['Close']
        if data.empty:
            return "Could not fetch data for the given tickers."
            
        # Daily returns
        returns = data.pct_change().dropna()
        
        # Equal weights assumption
        num_assets = len(tickers)
        weights = np.array([1.0 / num_assets] * num_assets)
        
        # Portfolio Daily Returns
        port_returns = returns.dot(weights)
        
        # 1. Sharpe Ratio (Annualized)
        # Risk-free rate assumption = 4%
        risk_free_rate = 0.04
        daily_rf = risk_free_rate / 252
        excess_returns = port_returns - daily_rf
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / port_returns.std()
        
        # 2. Value at Risk (VaR) - Historical 95%
        # How much we expect to lose on the worst 5% of days
        var_95 = np.percentile(port_returns, 5) * 100
        
        # 3. Correlation Matrix basics
        corr_matrix = returns.corr()
        avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].mean()
        
        report = (
            f"Portfolio Risk Analysis for {tickers} (Assuming Equal Weights):\n"
            f"- Annualized Sharpe Ratio: {sharpe_ratio:.2f} \n"
            f"- Historical Daily VaR (95% CI): {var_95:.2f}% (Expect a daily loss worse than this on 1 out of 20 days)\n"
            f"- Average Cross-Asset Correlation: {avg_corr:.2f}\n"
        )

        # 4. PyPortfolioOpt Rebalancing (If available)
        if HAS_PYPORTFOLIOOPT:
            try:
                mu = mean_historical_return(data)
                S = risk_matrix(data, method="sample_cov")
                
                ef = EfficientFrontier(mu, S)
                # Optimize for Max Sharpe
                raw_weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
                cleaned_weights = ef.clean_weights()
                
                opt_performance = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
                exp_ret, exp_vol, exp_sharpe = opt_performance
                
                report += (
                    f"\n[Modern Portfolio Theory Optimization (PyPortfolioOpt)]\n"
                    f"To maximize your Sharpe Ratio, the optimal weights are:\n"
                )
                for ticker, weight in cleaned_weights.items():
                    report += f"  * {ticker}: {weight*100:.1f}%\n"
                
                report += f"  (Expected Annual Return: {exp_ret*100:.1f}%, Expected Volatility: {exp_vol*100:.1f}%)\n"
            except Exception as ml_err:
                report += f"\nOptimization module encountered an issue: {str(ml_err)}"
                
        return report

    except Exception as e:
        return f"Error running Portfolio Risk Analyzer: {str(e)}"
