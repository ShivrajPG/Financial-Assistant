import os
from pathlib import Path
from sec_edgar_downloader import Downloader

class SECFetcher:
    def __init__(self, download_dir: str = "data/docs"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        # SEC requires a user agent format: "CompanyName ContactEmail"
        company_name = os.getenv("SEC_COMPANY_NAME", "FinancialAssistant")
        email = os.getenv("SEC_EMAIL", "research@financialassistant.local")
        self.downloader = Downloader(company_name, email, str(self.download_dir))

    def fetch_10k(self, ticker: str, amount: int = 1):
        """Downloads the latest 10-K filings for a given ticker."""
        print(f"Downloading {amount} 10-K filing(s) for {ticker}...")
        self.downloader.get("10-K", ticker, limit=amount)
        print(f"Download complete for {ticker}. Saved in {self.download_dir}")

    def fetch_10q(self, ticker: str, amount: int = 1):
        """Downloads the latest 10-Q filings for a given ticker."""
        print(f"Downloading {amount} 10-Q filing(s) for {ticker}...")
        self.downloader.get("10-Q", ticker, limit=amount)
        print(f"Download complete for {ticker}. Saved in {self.download_dir}")

if __name__ == "__main__":
    fetcher = SECFetcher()
    fetcher.fetch_10k("AAPL", amount=1)
