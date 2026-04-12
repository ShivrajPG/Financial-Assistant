import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.sec_fetcher import SECFetcher
from src.ingestion.loader import SECFilingLoader
from src.ingestion.embedder import ChromaEmbedder

def main():
    load_dotenv()
    
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <TICKER1> [TICKER2] ...")
        sys.exit(1)
        
    tickers = sys.argv[1:]
    
    fetcher = SECFetcher()
    loader = SECFilingLoader()
    embedder = ChromaEmbedder()
    
    for ticker in tickers:
        ticker = ticker.upper()
        # 1. Fetch
        print(f"--- Processing {ticker} ---")
        fetcher.fetch_10k(ticker, amount=1)
        
        # 2. Load and Chunk
        documents = loader.load_and_chunk(ticker)
        
        # 3. Embed and Store
        embedder.embed_and_store(documents)
        
    print("Ingestion complete!")

if __name__ == "__main__":
    main()
