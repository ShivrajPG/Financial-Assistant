import os
from pathlib import Path
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class SECFilingLoader:
    def __init__(self, data_dir: str = "data/docs"):
        self.data_dir = Path(data_dir)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len,
        )

    def extract_text_from_html(self, html_content: str) -> str:
        """Extracts clean text from SEC HTML filing."""
        soup = BeautifulSoup(html_content, "lxml")
        return soup.get_text(separator="\n", strip=True)

    def load_and_chunk(self, ticker: str) -> list[Document]:
        """Loads all downloaded filings for a ticker and splits them into chunks."""
        documents = []
        ticker_dir = self.data_dir / "sec-edgar-filings" / ticker
        
        if not ticker_dir.exists():
            print(f"No filings found for {ticker} in {self.data_dir}")
            return []

        # Find all .txt files (sec-edgar-downloader saves full-submission.txt)
        for filepath in ticker_dir.rglob("*.txt"):
            print(f"Processing {filepath}...")
            try:
                # SEC filings can have various encodings, utf-8 is usually safe but fallback to latin-1
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(filepath, "r", encoding="latin-1") as f:
                    content = f.read()
                    
            clean_text = self.extract_text_from_html(content)
            
            # Use accession number as part of the source/page metadata
            accession_number = filepath.parent.name
            filing_type = filepath.parent.parent.name
            
            # Create Langchain Document
            doc = Document(
                page_content=clean_text,
                metadata={
                    "source": f"{ticker}_{filing_type}_{accession_number}",
                    "ticker": ticker,
                    "filing_type": filing_type
                }
            )
            # Split document
            chunks = self.text_splitter.split_documents([doc])
            
            # Add chunk index (acting as page number proxy)
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = i
                
            documents.extend(chunks)
            print(f"Created {len(chunks)} chunks from {filepath.name}")
            
        # Truncating to 50 chunks to prevent Gemini API Free Tier rate limits!
        documents = documents[:50]
        return documents

def chunk_documents(docs: list[Document]) -> list[Document]:
    """Utility function if needed separately."""
    pass
