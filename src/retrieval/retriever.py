from src.ingestion.embedder import ChromaEmbedder

def get_financial_retriever(k: int = 5):
    """
    Returns the retriever interface for the SEC filings vector store.
    Assumes that the database has already been populated.
    """
    embedder = ChromaEmbedder()
    return embedder.get_retriever(k=k)
