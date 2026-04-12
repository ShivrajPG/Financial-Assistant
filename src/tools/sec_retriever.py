from langchain_core.tools import create_retriever_tool
from src.retrieval.retriever import get_financial_retriever

def get_sec_retriever_tool():
    """Builds a formal Langchain Tool from the SEC filings Chroma DB."""
    # We call the function from Phase 1 to get our pre-configured ChromaDB retriever
    retriever = get_financial_retriever(k=5)
    
    # Wrap it in a Langchain Tool format with an explicit description
    name = "sec_10k_filings_search"
    description = (
        "Useful for searching through a company's SEC EDGAR 10-K or 10-Q filings. "
        "Use this tool when you need fundamental data, qualitative context, risk factors, "
        "or historical company performance numbers directly from SEC documents."
    )
    
    return create_retriever_tool(
        retriever,
        name,
        description
    )
