import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

class ChromaEmbedder:
    def __init__(self, persist_directory: str = "data/chroma_db"):
        # Make sure GEMINI_API_KEY is in the environment
        if "GEMINI_API_KEY" not in os.environ:
            print("WARNING: GEMINI_API_KEY not found in environment. Embeddings may fail.")
            
        self.persist_directory = persist_directory
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        
        # Initialize chroma vector store
        self.vectorstore = Chroma(
            collection_name="sec_filings",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def embed_and_store(self, documents: list[Document]):
        """Embeds a list of documents and stores them into ChromaDB."""
        if not documents:
            print("No documents to embed.")
            return
            
        print(f"Embedding {len(documents)} document chunks...")
        self.vectorstore.add_documents(documents)
        print(f"Successfully embedded and saved to {self.persist_directory}")
        
    def get_retriever(self, k: int = 5):
        """Returns a retriever interface for the vector store."""
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
