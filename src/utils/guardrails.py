import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/audit.log"),
        logging.StreamHandler()
    ]
)
audit_logger = logging.getLogger("AuditLogger")

FINANCIAL_DISCLAIMER = """
========================================
IMPORTANT DISCLAIMER:
This financial assistant is built for educational and research purposes only. 
Nothing it produces constitutes financial, investment, legal, or tax advice. 
Always consult a qualified financial advisor before making any investment decisions.
========================================
"""

def inject_disclaimer(response: str) -> str:
    """Appends the mandatory financial disclaimer to the response."""
    return f"{response}\n\n{FINANCIAL_DISCLAIMER}"

def log_interaction(query: str, retrieved_sources: list, response: str):
    """Audit logging guardrail. Logs the interaction for compliance."""
    source_info = [doc.metadata.get("source", "unknown") for doc in retrieved_sources]
    log_entry = (
        f"\n[QUERY]: {query}\n"
        f"[SOURCES]: {source_info}\n"
        f"[RESPONSE]: {response[:200]}... (truncated)\n"
    )
    audit_logger.info(log_entry)
