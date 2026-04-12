import sys
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.agent.chain import FinancialRAGChain
from src.utils.guardrails import inject_disclaimer, log_interaction

def main():
    load_dotenv()
    console = Console()
    
    if "GEMINI_API_KEY" not in os.environ:
        console.print("[bold red]Error:[/bold red] GEMINI_API_KEY not found in environment. Please check your .env file.")
        sys.exit(1)

    console.print(Panel.fit("[bold blue]Financial Assistant CLI - Phase 1[/bold blue]\nType 'exit' or 'quit' to close."))
    
    try:
        agent = FinancialRAGChain()
    except Exception as e:
        console.print(f"[bold red]Failed to initialize the agent. Did you run the ingestion script? Error: {e}[/bold red]")
        sys.exit(1)
        
    while True:
        try:
            query = console.input("\n[bold green]You:[/bold green] ")
            if query.lower() in ["exit", "quit"]:
                break
            if not query.strip():
                continue
                
            with console.status("[bold yellow]Analyzing SEC filings...", spinner="dots"):
                response, retrieved_docs = agent.invoke(query)
                
            # Apply Output Guardrails
            final_response = inject_disclaimer(response)
            log_interaction(query, retrieved_docs, final_response)
            
            console.print("\n[bold magenta]Assistant:[/bold magenta]")
            console.print(Markdown(final_response))
            
        except KeyboardInterrupt:
            console.print("\nExiting...")
            break
        except Exception as e:
            console.print(f"[bold red]Error during processing:[/bold red] {e}")

if __name__ == "__main__":
    main()
