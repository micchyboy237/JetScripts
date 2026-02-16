# run_web_search_tool.py
import shutil
from pathlib import Path

from jet.libs.smolagents.custom_models import OpenAIModel
from jet.libs.smolagents.tools.searxng_search_tool import SearXNGSearchTool
from jet.libs.smolagents.tools.web_crawler_tool import crawl_web_page
from smolagents import LogLevel, ToolCallingAgent

OUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUT_DIR, ignore_errors=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

model = OpenAIModel(
    temperature=0.7,
    max_tokens=2048,
)

# Create tool instance with logging enabled
web_search = SearXNGSearchTool(max_results=10)

agent = ToolCallingAgent(
    tools=[web_search, crawl_web_page],
    model=model,
    add_base_tools=False,
    verbosity_level=LogLevel.DEBUG,  # or LogLevel.INFO
)

if __name__ == "__main__":
    # You can change the query to anything you want to test
    question = (
        "top 10 ongoing isekai and romcom anime 2026 with release dates and plots"
    )
    # question = "latest news about MRT-3 rehabilitation Philippines"

    print(f"\nRunning query: {question}\n")
    result = agent.run(question)
    print("\n" + "=" * 70)
    print("Final agent answer:")
    print(result)
    print("=" * 70 + "\n")

    print(f"Logs saved in: {OUT_DIR / 'web_search_logs'}")
    print("Look for folders like call_0001/ containing:")
    print("  • request.json")
    print("  • response.json")
    print("  • full_results.md  ← full markdown returned to the agent")
