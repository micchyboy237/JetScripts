from jet.logger import logger
from jet.transformers.formatters import format_json
from langchain_tavily.tavily_search import TavilySearch
from typing import Dict, Any, List
from jet.file.utils import save_file
import asyncio
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

async def search_ai_news(query: str) -> Dict[str, Any]:
    """
    Search for recent AI news from trusted domains.
    
    Args:
        query: The search query to execute.
    
    Returns:
        Dictionary containing search results, images, and response time.
    """
    searcher = TavilySearch(
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        max_results=5,                   # Return up to 5 results
        search_depth="advanced",         # For in-depth results
        include_domains=["arstechnica.com", "techcrunch.com"],  # Trusted sources
        exclude_domains=["twitter.com"], # Exclude social media
        time_range="month",             # Results from the past month
        include_images=True,            # Include images for visual context
        topic="general",                # General topic for AI news
        include_answer=True,            # Provide a short answer
        include_raw_content="markdown"  # Return content in markdown
    )
    return await searcher._arun(query=query)

# Example usage
query = "recent advancements in AI 2025"
result = asyncio.run(search_ai_news(query))

logger.gray("Result:")
logger.success(format_json(result))
save_file(result, f"{OUTPUT_DIR}/result.json")