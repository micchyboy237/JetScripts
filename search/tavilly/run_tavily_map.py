from jet.logger import logger
from jet.transformers.formatters import format_json
from langchain_tavily.tavily_map import TavilyMap
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

async def map_website_structure(url: str) -> Dict[str, Any]:
    """
    Map the structure of a website starting from a root URL.
    
    Args:
        url: The root URL to start mapping.
    
    Returns:
        Dictionary containing the base URL, mapped URLs, and response time.
    """
    mapper = TavilyMap(
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        max_depth=2,              # Explore up to 2 link hops
        max_breadth=10,          # Follow up to 10 links per page
        limit=30,                # Process up to 30 URLs total
        # instructions="developer resources and documentation",
        # select_paths=["/documentation/.*", "/developers/.*"],
        exclude_paths=["/blog/.*"],
        # categories=["Documentation", "Developers"],
        allow_external=False      # Stay within the domain
    )
    return await mapper._arun(url=url)

# Example usage
url = "https://docs.tavily.com/documentation/api-reference/endpoint/crawl"
result = asyncio.run(map_website_structure(url))

logger.gray("Result:")
logger.success(format_json(result))
save_file(result, f"{OUTPUT_DIR}/result.json")