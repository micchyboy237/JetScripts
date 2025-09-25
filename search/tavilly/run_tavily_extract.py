from jet.logger import logger
from jet.transformers.formatters import format_json
from langchain_tavily.tavily_extract import TavilyExtract
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

async def extract_api_docs(urls: List[str]) -> Dict[str, Any]:
    """
    Extract content from a list of API documentation URLs.
    
    Args:
        urls: List of URLs to extract content from.
    
    Returns:
        Dictionary containing extracted results and failed URLs.
    """
    extractor = TavilyExtract(
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        extract_depth="advanced",  # For tables and embedded content
        include_images=True,      # Include images for visual references
        include_favicon=False,     # Skip favicons to reduce response size
        format="markdown"         # Return content in markdown format
    )
    return await extractor._arun(urls=urls)

# Example usage
urls = [
    "https://docs.tavily.com/documentation/api-reference/endpoint/crawl"
]
result = asyncio.run(extract_api_docs(urls))

logger.gray("Result:")
logger.success(format_json(result))
save_file(result, f"{OUTPUT_DIR}/result.json")