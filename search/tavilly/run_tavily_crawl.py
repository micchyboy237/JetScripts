from jet.logger import logger
from jet.transformers.formatters import format_json
from langchain_tavily.tavily_crawl import TavilyCrawl
from typing import Dict, Any
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

async def crawl_api_docs() -> Dict[str, Any]:
    crawler = TavilyCrawl(
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        max_depth=2,
        max_breadth=10,
        limit=30,
        # instructions="API documentation",
        # select_paths=["/documentation/.*"],
        # exclude_paths=["/blog/.*"],
        include_images=True,
        # categories=["Documentation"],
        extract_depth="advanced"
    )
    return await crawler._arun(url="https://docs.tavily.com/documentation/api-reference/endpoint/crawl")

# Run the crawl
result = asyncio.run(crawl_api_docs())

logger.gray("Result:")
logger.success(format_json(result))
save_file(result, f"{OUTPUT_DIR}/result.json")