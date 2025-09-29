from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.search.playwright.playwright_search import PlaywrightSearch
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

async def search_ai_news(query: str) -> Dict[str, Any]:
    """
    Search for recent AI news from trusted domains.
    Args:
        query: The search query to execute.
    Returns:
        Dictionary containing search results, images, and response time.
    """
    searcher = PlaywrightSearch(
        max_results=5,
        search_depth="advanced",
        include_domains=["arstechnica.com", "techcrunch.com"],
        exclude_domains=["twitter.com"],
        time_range="month",
        include_images=False,
        topic="general",
        include_answer=True,
        include_raw_content="markdown"
    )
    return await searcher._arun(query=query)

async def search_finance_updates(query: str) -> Dict[str, Any]:
    """
    Search for recent finance updates with date range.
    Args:
        query: The search query to execute.
    Returns:
        Dictionary containing search results, images, and response time.
    """
    searcher = PlaywrightSearch(
        max_results=3,
        search_depth="basic",
        include_domains=["bloomberg.com", "wsj.com"],
        time_range="week",
        topic="finance",
        include_images=False,
        include_answer="basic",
        include_raw_content="text",
        start_date="2025-09-01",
        end_date="2025-09-28"
    )
    return await searcher._arun(query=query)

def sync_search_example(query: str) -> Dict[str, Any]:
    """
    Demonstrate synchronous usage of PlaywrightSearch.
    Args:
        query: The search query to execute.
    Returns:
        Dictionary containing search results, images, and response time.
    """
    searcher = PlaywrightSearch(
        max_results=4,
        search_depth="basic",
        include_images=False,
        include_favicon=False,
        topic="news"
    )
    result = searcher._run(query=query)
    logger.gray("Synchronous search result:")
    logger.success(format_json(result))
    save_file(result, f"{OUTPUT_DIR}/sync_result.json")
    return result

async def async_search_example(query: str) -> Dict[str, Any]:
    """
    Demonstrate asynchronous usage of PlaywrightSearch.
    Args:
        query: The search query to execute.
    Returns:
        Dictionary containing search results, images, and response time.
    """
    searcher = PlaywrightSearch(
        max_results=5,
        search_depth="advanced",
        include_domains=["theverge.com"],
        include_images=False,
        include_favicon=False,
        topic="general",
        include_answer="advanced",
        include_raw_content="markdown"
    )
    result = await searcher._arun(query=query)
    logger.gray("Asynchronous search result:")
    logger.success(format_json(result))
    save_file(result, f"{OUTPUT_DIR}/async_result.json")
    return result

if __name__ == "__main__":
    query = "recent advancements in AI 2025"
    
    # Synchronous example
    print("Running synchronous search example...")
    sync_search_result = sync_search_example(query)
    print(f"Found {len(sync_search_result['results'])} results")
    print(f"Response time: {sync_search_result['response_time']:.2f} seconds")
    
    # # Asynchronous examples
    # print("\nRunning asynchronous search examples...")
    # async def run_async_examples():
    #     ai_news_result = await search_ai_news(query)
    #     print(f"AI news search found {len(ai_news_result['results'])} results")
    #     print(f"Response time: {ai_news_result['response_time']:.2f} seconds")
    #     save_file(ai_news_result, f"{OUTPUT_DIR}/ai_news_result.json")
        
    #     finance_query = "stock market trends 2025"
    #     finance_result = await search_finance_updates(finance_query)
    #     print(f"Finance search found {len(finance_result['results'])} results")
    #     print(f"Response time: {finance_result['response_time']:.2f} seconds")
    #     save_file(finance_result, f"{OUTPUT_DIR}/finance_result.json")
        
    #     async_result = await async_search_example(query)
    #     print(f"Async example search found {len(async_result['results'])} results")
    #     print(f"Response time: {async_result['response_time']:.2f} seconds")
    
    # asyncio.run(run_async_examples())
