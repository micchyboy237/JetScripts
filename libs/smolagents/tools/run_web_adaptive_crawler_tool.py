import asyncio
from crawl4ai import (
    AsyncWebCrawler,
    AdaptiveCrawler,
    AdaptiveConfig,
    LLMConfig
)

async def main():
    # === Local LLM / Embedding API Setup ===
    # Set base_url to your local OpenAI-compatible server
    local_server_base = "http://localhost:8000/v1"

    # Create an LLMConfig for BOTH embeddings and text completions
    llm_config = LLMConfig(
        provider="openai/gpt-3.5-turbo",  # we use a generic OpenAI model identifier
        api_token=None,                   # no token needed for local
        base_url=local_server_base        # points to local API
    )

    # === Adaptive Crawler Config ===
    adaptive_config = AdaptiveConfig(
        strategy="embedding",               # use semantic embedding-driven crawling
        max_pages=25,                       # limits pages to crawl
        confidence_threshold=0.75,          # stops when confidence is high enough
        n_query_variations=8,               # expand query for more coverage
        embedding_llm_config=llm_config,    # use local embedding + LLM server
        embedding_coverage_radius=0.25,
        embedding_k_exp=2.5,
    )

    # URL and query to crawl
    start_url = "https://example.com"
    query = "overview of Python 3.11 features"

    # Run the adaptive crawl
    async with AsyncWebCrawler() as crawler:
        adaptive = AdaptiveCrawler(crawler, adaptive_config)
        state = await adaptive.digest(start_url=start_url, query=query)

    # Print some results
    print(f"\n== Crawl Result Stats ==")
    print(f"Confidence Achieved: {adaptive.confidence:.1%}")
    print(f"Pages Crawled: {state.pages_crawled}")
    
    # Display top relevant pages
    top_pages = adaptive.get_relevant_content(top_k=5)
    for idx, page in enumerate(top_pages, start=1):
        print(f"\n--- Page #{idx} ---")
        print(f"URL: {page.get('url')}")
        print(f"Content snippet: {page.get('text')[:300]}")

if __name__ == "__main__":
    asyncio.run(main())
