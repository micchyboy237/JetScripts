import os
import shutil
import asyncio
from typing import List
from jet.file.utils import save_file
from jet.search.deep_search import aweb_deep_search, rag_search, WebDeepSearchResult, RagSearchResult, HeaderSearchResult

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

async def example_usage():
    query = "Latest advancements in AI research 2025"

    print("\nRunning rag_search with existing search results...")
    rag_result: RagSearchResult = await rag_search(
        query=query,
        search_results=deep_search_result['search_results'],
        llm_model="llama-3.2-3b-instruct-4bit",
        max_tokens=1000  # Smaller max_tokens for demonstration
    )

    # Print RAG-specific results
    print(f"Number of filtered results: {len(rag_result['filtered_results'])}")
    print(f"Filtered URLs: {len(rag_result['filtered_urls'])}")
    print(f"Context length: {len(rag_result['context'])} characters")
    print(f"RAG response text: {rag_result['response_text'][:200]}...")  # Truncated for brevity

    # Example of processing results
    for url_info in deep_search_result['high_score_urls'][:2]:  # Show top 2 high-score URLs
        print(f"\nHigh-score URL: {url_info['url']}")
        print(f"High-score tokens: {url_info['high_score_tokens']}")
        print(f"Medium-score tokens: {url_info['medium_score_tokens']}")

    save_file(rag_result, f"{OUTPUT_DIR}/rag_result.json")

if __name__ == "__main__":
    asyncio.run(example_usage())