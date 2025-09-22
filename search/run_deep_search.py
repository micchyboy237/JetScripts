import os
import shutil
import asyncio
from typing import List
from jet.file.utils import save_file
from jet.search.deep_search import web_deep_search, rag_search, llm_generate, prepare_context, WebDeepSearchResult, UrlProcessingResult, LLMGenerateResult, HeaderSearchResult

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

async def example_usage():
    """
    Example usage of web_deep_search, rag_search, prepare_context, and llm_generate functions.
    Demonstrates searching for information on recent AI advancements.
    """
    query = "Latest advancements in AI research 2025"

    # Example 1: Using web_deep_search
    print("Running web_deep_search...")
    deep_search_result: WebDeepSearchResult = await web_deep_search(
        query=query,
        embed_model="all-MiniLM-L6-v2",
        llm_model="llama-3.2-3b-instruct-4bit"
    )

    # Print key results
    print(f"\nQuery: {deep_search_result['query']}")
    print(f"Number of search results: {len(deep_search_result['search_results'])}")
    print(f"High score URLs: {len(deep_search_result['high_score_urls'])}")
    print(f"Response text: {deep_search_result['response_text'][:200]}...")  # Truncated for brevity
    print(f"Total tokens: {deep_search_result['token_info']['total_tokens']}")

    save_file(deep_search_result, f"{OUTPUT_DIR}/deep_search_result.json")

    # Example 2: Using rag_search directly with sample URLs
    print("\nRunning rag_search with sample URLs...")
    sample_urls = [
        "https://example.com/ai-research-2025",
        "https://example.com/tech-news"
    ]
    rag_search_result: UrlProcessingResult = await rag_search(
        urls=sample_urls,
        query=query,
        embed_model="all-MiniLM-L6-v2",
        top_k=None,
        threshold=0.0,
        chunk_size=200,
        chunk_overlap=50,
        merge_chunks=False
    )

    # Print URL processing results
    print(f"Number of processed URLs: {len(rag_search_result['all_completed_urls'])}")
    print(f"Number of search results: {len(rag_search_result['search_results'])}")
    print(f"Total tokens: {rag_search_result['headers_total_tokens']}")
    print(f"High score tokens: {rag_search_result['headers_high_score_tokens']}")
    print(f"Medium score tokens: {rag_search_result['headers_medium_score_tokens']}")
    print(f"MTLD score average: {rag_search_result['headers_mtld_score_average']}")

    # Example of processing high-score URLs
    for url in rag_search_result['all_urls_with_high_scores'][:2]:  # Show top 2 high-score URLs
        print(f"\nHigh-score URL: {url}")

    save_file(rag_search_result, f"{OUTPUT_DIR}/rag_search_result.json")

    # Example 3: Using prepare_context and llm_generate with existing search results
    print("\nRunning prepare_context and llm_generate with existing search results...")
    context_data = prepare_context(
        query=query,
        search_results=rag_search_result['search_results'],
        llm_model="llama-3.2-3b-instruct-4bit",
        max_tokens=1000  # Smaller max_tokens for demonstration
    )

    save_file(context_data, f"{OUTPUT_DIR}/context_data.json")

    llm_result: LLMGenerateResult = await llm_generate(
        query=query,
        search_results=rag_search_result['search_results'],
        llm_model="llama-3.2-3b-instruct-4bit",
        max_tokens=1000
    )

    # Print context preparation and LLM generation results
    print(f"Number of sorted search results: {len(context_data['sorted_search_results'])}")
    print(f"Number of filtered results: {len(context_data['filtered_results'])}")
    print(f"Filtered URLs: {len(context_data['filtered_urls'])}")
    print(f"Context length: {len(context_data['context'])} characters")
    print(f"LLM response text: {llm_result['response_text'][:200]}...")  # Truncated for brevity
    print(f"LLM total tokens: {llm_result['token_info']['total_tokens']}")

    save_file(llm_result, f"{OUTPUT_DIR}/llm_result.json")

if __name__ == "__main__":
    asyncio.run(example_usage())
