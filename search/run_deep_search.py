import os
import shutil
import asyncio
from typing import List
from jet.file.utils import save_file
from jet.search.deep_search import web_deep_search, rag_search, prepare_context, llm_generate, WebDeepSearchResult, RagSearchResult, ContextData, LLMGenerateResult, HeaderSearchResult
from jet.utils.text import format_sub_dir

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

async def example_usage(query: str, use_cache: bool = True):
    """
    Example usage of web_deep_search, rag_search, prepare_context, and llm_generate functions.
    Demonstrates searching for information on recent AI advancements.
    """
    sub_dir = f"{OUTPUT_DIR}/{format_sub_dir(query)}"

    # Example 1: Using web_deep_search
    print("Running web_deep_search...")
    deep_search_result: WebDeepSearchResult = await web_deep_search(
        query=query,
        embed_model="all-MiniLM-L6-v2",
        llm_model="llama-3.2-3b-instruct-4bit",
        use_cache=use_cache,
        urls_limit=10
    )

    # Print key results
    print(f"\nQuery: {deep_search_result['query']}")
    print(f"Number of search results: {len(deep_search_result['search_results'])}")
    print(f"High score URLs: {len(deep_search_result['high_score_urls'])}")
    print(f"Response text: {deep_search_result['response_text'][:200]}...")  # Truncated for brevity
    print(f"Total tokens: {deep_search_result['token_info']['total_tokens']}")

    save_file(deep_search_result, f"{sub_dir}/deep_search_result.json")

    # Example 2: Using rag_search with sample URLs
    print("\nRunning rag_search with sample URLs...")
    sample_urls = [
        "https://www.crescendo.ai/news/latest-ai-news-and-updates",
    ]
    rag_search_result: RagSearchResult = await rag_search(
        query=query,
        embed_model="all-MiniLM-L6-v2",
        top_k=None,
        threshold=0.0,
        chunk_size=200,
        chunk_overlap=50,
        merge_chunks=False,
        urls=sample_urls,
        use_cache=use_cache,
        urls_limit=10
    )

    # Print URL processing results
    print(f"Number of processed URLs: {len(rag_search_result['all_completed_urls'])}")
    print(f"Number of search results: {len(rag_search_result['search_results'])}")
    print(f"Total tokens: {rag_search_result['headers_total_tokens']}")
    print(f"High score tokens: {rag_search_result['headers_high_score_tokens']}")
    print(f"Medium score tokens: {rag_search_result['headers_medium_score_tokens']}")
    print(f"MTLD score average: {rag_search_result['headers_mtld_score_average']}")
    print(f"Number of search engine results: {len(rag_search_result['search_engine_results'])}")

    # Example of processing high-score URLs
    for url in rag_search_result['all_urls_with_high_scores'][:2]:  # Show top 2 high-score URLs
        print(f"\nHigh-score URL: {url}")

    save_file(rag_search_result, f"{sub_dir}/rag_search_result.json")

    # Example 3: Using rag_search without URLs to fetch search results
    print("\nRunning rag_search without URLs...")
    rag_search_no_urls_result: RagSearchResult = await rag_search(
        query=query,
        embed_model="all-MiniLM-L6-v2",
        top_k=None,
        threshold=0.0,
        chunk_size=200,
        chunk_overlap=50,
        merge_chunks=False,
        urls=None,
        use_cache=use_cache,
        urls_limit=5
    )

    # Print results for no-URLs case
    print(f"Number of processed URLs: {len(rag_search_no_urls_result['all_completed_urls'])}")
    print(f"Number of search engine results: {len(rag_search_no_urls_result['search_engine_results'])}")

    save_file(rag_search_no_urls_result, f"{sub_dir}/rag_search_no_urls_result.json")

    # Example 4: Using prepare_context and llm_generate with existing search results
    print("\nRunning prepare_context and llm_generate with existing search results...")
    context_data: ContextData = prepare_context(
        query=query,
        search_results=rag_search_result['search_results'],  # Fixed: Changed url_result to rag_search_result
        llm_model="llama-3.2-3b-instruct-4bit",
        max_tokens=1000  # Smaller max_tokens for demonstration
    )

    save_file(context_data, f"{sub_dir}/context_data.json")

    llm_result: LLMGenerateResult = await llm_generate(
        query=query,
        context_data=context_data,
        llm_model="llama-3.2-3b-instruct-4bit"
    )

    # Print context preparation and LLM generation results
    print(f"Number of sorted search results: {len(context_data['sorted_search_results'])}")
    print(f"Number of filtered results: {len(context_data['filtered_results'])}")
    print(f"Filtered URLs: {len(context_data['filtered_urls'])}")
    print(f"Context length: {len(context_data['context'])} characters")
    print(f"LLM response text: {llm_result['response_text'][:200]}...")  # Truncated for brevity
    print(f"LLM total tokens: {llm_result['token_info']['total_tokens']}")

    save_file(llm_result, f"{sub_dir}/llm_result.json")

if __name__ == "__main__":
    query = "Latest advancements in AI research 2025"
    use_cache = False
    asyncio.run(example_usage(query, use_cache))
