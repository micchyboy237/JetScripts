import asyncio
import json
import os
import shutil
from jet.llm.mlx.base import MLX
from jet.llm.mlx.helpers import rewrite_query
from jet.llm.mlx.mlx_types import LLMModelType
from jet.file.utils import load_file, save_file
from jet.llm.utils.search_docs import search_docs
from jet.scrapers.hrequests_utils import scrape_urls
from jet.scrapers.utils import search_data
from jet.search.searxng import SearchResult
from jet.transformers.formatters import format_json
from jet.utils.markdown import extract_json_block_content
from jet.wordnet.text_chunker import truncate_texts
from jet.utils.collection_utils import GroupedResult, group_by
from jet.logger import logger
import re
from collections import defaultdict
from typing import Sequence, Tuple, Union, Dict, Any, TypedDict, List


def create_hierarchical_context(grouped_docs: List[GroupedResult], max_length: int = 50) -> str:
    """Create a hierarchical markdown context string from grouped documents."""
    context = ""

    for group in grouped_docs:
        parent_header = group['group'] or ""
        context += f"{parent_header}\n".strip()

        for item in group['items']:
            header = item['metadata']['header']
            content = item['metadata']['content']

            # Create markdown header based on header_level (minimum level 3)
            context += f"{header}\n"

            # Truncate content to max_length tokens
            truncated_content = truncate_texts(content, max_length)
            context += f"{truncated_content}\n\n"

    return context


async def fetch_search_results(query: str, output_dir: str) -> Tuple[List[SearchResult], List[str]]:
    """Fetch search results and save them."""
    browser_search_results = search_data(query)
    save_file(
        {"query": query, "count": len(
            browser_search_results), "results": browser_search_results},
        os.path.join(output_dir, "browser_search_results.json")
    )
    urls = [item["url"] for item in browser_search_results]
    html_list = await scrape_urls(urls, num_parallel=5)
    return browser_search_results, html_list


async def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), "generated", os.path.splitext(os.path.basename(__file__))[0])

    shutil.rmtree(output_dir, ignore_errors=True)

    MLX_LOG_DIR = f"{output_dir}/mlx-logs"
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"

    llm_model: LLMModelType = "llama-3.2-3b-instruct-4bit"
    seed = 42

    # Get context
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    # Load JSON data
    docs: list[dict] = load_file(docs_file)
    print(f"Loaded JSON data {len(docs)} from: {docs_file}")
    grouped_docs = group_by(docs, "['metadata']['parent_header']")
    headers = [doc["metadata"]["header"]
               for doc in docs if doc["metadata"]["header_level"] != 1]

    # Create hierarchical markdown context
    context = create_hierarchical_context(grouped_docs, max_length=50)
    save_file(context, f"{output_dir}/context_anime_titles.md")

    """Example of using the .stream_chat method for streaming chat completions."""
    messages = [
        {
            "role": "system",
            "content": "You are a precise data extraction tool. Given a list of documents, extract only the anime titles from the 'header' field of each document. Return the titles as a JSON array with no additional text or explanations. Ensure the output is valid JSON surrounded by a json block."
        },
        {
            "role": "user",
            "content": f"Extract anime titles from the following context:\n\n{context}"
        },
    ]

    logger.debug("Streaming Chat Response:")
    response = ""

    mlx = MLX(seed=seed)

    for stream_response in mlx.stream_chat(
        messages=messages,
        model=llm_model,
        temperature=0.3,
        log_dir=MLX_LOG_DIR,
        verbose=True,
        logit_bias="```json"
    ):
        content = stream_response["choices"][0]["message"]["content"]
        response += content

        if stream_response["choices"][0]["finish_reason"]:
            logger.newline()

    json_result = extract_json_block_content(response)
    anime_titles = json.loads(json_result)
    save_file(anime_titles, f"{output_dir}/anime_titles.json")

    messages.append(
        {"role": "assistant", "content": response}
    )

    save_file(messages, f"{output_dir}/stream_chat_anime_titles.json")

    # Search all anime titles if available in aniwatch for watching episodes
    query = f"Find the link to this anime title ({anime_titles[0]}) to watch episodes on aniwatch."
    # query = rewrite_query(query, llm_model)
    browser_search_results, html_list = await fetch_search_results(query, output_dir)
    save_file(context, f"{output_dir}/search_results_aniwatch.json")

    documents = [
        (
            f"URL: {search_result['url']}\n"
            f"Title: {search_result['title']}\n"
            f"Content: {search_result['content']}\n"
        ).strip()
        for search_result in browser_search_results
        if search_result.get("title")
    ]
    context = f"Search results:\n\n{'n\n'.join(documents)}"
    save_file(context, f"{output_dir}/context_aniwatch_urls.md")

    messages = [
        {
            "role": "system",
            "content": "You are a precise URL extraction tool. Given search results, identify and return only the URL that best provides a link to watch episodes of the specified anime on AniWatch. Return only the URL as plain text, with no additional text, explanations, or formatting. If no matching AniWatch URL is found, return 'None'."
        },
        {
            "role": "user",
            "content": f"Extract the AniWatch URL for watching episodes of \"{anime_titles[0]}\" from the following search results:\n\n{context}"
        },
    ]

    response = ""
    for chunk in mlx.stream_chat(
        messages=messages,
        model=llm_model,
        temperature=0.3,
        log_dir=MLX_LOG_DIR,
        verbose=True,
        logit_bias=["Link:", "None"]
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content

    messages.append(
        {"role": "assistant", "content": response}
    )

    save_file(messages, f"{output_dir}/stream_chat_aniwatch_url.json")

if __name__ == "__main__":
    asyncio.run(main())
