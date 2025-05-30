import asyncio
import json
import os
import shutil
import re
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
from collections import defaultdict
from typing import Sequence, Tuple, Union, Dict, Any, TypedDict, List
from pathlib import Path
from pydantic import BaseModel


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


async def fetch_search_results(query: str) -> Tuple[List[SearchResult], List[str]]:
    """Fetch search results and save them."""
    browser_search_results = search_data(query, use_cache=False)
    urls = [item["url"] for item in browser_search_results]
    html_list = await scrape_urls(urls, num_parallel=5)
    return browser_search_results, html_list


def format_sub_url_dir(url: str) -> str:
    """Format a URL into a lowercase directory name, replacing hyphens and spaces with underscores."""
    clean_url = re.sub(r'^(https?://|www\.)|(\?.*)', '', url)
    formatted = re.sub(r'[- ]+', '_', clean_url).lower()
    formatted = re.sub(r'[^\w./]', '_', formatted)
    formatted = re.sub(r'_+', '_', formatted)
    return formatted.strip('_')


async def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), "generated", os.path.splitext(os.path.basename(__file__))[0])

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    MLX_LOG_DIR = f"{output_dir}/mlx-logs"
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"

    llm_model: LLMModelType = "llama-3.2-3b-instruct-4bit"
    seed = 42

    # Load JSON data
    docs: list[dict] = load_file(docs_file)
    print(f"Loaded JSON data {len(docs)} from: {docs_file}")
    grouped_docs_by_source_url = group_by(docs, "['metadata']['source_url']")

    for group_source_url_docs in grouped_docs_by_source_url:
        source_url = group_source_url_docs['group']
        sub_url_dir = format_sub_url_dir(source_url)
        sub_output_dir = os.path.join(output_dir, sub_url_dir)
        os.makedirs(sub_output_dir, exist_ok=True)

        headers = [doc["metadata"]["header"]
                   for doc in group_source_url_docs['items']]
        context = "\n\n".join(headers)
        save_file(context, f"{sub_output_dir}/context_anime_titles.md")

        """Example of using the .stream_chat method for streaming chat completions."""
        messages = [
            {
                "role": "system",
                "content": "You are a precise data extraction tool. Given a list of documents, extract each anime title from the 'header' field of each document. Return each title as a separate JSON object on a new line in JSONL format (e.g., {\"title\": \"Anime1\"}\\n{\"title\": \"Anime2\"}). Do not return an array or additional text, only valid JSONL lines."
            },
            {
                "role": "user",
                "content": f"Extract anime titles from the following context:\n\n{context}"
            },
        ]

        logger.debug("Streaming Chat Response:")
        anime_titles_file = f"{sub_output_dir}/anime_titles.jsonl"

        mlx = MLX(seed=seed)
        current_line = ""

        # Accumulate chunks until newline, then save as JSONL line
        for stream_response in mlx.stream_chat(
            messages=messages,
            model=llm_model,
            temperature=0.3,
            log_dir=MLX_LOG_DIR,
            verbose=True,
            logit_bias=["{", "}"],
            max_tokens=-1
        ):
            content = stream_response["choices"][0]["message"]["content"]
            current_line += content

            # Check for newline to save complete JSONL line
            if "\n" in current_line:
                lines = current_line.split("\n")
                for line in lines[:-1]:  # Process all complete lines
                    line = line.strip()
                    if line:
                        try:
                            json.loads(line)  # Validate JSON
                            save_file(line, anime_titles_file,
                                      verbose=False, append=True)
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Skipping invalid JSON line: {line}")
                            continue
                current_line = lines[-1]  # Keep the last incomplete line

        # Save any remaining content if valid
        if current_line.strip():
            try:
                json.loads(current_line)
                save_file(current_line, anime_titles_file,
                          verbose=False, append=True)
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line: {current_line}")
                pass

        # Append the full conversation to stream_chat_anime_titles.jsonl
        loaded_titles = load_file(anime_titles_file) or []
        messages.append(
            {"role": "assistant", "content": "\n".join(
                json.dumps(item) for item in loaded_titles)}
        )
        save_file(
            messages, f"{sub_output_dir}/stream_chat_anime_titles.jsonl", append=True)

        # Search all anime titles if available in aniwatch for watching episodes
        query_aniwatch_search_links = f"Aniwatch anime search links"
        # query = rewrite_query(query, llm_model)
        browser_aniwatch_search_links_results, html_list = await fetch_search_results(query_aniwatch_search_links)
        save_file({
            "query": query_aniwatch_search_links,
            "count": len(browser_aniwatch_search_links_results),
            "results": browser_aniwatch_search_links_results
        }, f"{sub_output_dir}/browser_aniwatch_search_links_results.json")

        documents = [
            search_result['url']
            for search_result in browser_aniwatch_search_links_results
            if search_result.get("title")
        ]
        save_file({
            "query": query_aniwatch_search_links,
            "results": documents
        }, f"{sub_output_dir}/context_aniwatch_urls.json")

        context_aniwatch_search_urls = "\n\n".join(documents)
        save_file(context_aniwatch_search_urls,
                  f"{sub_output_dir}/context_aniwatch_urls.md")

        messages = [
            {
                "role": "system",
                "content": "You are a precise URL extraction tool. Given search results, identify and return only the URL that best provides a link to watch episodes of the specified anime on AniWatch. Return only the URL as plain text, with no additional text, explanations, or formatting. If no matching AniWatch URL is found, return 'None'."
            },
            {
                "role": "user",
                "content": f"Generate the AniWatch search URL for watching episodes of \"{loaded_titles[0]}\" derived from the following urls:\n\n{context_aniwatch_search_urls}"
            },
        ]

        response = ""
        for chunk in mlx.stream_chat(
            messages=messages,
            model=llm_model,
            temperature=0.3,
            log_dir=MLX_LOG_DIR,
            verbose=True,
            logit_bias=["Link:", "None"],
            max_tokens=-1
        ):
            content = chunk["choices"][0]["message"]["content"]
            response += content

        messages.append(
            {"role": "assistant", "content": response}
        )

        save_file(messages, f"{sub_output_dir}/stream_chat_aniwatch_url.json")

if __name__ == "__main__":
    asyncio.run(main())
