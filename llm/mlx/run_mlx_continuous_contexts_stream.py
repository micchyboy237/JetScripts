import asyncio
import json
import os
import shutil
import re
from jet.code.splitter_markdown_utils import get_md_header_docs
from jet.llm.mlx.base import MLX
from jet.llm.mlx.helpers.base import rewrite_query
from jet.llm.mlx.mlx_types import LLMModelType
from jet.file.utils import load_file, save_file
from jet.llm.utils.search_docs import search_docs
from jet.scrapers.hrequests_utils import scrape_urls, sync_scrape_url
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

            context += f"{header}\n"
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
    results = []

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
                "content": "You are a precise data extraction tool. Given a list of documents, extract each anime title from the 'header' field of each document. Return each title as a separate JSON object in JSONL format, with each line containing a single JSON object (e.g., {\"title\": \"Anime1\"}\\n{\"title\": \"Anime2\"}). Ensure each line is valid JSON, contains only the 'title' key, and is terminated with a newline. Do not return an array, additional text, or explanations."
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
            repetition_penalty=1.0,
            max_tokens=-1
        ):
            content = stream_response["choices"][0]["message"]["content"]
            current_line += content

            if "\n" in current_line:
                lines = current_line.split("\n")
                for line in lines[:-1]:  # Process all complete lines
                    line = line.strip()
                    if line:
                        try:
                            json.loads(line)  # Validate JSON
                            save_file(line + "\n", anime_titles_file,
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
                save_file(current_line + "\n", anime_titles_file,
                          verbose=False, append=True)
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line: {current_line}")

        # Load titles and save conversation as JSONL
        loaded_titles = []
        if os.path.exists(anime_titles_file):
            with open(anime_titles_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            loaded_titles.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Invalid JSONL line in {anime_titles_file}: {line}")
        messages.append(
            {"role": "assistant", "content": "\n".join(
                item["title"] for item in loaded_titles)}
        )
        # Save messages as JSONL
        with open(f"{sub_output_dir}/stream_chat_anime_titles.jsonl", 'a') as f:
            for message in messages:
                f.write(json.dumps(message) + "\n")

        # # Search all anime titles if available in aniwatch for watching episodes
        # query_aniwatch_search_links = f"Aniwatch anime search links"
        # # browser_aniwatch_search_links_results, html_list = await fetch_search_results(query_aniwatch_search_links)
        # browser_aniwatch_search_links_results = search_data(
        #     query_aniwatch_search_links, use_cache=False)
        # save_file({
        #     "query": query_aniwatch_search_links,
        #     "count": len(browser_aniwatch_search_links_results),
        #     "results": browser_aniwatch_search_links_results
        # }, f"{sub_output_dir}/browser_aniwatch_search_links_results.json")

        # documents = [
        #     search_result['url']
        #     for search_result in browser_aniwatch_search_links_results
        #     if search_result.get("title")
        # ]
        # save_file({
        #     "query": query_aniwatch_search_links,
        #     "results": documents
        # }, f"{sub_output_dir}/context_aniwatch_urls.json")

        # context_aniwatch_search_urls = "\n\n".join(documents)
        # save_file(context_aniwatch_search_urls,
        #           f"{sub_output_dir}/context_aniwatch_urls.md")

        # messages = [
        #     {
        #         "role": "system",
        #         "content": "You are a precise URL extraction tool. Given search results, identify and return only the URL that best provides a link to watch episodes of the specified anime on AniWatch. Return only the URL as plain text, with no additional text, explanations, or formatting. If no matching AniWatch URL is found, return 'None'."
        #     },
        #     {
        #         "role": "user",
        #         "content": f"Generate the AniWatch search URL for watching episodes of \"{loaded_titles[0]['title']}\" derived from any of following urls that provides a query to replace with.\nMake sure to update a search parameter with the anime title.\n{context_aniwatch_search_urls}"
        #     },
        # ]

        # response = ""
        # for chunk in mlx.stream_chat(
        #     messages=messages,
        #     model=llm_model,
        #     temperature=0.3,
        #     log_dir=MLX_LOG_DIR,
        #     verbose=True,
        #     logit_bias=["Link:"],
        #     repetition_penalty=1.2
        # ):
        #     content = chunk["choices"][0]["message"]["content"]
        #     response += content

        # messages.append(
        #     {"role": "assistant", "content": response}
        # )

        # save_file(messages, f"{output_dir}/stream_chat_aniwatch_url.json")

        search_link_template = "https://aniwatchtv.to/search?keyword={anime_title}"
        for anime_title in loaded_titles:
            title = anime_title["title"].lower()
            search_link = search_link_template.format(anime_title=title)
            html_str = sync_scrape_url(search_link)
            if not html_str:
                continue
            docs = get_md_header_docs(html_str)
            header_texts = [doc["header"].lower() for doc in docs]
            search_results = search_docs(
                title, header_texts, threshold=0.8)
            if search_results:
                output = {
                    "title": title,
                    "link": search_link,
                    "results": search_results
                }
                results.append(output)
                save_file(results, f"{output_dir}/existing_animes.json")


if __name__ == "__main__":
    asyncio.run(main())
