import asyncio
import os
import re
import shutil
import string
import numpy as np

from collections import defaultdict
from typing import DefaultDict, List, Set
from jet.code.html_utils import preprocess_html
from jet.code.markdown_types.markdown_parsed_types import HeaderDoc
from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.code.markdown_utils._markdown_analyzer import analyze_markdown
from jet.code.markdown_utils._markdown_parser import base_parse_markdown, derive_by_header_hierarchy, parse_markdown
from jet.code.markdown_utils._preprocessors import clean_markdown_links, link_to_text_ratio
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.logger.config import colorize_log
from jet.models.embeddings.base import generate_embeddings
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.models.tokenizer.base import get_tokenizer_fn, count_tokens
from jet.scrapers.hrequests_utils import scrape_urls
from jet.scrapers.utils import scrape_links, search_data
from jet.vectors.filters import select_diverse_texts
from jet.vectors.filters.mmr import select_mmr_texts
from jet.vectors.filters.select_diverse_texts import DiverseResult
from jet.vectors.semantic_search.header_vector_search import HeaderSearchResult, search_headers
from jet.wordnet.analyzers.text_analysis import analyze_readability
from jet.wordnet.similarity import group_similar_texts


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

PROMPT_TEMPLATE = """\
Context information is below.
---------------------
{context}
---------------------

Given the context information, answer the query.

Query: {query}
"""

HIGH_QUALITY_SCORE = 0.6
MEDIUM_QUALITY_SCORE = 0.4
TARGET_HIGH_SCORE_TOKENS = 4000
TARGET_MEDIUM_SCORE_TOKENS = 10000


def format_sub_dir(text: str) -> str:
    return text.lower().strip('.,!?').replace(' ', '_').replace('.', '_').replace(',', '_').replace('!', '_').replace('?', '_').strip()


def format_sub_source_dir(source: str) -> str:
    """Format a source (URL or file path) into a directory name."""
    clean_source = re.sub(r'^(https?://|www\.)|(\?.*)', '', source)
    clean_source = clean_source.replace(os.sep, '_')
    trans_table = str.maketrans({p: '_' for p in string.punctuation})
    formatted = clean_source.translate(trans_table).lower()
    formatted = re.sub(r'_+', '_', formatted)
    return formatted.strip('_')


def sort_search_results_by_url_and_category(results: List[HeaderSearchResult], sorted_urls: List[str]):
    """
    Sorts results in three stages:
    1. Results with score >= HIGH_QUALITY_SCORE, sorted by url order in sorted_urls, then by score descending within each url.
    2. Results with score >= MEDIUM_QUALITY_SCORE but < HIGH_QUALITY_SCORE, sorted by score descending.
    3. Results with score < MEDIUM_QUALITY_SCORE, sorted by score descending.
    Returns the concatenated list.
    """
    # Stage 1: Get results with score >= HIGH_QUALITY_SCORE, grouped by url order
    url_to_results = {url: [] for url in sorted_urls}
    high_score_results = []
    medium_score_results = []
    low_score_results = []
    for r in results:
        url = r["metadata"]["source"]
        if r["score"] >= HIGH_QUALITY_SCORE and url in url_to_results:
            url_to_results[url].append(r)
        elif r["score"] >= HIGH_QUALITY_SCORE:
            # If url not in sorted_urls, treat as extra at end
            high_score_results.append(r)
        elif r["score"] >= MEDIUM_QUALITY_SCORE:
            medium_score_results.append(r)
        else:
            low_score_results.append(r)

    # Collect high score results in url order, sorting by score within each url
    sorted_high_score = []
    for url in sorted_urls:
        url_group = url_to_results[url]
        url_group_sorted = sorted(
            url_group, key=lambda r: r["score"], reverse=True)
        sorted_high_score.extend(url_group_sorted)
    # Add any >= HIGH_QUALITY_SCORE results whose url wasn't in sorted_urls, sorted by score
    if high_score_results:
        sorted_high_score.extend(
            sorted(high_score_results, key=lambda r: r["score"], reverse=True))

    # Stage 2: Medium score results (MEDIUM_QUALITY_SCORE <= score < HIGH_QUALITY_SCORE), sort by score descending
    sorted_medium_score = sorted(
        medium_score_results, key=lambda r: r["score"], reverse=True)

    # Stage 3: Low score results (score < MEDIUM_QUALITY_SCORE), sort by score descending
    sorted_low_score = sorted(
        low_score_results, key=lambda r: r["score"], reverse=True)

    return sorted_high_score + sorted_medium_score + sorted_low_score


def group_results_by_source_for_llm_context(
    results: List[HeaderSearchResult]
) -> str:
    def strip_hashtags(text: str) -> str:
        if text:
            return text.lstrip('#').strip()
        return text

    # Initialize tokenizer
    tokenizer = get_tokenizer_fn(
        "qwen3-1.7b-4bit", add_special_tokens=False, remove_pad_tokens=True)
    separator = "\n\n"
    separator_tokens = len(tokenizer.encode(separator))

    # Calculate high and medium score tokens per URL
    url_score_tokens = defaultdict(
        lambda: {"high_score_tokens": 0, "medium_score_tokens": 0}
    )
    for result in results:
        url = result["metadata"].get("source", "Unknown")
        if result["score"] >= HIGH_QUALITY_SCORE:
            url_score_tokens[url]["high_score_tokens"] += result["metadata"].get(
                "num_tokens", 0)
        elif result["score"] >= MEDIUM_QUALITY_SCORE:
            url_score_tokens[url]["medium_score_tokens"] += result["metadata"].get(
                "num_tokens", 0)

    # Sort URLs by high_score_tokens, then medium_score_tokens (descending)
    sorted_urls = sorted(
        url_score_tokens.keys(),
        key=lambda url: (url_score_tokens[url]["high_score_tokens"],
                         url_score_tokens[url]["medium_score_tokens"]),
        reverse=True
    )

    # Group results by URL and sort within each URL by score
    grouped_temp: DefaultDict[str,
                              List[HeaderSearchResult]] = defaultdict(list)
    seen_header_text: DefaultDict[str, Set[str]] = defaultdict(set)
    for result in results:
        url = result["metadata"].get("source", "Unknown")
        grouped_temp[url].append(result)

    context_blocks = []
    for url in sorted_urls:
        docs = sorted(grouped_temp[url],
                      key=lambda x: x["score"], reverse=True)
        block = f"<!-- Source: {url} -->\n\n"
        seen_header_text_in_block = set()

        # Group by doc_index and header to handle overlaps
        grouped_by_header: DefaultDict[tuple[int, str],
                                       List[HeaderSearchResult]] = defaultdict(list)
        for doc in sorted(docs, key=lambda x: (x["metadata"].get("doc_index", 0), x["metadata"].get("start_idx", 0))):
            doc_index = doc["metadata"].get("doc_index", 0)
            header = doc.get("header", "") or ""
            grouped_by_header[(doc_index, header)].append(doc)

        for (doc_index, header), chunks in grouped_by_header.items():
            parent_header = chunks[0].get("parent_header", "None")
            parent_level = chunks[0]["metadata"].get("parent_level", None)
            doc_level = chunks[0]["metadata"].get(
                "level", 0) if chunks[0]["metadata"].get("level") is not None else 0
            parent_header_key = strip_hashtags(
                parent_header) if parent_header and parent_header != "None" else None
            header_key = strip_hashtags(header) if header else None

            # Check for matching child headers to avoid redundant parent headers
            has_matching_child = any(
                strip_hashtags(d.get("header", "")) == parent_header_key
                for d in docs
                if d.get("header") and d["metadata"].get("level", 0) >= 0
            )

            # Add parent header if it hasn't been added and has no matching child
            if parent_header_key and parent_level is not None and not has_matching_child and parent_header_key not in seen_header_text_in_block:
                block += f"{parent_header}\n\n"
                seen_header_text_in_block.add(parent_header_key)

            # Add header if it hasn't been added
            if header_key and header_key not in seen_header_text_in_block and doc_level >= 0:
                block += f"{header}\n\n"
                seen_header_text_in_block.add(header_key)
                seen_header_text[url].add(header_key)

            # Sort chunks by start_idx and merge overlapping or adjacent chunks
            chunks.sort(key=lambda x: x["metadata"]["start_idx"])
            merged_content = ""
            start_idx = chunks[0]["metadata"]["start_idx"]
            end_idx = chunks[0]["metadata"]["end_idx"]
            current_content = chunks[0]["content"]
            merged_content = current_content

            for next_chunk in chunks[1:]:
                next_start = next_chunk["metadata"]["start_idx"]
                next_end = next_chunk["metadata"]["end_idx"]
                next_content = next_chunk["content"]
                if not isinstance(next_content, str):
                    logger.debug(
                        f"Non-string content in chunk for source: {url}, doc_index: {doc_index}, type: {type(next_content)}. Converting to string.")
                    next_content = str(next_content) if next_content else ""

                # Merge if chunks overlap or are adjacent
                if next_start <= end_idx + 1:
                    overlap = end_idx - next_start + 1 if next_start <= end_idx else 0
                    additional_content = next_content[overlap:
                                                      ] if overlap > 0 else next_content
                    merged_content += additional_content
                    end_idx = max(end_idx, next_end)
                else:
                    # Append merged content to block
                    block += merged_content + "\n\n"
                    # Start new merged chunk
                    merged_content = next_content
                    start_idx = next_start
                    end_idx = next_end

            # Append the last merged chunk
            block += merged_content + "\n\n"

        block_tokens = len(tokenizer.encode(block))
        if block_tokens > len(tokenizer.encode(f"<!-- Source: {url} -->\n\n")):
            context_blocks.append(block.strip())
        else:
            logger.warning(
                f"Empty block for {url} after processing; skipping.")

    result = "\n\n".join(context_blocks)
    final_token_count = len(tokenizer.encode(result))
    logger.debug(
        f"Grouped context created with {final_token_count} tokens for {len(grouped_temp)} sources")
    return result


# Helper function to create list of dicts for URLs
def create_url_dict_list(urls: List[str], search_results: List[HeaderSearchResult]) -> List[dict]:
    # Calculate stats per URL from search_results
    url_stats = defaultdict(
        lambda: {
            "high_score_tokens": 0,
            "high_score_headers": 0,
            "medium_score_tokens": 0,
            "medium_score_headers": 0,
            "headers": 0,
            "max_score": float('-inf'),
            "min_score": float('inf')
        })
    for result in search_results:
        url = result["metadata"].get("source", "Unknown")
        score = result["score"]
        url_stats[url]["headers"] += 1
        url_stats[url]["max_score"] = max(url_stats[url]["max_score"], score)
        url_stats[url]["min_score"] = min(url_stats[url]["min_score"], score)
        if result["score"] >= HIGH_QUALITY_SCORE:
            url_stats[url]["high_score_tokens"] += result["metadata"].get(
                "num_tokens", 0)
            url_stats[url]["high_score_headers"] += 1
        elif result["score"] >= MEDIUM_QUALITY_SCORE:
            url_stats[url]["medium_score_tokens"] += result["metadata"].get(
                "num_tokens", 0)
            url_stats[url]["medium_score_headers"] += 1

    return [
        {
            "url": url,
            "max_score": url_stats[url]["max_score"],
            "min_score": url_stats[url]["min_score"],
            "high_score_tokens": url_stats[url]["high_score_tokens"],
            "high_score_headers": url_stats[url]["high_score_headers"],
            "medium_score_tokens": url_stats[url]["medium_score_tokens"],
            "medium_score_headers": url_stats[url]["medium_score_headers"],
            "headers": url_stats[url]["headers"],
        }
        for url in urls
    ]


async def main(query):
    """Main function to demonstrate file search."""
    embed_model: EmbedModelType = "all-MiniLM-L6-v2"
    llm_model: LLMModelType = "qwen3-1.7b-4bit"
    max_tokens = 2000
    use_cache = True
    urls_limit = 10

    top_k = None
    threshold = 0.0
    chunk_size = 200
    chunk_overlap = 50
    merge_chunks = False

    query_output_dir = f"{OUTPUT_DIR}/{format_sub_dir(query)}"
    shutil.rmtree(query_output_dir, ignore_errors=True)

    save_file(query, f"{query_output_dir}/query.md")

    search_engine_results = search_data(query, use_cache=use_cache)
    save_file(search_engine_results,
              f"{query_output_dir}/search_engine_results.json")

    urls = [r["url"] for r in search_engine_results][:urls_limit]

    html_list = []
    header_docs: List[HeaderDoc] = []
    search_results: List[HeaderSearchResult] = []

    headers_total_tokens = 0
    headers_high_score_tokens = 0
    headers_medium_score_tokens = 0  # New variable to track medium-score tokens
    headers_mtld_score_average = 0

    all_started_urls = []
    all_completed_urls = []
    all_searched_urls = []
    all_urls_with_high_scores = []
    all_urls_with_low_scores = []

    async for url, status, html in scrape_urls(urls, show_progress=True):
        if status == "started":
            all_started_urls.append(url)
        elif status == "completed" and html:
            all_completed_urls.append(url)
            html_list.append(html)

            sub_source_dir = format_sub_source_dir(url)
            sub_output_dir = os.path.join(
                query_output_dir, "pages", sub_source_dir)

            save_file(html, f"{sub_output_dir}/page.html")
            save_file(preprocess_html(html),
                      f"{sub_output_dir}/page_preprocessed.html")

            links = set(scrape_links(html, url))
            links = [link for link in links if (
                link != url if isinstance(link, str) else link["url"] != url)]
            save_file(links, os.path.join(
                sub_output_dir, "links.json"))

            doc_markdown = convert_html_to_markdown(html, ignore_links=False)
            save_file(doc_markdown, f"{sub_output_dir}/page.md")

            doc_analysis = analyze_markdown(doc_markdown)
            save_file(doc_analysis, f"{sub_output_dir}/analysis.json")
            doc_markdown_tokens = base_parse_markdown(doc_markdown)
            save_file(doc_markdown_tokens,
                      f"{sub_output_dir}/markdown_tokens.json")

            original_docs: List[HeaderDoc] = derive_by_header_hierarchy(
                doc_markdown)

            save_file(original_docs, f"{sub_output_dir}/docs.json")

            for doc in original_docs:
                doc["source"] = url

            sub_results = list(
                search_headers(
                    original_docs,
                    query,
                    top_k=top_k,
                    threshold=threshold,
                    embed_model=embed_model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    tokenizer_model=embed_model,
                    merge_chunks=merge_chunks
                )
            )
            all_searched_urls.append(url)

            # Add ltr_ratio using link_to_text_ratio on each result by content
            filtered_sub_results = []
            for result in sub_results:
                ltr = link_to_text_ratio(result["content"])
                result["metadata"]["ltr_ratio"] = ltr
                mtld_result = analyze_readability(result["content"])
                result["metadata"]["mtld"] = mtld_result["mtld"]
                result["metadata"]["mtld_category"] = mtld_result["mtld_category"]
                if (
                    result["score"] >= MEDIUM_QUALITY_SCORE
                    and result["metadata"]["mtld_category"] != "very_low"
                ):
                    filtered_sub_results.append(result)

            sub_total_tokens = sum(
                result["metadata"]["num_tokens"] for result in filtered_sub_results)

            sub_high_score_tokens = sum(
                result["metadata"]["num_tokens"]
                for result in filtered_sub_results
                if (
                    result["score"] >= HIGH_QUALITY_SCORE
                )
            )

            sub_medium_score_tokens = sum(  # New calculation for medium-score tokens
                result["metadata"]["num_tokens"]
                for result in filtered_sub_results
                if (
                    result["score"] >= MEDIUM_QUALITY_SCORE
                    and result["score"] < HIGH_QUALITY_SCORE
                )
            )

            sub_mtld_score_values = [
                analyze_readability(result["content"])["mtld"]
                for result in filtered_sub_results
                if (
                    result["score"] >= HIGH_QUALITY_SCORE
                    and analyze_readability(result["content"])["mtld_category"]
                )
            ]
            sub_mtld_score_average = (
                sum(sub_mtld_score_values) /
                len(sub_mtld_score_values)
                if sub_mtld_score_values else 0
            )

            save_file({
                "query": query,
                "url": url,
                "count": len(filtered_sub_results),
                "max_score": max((result["score"] for result in filtered_sub_results), default=0.0),
                "min_score": min((result["score"] for result in filtered_sub_results), default=0.0),
                "mtld": analyze_readability(html)["mtld"],
                "mtld_category": analyze_readability(html)["mtld_category"],
                "total_tokens": sub_total_tokens,
                "high_score_tokens": sub_high_score_tokens,
                # Add medium_score_tokens to output
                "medium_score_tokens": sub_medium_score_tokens,
                "mtld_score_average": sub_mtld_score_average,
                "results": filtered_sub_results,
            }, f"{sub_output_dir}/search_results.json")

            header_docs.extend(original_docs)
            search_results.extend(filtered_sub_results)
            if sub_high_score_tokens > 0:
                all_urls_with_high_scores.append(url)
            else:
                all_urls_with_low_scores.append(url)

            headers_total_tokens += sub_total_tokens
            headers_high_score_tokens += sub_high_score_tokens
            headers_medium_score_tokens += sub_medium_score_tokens  # Track medium-score tokens
            headers_mtld_score_average += round(
                sub_mtld_score_average, 2)

            # Stop processing if either high-score tokens reach TARGET_HIGH_SCORE_TOKENS
            # or combined high and medium-score tokens reach TARGET_MEDIUM_SCORE_TOKENS
            if headers_high_score_tokens >= TARGET_HIGH_SCORE_TOKENS or \
               (headers_high_score_tokens + headers_medium_score_tokens) >= TARGET_MEDIUM_SCORE_TOKENS:
                logger.info(
                    f"Stopping processing: {headers_high_score_tokens} high-score tokens "
                    f"and {headers_medium_score_tokens} medium-score tokens collected from source: {url}.")
                break

    # Clean up url result dirs with 0 total tokens
    for url in all_completed_urls:
        sub_source_dir = format_sub_source_dir(url)
        sub_output_dir = os.path.join(
            query_output_dir, "pages", sub_source_dir)
        sub_results_path = f"{sub_output_dir}/search_results.json"
        if os.path.exists(sub_results_path):
            sub_results_data = load_file(sub_results_path)
            if sub_results_data.get("total_tokens", 0) == 0:
                shutil.rmtree(sub_output_dir, ignore_errors=True)
                logger.info(
                    f"Removed {sub_output_dir} due to zero total tokens during final cleanup.")

    save_file({
        "expected_order": urls,
        "started_urls": all_started_urls,
        "searched_urls": all_searched_urls,
        "high_score_urls": create_url_dict_list(all_urls_with_high_scores, search_results),
    }, f"{query_output_dir}/_scraped_url_order_logs.json")

    # Sort search_results by score then update rank
    search_results = sorted(
        search_results, key=lambda x: x["score"], reverse=True)
    for i, result in enumerate(search_results, 1):
        result["rank"] = i

    save_file({
        "query": query,
        "count": len(header_docs),
        "documents": header_docs
    }, f"{query_output_dir}/docs.json")

    # Calculate high_score_tokens, medium_score_tokens, and header_count per URL
    url_stats = defaultdict(
        lambda: {"high_score_tokens": 0, "medium_score_tokens": 0, "header_count": 0})
    for result in search_results:
        url = result["metadata"].get("source", "Unknown")
        if result["score"] >= HIGH_QUALITY_SCORE:
            url_stats[url]["high_score_tokens"] += result["metadata"].get(
                "num_tokens", 0)
            url_stats[url]["header_count"] += 1
        elif result["score"] >= MEDIUM_QUALITY_SCORE:
            url_stats[url]["medium_score_tokens"] += result["metadata"].get(
                "num_tokens", 0)
            url_stats[url]["header_count"] += 1

    # Filter URLs with high_score_tokens > 0 or medium_score_tokens > 0 and format as list of dicts, sorted by high_score_tokens then medium_score_tokens
    sorted_urls = [
        {
            "url": url,
            "high_score_tokens": stats["high_score_tokens"],
            "medium_score_tokens": stats["medium_score_tokens"],
            "header_count": stats["header_count"]
        }
        for url, stats in sorted(
            url_stats.items(),
            key=lambda x: (x[1]["high_score_tokens"],
                           x[1]["medium_score_tokens"]),
            reverse=True
        )
        if stats["high_score_tokens"] > 0 or stats["medium_score_tokens"] > 0
    ]

    save_file({
        "query": query,
        "count": len(search_results),
        "max_score": max((result["score"] for result in search_results), default=0.0),
        "min_score": min((result["score"] for result in search_results), default=0.0),
        "total_tokens": headers_total_tokens,
        "high_score_tokens": headers_high_score_tokens,
        "medium_score_tokens": headers_medium_score_tokens,
        "mtld_score_average": headers_mtld_score_average,
        "urls": sorted_urls,
        "results": search_results
    }, f"{query_output_dir}/search_results.json")

    # # Cluster search results
    # # Prepare result_texts and result_ids for clustering
    # result_texts = [
    #     clean_markdown_links(
    #         f"{result['header'].lstrip('#').strip()}\n{result['content']}")
    #     # f"{result['header'].lstrip('#').strip()}\n{result['content']}"
    #     for result in search_results
    # ]
    # result_ids = [result["id"] for result in search_results]
    # grouped_similar_texts = group_similar_texts(
    #     result_texts, model_name=embed_model, ids=result_ids, threshold=0.7)
    # # Map grouped_similar_texts (which contains lists of result_ids) back to the full search_results dicts
    # id_to_result = {
    #     result["id"]: f"{result['header']}\n{result['content']}" for result in search_results}
    # clustered_search_results = [
    #     [id_to_result[result_id]
    #         # Access 'texts' field
    #         for result_id in cluster["texts"] if result_id in id_to_result]
    #     for cluster in grouped_similar_texts
    # ]
    # save_file({
    #     "query": query,
    #     "count": len(clustered_search_results),
    #     "results": clustered_search_results
    # }, f"{query_output_dir}/clustered_search_results.json")

    # # Select diverse texts on each cluster
    # documents = [
    #     f"{doc['header'].lstrip('#').strip()}\n{doc['content']}" for doc in search_results]

    # embeddings = generate_embeddings(
    #     documents, embed_model, show_progress=True)

    # doc_id_to_doc = {doc["id"]: doc for doc in search_results}
    # doc_id_to_embeddings = {doc["id"]: emb for doc,
    #                         emb in zip(search_results, embeddings)}

    # diverse_cluster_texts = []
    # diverse_cluster_results = []
    # for cluster in grouped_similar_texts:  # Iterate over ClusterResult dictionaries
    #     cluster_texts = []
    #     cluster_embeddings = []
    #     cluster_ids = []
    #     for doc_id in cluster["texts"]:  # Access 'texts' field
    #         header_doc = doc_id_to_doc[doc_id]
    #         text = f"{header_doc['header']}\n{header_doc['content']}"
    #         text_embeddings = doc_id_to_embeddings[doc_id]
    #         cluster_texts.append(text)
    #         cluster_embeddings.append(text_embeddings)
    #         cluster_ids.append(doc_id)

    #     cluster_embeddings = np.ascontiguousarray(cluster_embeddings)

    #     diverse_results: List[DiverseResult] = select_diverse_texts(
    #         cluster_embeddings,
    #         cluster_texts,
    #         diversity_threshold=0.7,
    #         max_texts=len(cluster_texts),
    #         ids=cluster_ids
    #     )
    #     diverse_cluster_texts.extend([result["text"]
    #                                   for result in diverse_results])
    #     diverse_cluster_results.extend(diverse_results)

    # token_counts: List[int] = count_tokens(
    #     embed_model, diverse_cluster_texts, prevent_total=True)
    # save_file({
    #     "count": len(diverse_cluster_texts),
    #     "total_tokens": sum(token_counts),
    #     "texts": [{
    #         "id": result["id"],
    #         "index": result["index"],
    #         "tokens": tokens,
    #         "text": result["text"],
    #         "score": result["score"]
    #     } for tokens, result in zip(token_counts, diverse_cluster_results)],
    # }, f"{query_output_dir}/diverse_cluster_results.json")

    # # Map diverse_results back to original search_results by doc_id
    # id_to_search_result = {doc["id"]: doc for doc in search_results}
    # diverse_search_results = []
    # for result in diverse_cluster_results:
    #     doc_id = result["id"]
    #     mapped_result = dict(id_to_search_result[doc_id])
    #     # Optionally, you can add the diverse score to the search result
    #     mapped_result["diverse_score"] = result.get("score", None)
    #     diverse_search_results.append(mapped_result)

    # save_file({
    #     "count": len(diverse_search_results),
    #     "total_tokens": sum(token_counts),
    #     "texts": [{
    #         **result,
    #         "tokens": tokens,
    #     } for tokens, result in zip(token_counts, diverse_search_results)],
    # }, f"{query_output_dir}/diverse_search_results.json")

    # # Group results by URL and calculate total high_score_tokens and medium_score_tokens per URL
    # url_score_tokens = defaultdict(
    #     lambda: {"high_score_tokens": 0, "medium_score_tokens": 0})
    # for result in search_results:
    #     url = result["metadata"].get("source", "Unknown")
    #     if result["score"] >= HIGH_QUALITY_SCORE:
    #         url_score_tokens[url]["high_score_tokens"] += result["metadata"].get(
    #             "num_tokens", 0)
    #     elif result["score"] >= MEDIUM_QUALITY_SCORE:
    #         url_score_tokens[url]["medium_score_tokens"] += result["metadata"].get(
    #             "num_tokens", 0)

    # # Sort URLs by total high_score_tokens (descending), with medium_score_tokens as tiebreaker
    # sorted_urls = sorted(
    #     url_score_tokens.keys(),
    #     key=lambda url: (url_score_tokens[url]["high_score_tokens"],
    #                      url_score_tokens[url]["medium_score_tokens"]),
    #     reverse=True
    # )
    # save_file(sorted_urls, f"{query_output_dir}/sorted_urls.json")

    # # Sort all results by score
    # sorted_results = sort_search_results_by_url_and_category(
    #     search_results, sorted_urls)
    # total_tokens = sum(result["metadata"].get("num_tokens", 0)
    #                    for result in sorted_results)
    # save_file({
    #     "query": query,
    #     "count": len(sorted_results),
    #     "total_tokens": total_tokens,
    #     "high_score_tokens": sum(url_score_tokens[url]["high_score_tokens"] for url in sorted_urls),
    #     "medium_score_tokens": sum(url_score_tokens[url]["medium_score_tokens"] for url in sorted_urls),
    #     "results": sorted_results
    # }, f"{query_output_dir}/sorted_search_results.json")

    # # Select diverse search results
    # result_texts = [
    #     f"{result['header'].lstrip('#').strip()}\n{result['content']}"
    #     for result in search_results
    # ]
    # doc_embeddings = generate_embeddings(
    #     result_texts, embed_model, show_progress=True)
    # all_doc_embeddings = np.ascontiguousarray(doc_embeddings)

    # doc_ids = [doc["metadata"]["doc_id"] for doc in search_results]
    # diverse_results: List[DiverseResult] = select_diverse_texts(
    #     cluster_embeddings=all_doc_embeddings,
    #     cluster_texts=result_texts,
    #     max_diverse_texts=len(result_texts),
    #     ids=doc_ids
    # )
    # diverse_texts = [result["text"] for result in diverse_results]
    # token_counts: List[int] = count_tokens(
    #     embed_model, diverse_texts, prevent_total=True)
    # save_file({
    #     "count": len(diverse_results),
    #     "total_tokens": sum(token_counts),
    #     "texts": [{
    #         "id": result["id"],
    #         "index": result["index"],
    #         "tokens": tokens,
    #         "text": result["text"],
    #         "score": result["score"]
    #     } for tokens, result in zip(token_counts, diverse_results)],
    # }, f"{query_output_dir}/diverse_search_results.json")

    # # Map diverse_results back to original search_results by doc_id
    # id_to_search_result = {result["metadata"]
    #                        ["doc_id"]: result for result in search_results}
    # diverse_search_results = []
    # for result in diverse_results:
    #     doc_id = result["id"]
    #     mapped_result = dict(id_to_search_result[doc_id])
    #     # Optionally, you can add the diverse score to the search result
    #     mapped_result["diverse_score"] = result.get("score", None)
    #     diverse_search_results.append(mapped_result)

    # Generate embeddings for texts and query
    model = SentenceTransformerRegistry.load_model(embed_model)
    search_result_texts = [
        f"{result["header"]}\n{result["content"]}" for result in search_results]
    texts = search_result_texts[:20]
    ids = [result["id"]
           for result in search_results[:20]]  # Slice ids to match texts
    embeddings = generate_embeddings(texts, model, show_progress=True)
    query_embedding = generate_embeddings(query, model)
    diverse_search_results = select_mmr_texts(
        embeddings=embeddings,
        texts=texts,
        query_embedding=query_embedding,
        lambda_param=0.5,
        max_texts=20,
        ids=ids
    )
    diverse_token_counts: List[int] = count_tokens(
        llm_model, [r["text"] for r in diverse_search_results], prevent_total=True)

    save_file({
        "count": len(diverse_search_results),
        "total_tokens": sum(diverse_token_counts),
        "texts": [{
            **result,
            "tokens": tokens,
        } for tokens, result in zip(diverse_token_counts, diverse_search_results)],
    }, f"{query_output_dir}/diverse_search_results.json")

    # Map diverse_results back to original search_results by id
    id_to_search_result = {result["id"]: result for result in search_results}

    # Filter results based on score, MTLD, and link-to-text ratio
    current_tokens = 0
    filtered_results = []
    for result in diverse_search_results:
        content = result['text']
        doc_id = result["id"]
        mapped_search_result = id_to_search_result[doc_id]

        tokens = count_tokens(llm_model, content)
        if current_tokens + tokens > max_tokens:
            break
        filtered_results.append(mapped_search_result)
        current_tokens += tokens

    # Before saving contexts.json, compute filtered_urls based on filtered_results
    filtered_url_stats = defaultdict(
        lambda: {"high_score_tokens": 0, "medium_score_tokens": 0, "header_count": 0})
    for result in filtered_results:
        url = result["metadata"]["source"]
        if result["score"] >= HIGH_QUALITY_SCORE:
            filtered_url_stats[url]["high_score_tokens"] += result["metadata"].get(
                "num_tokens", 0)
            filtered_url_stats[url]["header_count"] += 1
        elif result["score"] >= MEDIUM_QUALITY_SCORE:
            filtered_url_stats[url]["medium_score_tokens"] += result["metadata"].get(
                "num_tokens", 0)
            filtered_url_stats[url]["header_count"] += 1

    # Create filtered_urls list, sorted by high_score_tokens then medium_score_tokens
    filtered_urls = [
        {
            "url": url,
            "high_score_tokens": stats["high_score_tokens"],
            "medium_score_tokens": stats["medium_score_tokens"],
            "header_count": stats["header_count"]
        }
        for url, stats in sorted(
            filtered_url_stats.items(),
            key=lambda x: (x[1]["high_score_tokens"],
                           x[1]["medium_score_tokens"]),
            reverse=True
        )
    ]

    # Update the save_file call for contexts.json to use filtered_urls
    save_file({
        "query": query,
        "count": len(filtered_results),
        "total_tokens": current_tokens,
        "urls": filtered_urls,  # Use filtered_urls instead of sorted_urls
        "results": filtered_results
    }, f"{query_output_dir}/contexts.json")

    context = group_results_by_source_for_llm_context(filtered_results)
    save_file(context, f"{query_output_dir}/context.md")
    mlx = MLXModelRegistry.load_model(llm_model)
    prompt = PROMPT_TEMPLATE.format(query=query, context=context)
    save_file(prompt, f"{query_output_dir}/prompt.md")
    llm_response = mlx.chat(prompt, llm_model, temperature=0.7, verbose=True)
    save_file(llm_response["content"], f"{query_output_dir}/response.md")

    input_tokens = count_tokens(llm_model, prompt)
    output_tokens = count_tokens(llm_model, llm_response["content"])

    save_file({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }, f"{query_output_dir}/tokens_info.json")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Run semantic search and processing pipeline.")
    p.add_argument("query_pos", type=str, nargs="?",
                   help="Search query as positional argument")
    p.add_argument("-q", "--query", type=str,
                   help="Search query using optional flag")
    args = p.parse_args()

    query = args.query if args.query else args.query_pos or "Top isekai anime 2025."

    asyncio.run(main(query))
