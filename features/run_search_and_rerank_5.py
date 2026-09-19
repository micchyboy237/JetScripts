import asyncio
import json
import os
import re
import shutil
import string
import time
import uuid
from collections import defaultdict

from jet.adapters.llama_cpp.config import EMBED_MODEL_LG, LLM_MODEL
from jet.adapters.llama_cpp.llm_utils_observed import achat
from jet.adapters.llama_cpp.token_utils import count_tokens
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS, LLAMACPP_LLM_KEYS
from jet.code.html_utils import preprocess_html
from jet.code.markdown_types.markdown_parsed_types import HeaderDoc, HeaderSearchResult
from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.code.markdown_utils._markdown_analyzer import analyze_markdown
from jet.code.markdown_utils._markdown_parser import (
    base_parse_markdown,
    derive_by_header_hierarchy,
)
from jet.code.markdown_utils._preprocessors import link_to_text_ratio
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.models.utils import resolve_model_value
from jet.observability import (
    agent_span,
    embedding_span,
    get_tracer,
    hash_prompt,
    init_tracing,
    llm_span,
    redact,
    tool_span,
)
from jet.scrapers.hrequests_utils import scrape_urls
from jet.scrapers.utils import scrape_links, search_data
from jet.vectors.semantic_search.header_vector_search import search_headers
from jet.wordnet.analyzers.text_analysis import calculate_mtld, calculate_mtld_category
from openinference.semconv.trace import SpanAttributes

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "generated",
    os.path.splitext(os.path.basename(__file__))[0],
)

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

PROJECT_NAME = "search-and-rerank-pipeline"
try:
    init_tracing(project_name=PROJECT_NAME)
except Exception as e:
    logger.warning(f"Could not initialize tracing: {e}")

tracer = get_tracer(__name__)


def format_sub_dir(text: str) -> str:
    return (
        text.lower()
        .strip(".,!?")
        .replace(" ", "_")
        .replace(".", "_")
        .replace(",", "_")
        .replace("!", "_")
        .replace("?", "_")
        .strip()
    )


def format_sub_source_dir(source: str) -> str:
    clean_source = re.sub(r"^(https?://|www\.)|(\?.*)", "", source)
    clean_source = clean_source.replace(os.sep, "_")
    trans_table = str.maketrans({p: "_" for p in string.punctuation})
    formatted = clean_source.translate(trans_table).lower()
    formatted = re.sub(r"_+", "_", formatted)
    return formatted.strip("_")


def sort_urls_by_high_and_medium_score_tokens(
    results: list[HeaderSearchResult],
    medium_quality_score: float = MEDIUM_QUALITY_SCORE,
) -> list[str]:
    url_medium_score_tokens = defaultdict(int)
    for result in results:
        url = result["metadata"].get("source", "Unknown")
        if (
            result["score"] >= medium_quality_score
            and result.get("mtld_category") != "very_low"
        ):
            url_medium_score_tokens[url] += result["metadata"].get("num_tokens", 0)

    url_score_tokens = defaultdict(
        lambda: {"high_score_tokens": 0, "medium_score_tokens": 0}
    )
    sorted_urls = sorted(
        url_score_tokens.keys(),
        key=lambda url: (
            url_score_tokens[url]["high_score_tokens"],
            url_score_tokens[url]["medium_score_tokens"],
        ),
        reverse=True,
    )
    return sorted_urls


def sort_search_results_by_url_and_category(
    results: list[HeaderSearchResult],
    sorted_urls: list[str],
    high_quality_score: float = HIGH_QUALITY_SCORE,
    medium_quality_score: float = MEDIUM_QUALITY_SCORE,
):
    url_to_results = {url: [] for url in sorted_urls}
    high_score_results = []
    medium_score_results = []
    low_score_results = []
    for r in results:
        url = r["metadata"]["source"]
        if r["score"] >= high_quality_score and url in url_to_results:
            url_to_results[url].append(r)
        elif r["score"] >= high_quality_score:
            high_score_results.append(r)
        elif r["score"] >= medium_quality_score:
            medium_score_results.append(r)
        else:
            low_score_results.append(r)

    sorted_high_score = []
    for url in sorted_urls:
        url_group = url_to_results[url]
        url_group_sorted = sorted(url_group, key=lambda r: r["score"], reverse=True)
        sorted_high_score.extend(url_group_sorted)
    if high_score_results:
        sorted_high_score.extend(
            sorted(high_score_results, key=lambda r: r["score"], reverse=True)
        )

    sorted_medium_score = sorted(
        medium_score_results, key=lambda r: r["score"], reverse=True
    )
    sorted_low_score = sorted(low_score_results, key=lambda r: r["score"], reverse=True)
    return sorted_high_score + sorted_medium_score + sorted_low_score


def group_results_by_source_for_llm_context(
    results: list[HeaderSearchResult],
    high_quality_score: float = HIGH_QUALITY_SCORE,
    medium_quality_score: float = MEDIUM_QUALITY_SCORE,
) -> str:
    def strip_hashtags(text: str) -> str:
        if text:
            return text.lstrip("#").strip()
        return text

    from jet.adapters.llama_cpp.token_utils import get_tokenizer_fn

    tokenizer = get_tokenizer_fn(
        os.getenv("LLAMA_CPP_LLM_MODEL"),
        add_special_tokens=False,
    )

    url_score_tokens = defaultdict(
        lambda: {"high_score_tokens": 0, "medium_score_tokens": 0}
    )
    for result in results:
        url = result["metadata"].get("source", "Unknown")
        if result["score"] >= high_quality_score:
            url_score_tokens[url]["high_score_tokens"] += result["metadata"].get(
                "num_tokens", 0
            )
        elif result["score"] >= medium_quality_score:
            url_score_tokens[url]["medium_score_tokens"] += result["metadata"].get(
                "num_tokens", 0
            )

    sorted_urls = sorted(
        url_score_tokens.keys(),
        key=lambda url: (
            url_score_tokens[url]["high_score_tokens"],
            url_score_tokens[url]["medium_score_tokens"],
        ),
        reverse=True,
    )

    grouped_temp: defaultdict[str, list[HeaderSearchResult]] = defaultdict(list)
    seen_header_text: defaultdict[str, set[str]] = defaultdict(set)

    for result in results:
        url = result["metadata"].get("source", "Unknown")
        grouped_temp[url].append(result)

    context_blocks = []

    for url in sorted_urls:
        docs = sorted(grouped_temp[url], key=lambda x: x["score"], reverse=True)
        block = f"<!-- Source: {url} -->\n\n"
        seen_header_text_in_block = set()

        grouped_by_header: defaultdict[tuple[int, str], list[HeaderSearchResult]] = (
            defaultdict(list)
        )
        for doc in sorted(
            docs,
            key=lambda x: (
                x["metadata"].get("doc_index", 0),
                x["metadata"].get("start_idx", 0),
            ),
        ):
            doc_index = doc["metadata"].get("doc_index", 0)
            header = doc.get("header", "") or ""
            grouped_by_header[(doc_index, header)].append(doc)

        for (doc_index, header), chunks in grouped_by_header.items():
            parent_header = chunks[0].get("parent_header", "None")
            parent_level = chunks[0]["metadata"].get("parent_level", None)
            doc_level = (
                chunks[0]["metadata"].get("level", 0)
                if chunks[0]["metadata"].get("level") is not None
                else 0
            )

            parent_header_key = (
                strip_hashtags(parent_header)
                if parent_header and parent_header != "None"
                else None
            )
            header_key = strip_hashtags(header) if header else None

            has_matching_child = any(
                strip_hashtags(d.get("header", "")) == parent_header_key
                for d in docs
                if d.get("header") and d["metadata"].get("level", 0) >= 0
            )
            has_matching_child = any(
                strip_hashtags(d.get("header", "")) == parent_header_key
                for d in docs
                if d.get("header") and strip_hashtags(d.get("header", "")) != header_key
            )

            if (
                parent_header_key
                and parent_level is not None
                and has_matching_child
                and parent_header_key not in seen_header_text_in_block
            ):
                block += f"{parent_header}\n\n"
                seen_header_text_in_block.add(parent_header_key)

            if (
                header_key
                and header_key not in seen_header_text_in_block
                and doc_level >= 0
            ):
                block += f"{header}\n\n"
                seen_header_text_in_block.add(header_key)
                seen_header_text[url].add(header_key)

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
                        f"Non-string content in chunk for source: {url}, doc_index: {doc_index}, type: {type(next_content)}. Converting to string."
                    )
                    next_content = str(next_content) if next_content else ""

                if next_start <= end_idx + 1:
                    overlap = end_idx - next_start + 1 if next_start <= end_idx else 0
                    additional_content = (
                        next_content[overlap:] if overlap > 0 else next_content
                    )
                    merged_content += additional_content
                    end_idx = max(end_idx, next_end)
                else:
                    block += merged_content + "\n\n"
                    merged_content = next_content
                    start_idx = next_start
                    end_idx = next_end

            block += merged_content + "\n\n"

        block_tokens = len(tokenizer(block))
        if block_tokens > len(tokenizer(f"<!-- Source: {url} -->\n\n")):
            context_blocks.append(block.strip())
        else:
            logger.warning(f"Empty block for {url} after processing; skipping.")

    result = "\n\n".join(context_blocks)
    final_token_count = len(tokenizer(result))
    logger.debug(
        f"Grouped context created with {final_token_count} tokens for {len(grouped_temp)} sources"
    )
    return result


def create_url_dict_list(
    urls: list[str],
    search_results: list[HeaderSearchResult],
    high_quality_score: float = HIGH_QUALITY_SCORE,
    medium_quality_score: float = MEDIUM_QUALITY_SCORE,
) -> list[dict]:
    url_stats = defaultdict(
        lambda: {
            "high_score_tokens": 0,
            "high_score_headers": 0,
            "medium_score_tokens": 0,
            "medium_score_headers": 0,
            "headers": 0,
            "max_score": float("-inf"),
            "min_score": float("inf"),
        }
    )
    for result in search_results:
        url = result["metadata"].get("source", "Unknown")
        score = result["score"]
        url_stats[url]["headers"] += 1
        url_stats[url]["max_score"] = max(url_stats[url]["max_score"], score)
        url_stats[url]["min_score"] = min(url_stats[url]["min_score"], score)
        if result["score"] >= high_quality_score:
            url_stats[url]["high_score_tokens"] += result["metadata"].get(
                "num_tokens", 0
            )
            url_stats[url]["high_score_headers"] += 1
        elif result["score"] >= medium_quality_score:
            url_stats[url]["medium_score_tokens"] += result["metadata"].get(
                "num_tokens", 0
            )
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


def get_args() -> dict:
    """Parse command line arguments and return kwargs for main()."""
    import argparse

    p = argparse.ArgumentParser(
        description="Run semantic search and processing pipeline."
    )
    p.add_argument(
        "query_pos", type=str, nargs="?", help="Search query as positional argument"
    )
    p.add_argument("-q", "--query", type=str, help="Search query using optional flag")
    p.add_argument("--embed-model", type=str, default=None, help="Embedding model key")
    p.add_argument("--llm-model", type=str, default=None, help="LLM model key")
    p.add_argument(
        "--max-tokens", type=int, default=4000, help="Maximum tokens for context"
    )
    p.add_argument("--no-cache", action="store_true", help="Disable caching")
    p.add_argument("--urls-limit", type=int, default=10, help="URL processing limit")
    p.add_argument("--top-k", type=int, default=None, help="Top K results")
    p.add_argument("--threshold", type=float, default=0.0, help="Score threshold")
    p.add_argument("--chunk-size", type=int, default=200, help="Chunk size")
    p.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap")
    p.add_argument("--merge-chunks", action="store_true", help="Enable chunk merging")
    p.add_argument(
        "--high-quality-score",
        type=float,
        default=HIGH_QUALITY_SCORE,
        help="High quality threshold",
    )
    p.add_argument(
        "--medium-quality-score",
        type=float,
        default=MEDIUM_QUALITY_SCORE,
        help="Medium quality threshold",
    )
    p.add_argument(
        "--target-high-tokens",
        type=int,
        default=TARGET_HIGH_SCORE_TOKENS,
        help="Target high-score tokens",
    )
    p.add_argument(
        "--target-medium-tokens",
        type=int,
        default=TARGET_MEDIUM_SCORE_TOKENS,
        help="Target medium-score tokens",
    )

    args = p.parse_args()
    query = args.query if args.query else args.query_pos or "Top isekai anime 2026"

    kwargs = {
        "query": query,
        "use_cache": not args.no_cache,
        "urls_limit": args.urls_limit,
        "max_tokens": args.max_tokens,
        "top_k": args.top_k,
        "threshold": args.threshold,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "merge_chunks": args.merge_chunks,
        "high_quality_score": args.high_quality_score,
        "medium_quality_score": args.medium_quality_score,
        "target_high_score_tokens": args.target_high_tokens,
        "target_medium_score_tokens": args.target_medium_tokens,
    }
    if args.embed_model is not None:
        kwargs["embed_model"] = args.embed_model
    if args.llm_model is not None:
        kwargs["llm_model"] = args.llm_model
    return kwargs


async def main(
    query: str,
    embed_model: LLAMACPP_EMBED_KEYS = EMBED_MODEL_LG,
    llm_model: LLAMACPP_LLM_KEYS = LLM_MODEL,
    max_tokens: int = 4000,
    use_cache: bool = True,
    urls_limit: int = 10,
    top_k: int | None = None,
    threshold: float = 0.0,
    chunk_size: int = 200,
    chunk_overlap: int = 50,
    merge_chunks: bool = False,
    high_quality_score: float = HIGH_QUALITY_SCORE,
    medium_quality_score: float = MEDIUM_QUALITY_SCORE,
    target_high_score_tokens: int = TARGET_HIGH_SCORE_TOKENS,
    target_medium_score_tokens: int = TARGET_MEDIUM_SCORE_TOKENS,
):
    """Main function to demonstrate file search with full observability."""
    session_id = str(uuid.uuid4())
    start_time = time.time()

    with agent_span(
        name="search_pipeline.run",
        session_id=session_id,
        prompt_template_version="v1.0",
        system_prompt_hash=hash_prompt(PROMPT_TEMPLATE),
        max_steps=urls_limit,
    ) as root_span:
        root_span.set_attribute(SpanAttributes.INPUT_VALUE, redact(query))

        query_output_dir = f"{OUTPUT_DIR}/{format_sub_dir(query)}"
        shutil.rmtree(query_output_dir, ignore_errors=True)
        save_file(query, f"{query_output_dir}/query.md")

        # Stage 1: Search Engine Lookup
        with tool_span(
            name="tool.search_engine",
            tool_name="search_data",
            parameters={"query": query, "use_cache": use_cache},
        ) as span:
            search_engine_results = search_data(query, use_cache=use_cache)
            span.set_attribute("tool.output_count", len(search_engine_results))

        save_file(
            search_engine_results, f"{query_output_dir}/search_engine_results.json"
        )
        urls = [r["url"] for r in search_engine_results][:urls_limit]

        html_list = []
        header_docs: list[HeaderDoc] = []
        search_results: list[HeaderSearchResult] = []

        headers_total_tokens = 0
        headers_high_score_tokens = 0
        headers_medium_score_tokens = 0
        headers_mtld_score_average = 0

        all_started_urls = []
        all_completed_urls = []
        all_searched_urls = []
        all_urls_with_high_scores = []
        all_urls_with_low_scores = []

        # Stage 2: Scraping and Processing Loop
        async for url, status, html in scrape_urls(urls, show_progress=True):
            if status == "started":
                all_started_urls.append(url)
            elif status == "completed" and html:
                all_completed_urls.append(url)
                html_list.append(html)

                with tool_span(
                    name="tool.process_url",
                    tool_name="process_url",
                    parameters={"url": url},
                ) as url_span:
                    sub_source_dir = format_sub_source_dir(url)
                    sub_output_dir = os.path.join(
                        query_output_dir, "pages", sub_source_dir
                    )

                    save_file(html, f"{sub_output_dir}/page.html")
                    save_file(
                        preprocess_html(html),
                        f"{sub_output_dir}/page_preprocessed.html",
                    )

                    links = set(scrape_links(html, url))
                    links = [
                        link
                        for link in links
                        if (
                            link != url if isinstance(link, str) else link["url"] != url
                        )
                    ]
                    save_file(links, os.path.join(sub_output_dir, "links.json"))

                    doc_markdown = convert_html_to_markdown(html, ignore_links=False)
                    save_file(doc_markdown, f"{sub_output_dir}/page.md")

                    doc_analysis = analyze_markdown(doc_markdown)
                    save_file(doc_analysis, f"{sub_output_dir}/analysis.json")

                    doc_markdown_tokens = base_parse_markdown(doc_markdown)
                    save_file(
                        doc_markdown_tokens, f"{sub_output_dir}/markdown_tokens.json"
                    )

                    original_docs: list[HeaderDoc] = derive_by_header_hierarchy(
                        doc_markdown, ignore_links=True
                    )
                    save_file(original_docs, f"{sub_output_dir}/docs.json")

                    for doc in original_docs:
                        doc["source"] = url

                    # Vector Search with observed embed_batch
                    with embedding_span(
                        name="vector.search_headers",
                        model_name=resolve_model_value(embed_model),
                        texts=[redact(query)],
                    ) as emb_span:
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
                                merge_chunks=merge_chunks,
                            )
                        )
                        emb_span.set_attribute("vector.results_count", len(sub_results))

                    all_searched_urls.append(url)

                    # Quality Filtering
                    filtered_sub_results = []
                    for result in sub_results:
                        ltr = link_to_text_ratio(result["content"])
                        result["metadata"]["ltr_ratio"] = ltr
                        mtld_result = calculate_mtld(result["content"])
                        result["metadata"]["mtld"] = mtld_result
                        result["metadata"]["mtld_category"] = calculate_mtld_category(
                            mtld_result
                        )
                        if (
                            result["score"] >= medium_quality_score
                            and result["metadata"]["mtld_category"] != "very_low"
                        ):
                            filtered_sub_results.append(result)

                    sub_total_tokens = sum(
                        result["metadata"]["num_tokens"]
                        for result in filtered_sub_results
                    )
                    sub_high_score_tokens = sum(
                        result["metadata"]["num_tokens"]
                        for result in filtered_sub_results
                        if result["score"] >= high_quality_score
                    )
                    sub_medium_score_tokens = sum(
                        result["metadata"]["num_tokens"]
                        for result in filtered_sub_results
                        if medium_quality_score <= result["score"] < high_quality_score
                    )

                    sub_mtld_score_values = [
                        calculate_mtld(result["content"])
                        for result in filtered_sub_results
                        if result["score"] >= high_quality_score
                    ]
                    sub_mtld_score_average = (
                        sum(sub_mtld_score_values) / len(sub_mtld_score_values)
                        if sub_mtld_score_values
                        else 0
                    )

                    save_file(
                        {
                            "query": query,
                            "url": url,
                            "count": len(filtered_sub_results),
                            "max_score": max(
                                (r["score"] for r in filtered_sub_results), default=0.0
                            ),
                            "min_score": min(
                                (r["score"] for r in filtered_sub_results), default=0.0
                            ),
                            "mtld": calculate_mtld(html),
                            "mtld_category": calculate_mtld_category(
                                calculate_mtld(html)
                            ),
                            "total_tokens": sub_total_tokens,
                            "high_score_tokens": sub_high_score_tokens,
                            "medium_score_tokens": sub_medium_score_tokens,
                            "mtld_score_average": sub_mtld_score_average,
                            "results": filtered_sub_results,
                        },
                        f"{sub_output_dir}/search_results.json",
                    )

                    header_docs.extend(original_docs)
                    search_results.extend(filtered_sub_results)
                    if sub_high_score_tokens > 0:
                        all_urls_with_high_scores.append(url)
                    else:
                        all_urls_with_low_scores.append(url)

                    headers_total_tokens += sub_total_tokens
                    headers_high_score_tokens += sub_high_score_tokens
                    headers_medium_score_tokens += sub_medium_score_tokens
                    headers_mtld_score_average += round(sub_mtld_score_average, 2)

                    url_span.set_attribute("processed_tokens", sub_total_tokens)
                    url_span.set_attribute("high_score_tokens", sub_high_score_tokens)

                    if (
                        headers_high_score_tokens >= target_high_score_tokens
                        or (headers_high_score_tokens + headers_medium_score_tokens)
                        >= target_medium_score_tokens
                    ):
                        logger.info(
                            f"Stopping processing: {headers_high_score_tokens} high-score tokens "
                            f"and {headers_medium_score_tokens} medium-score tokens collected from source: {url}."
                        )
                        break

        # Cleanup zero-token dirs
        for url in all_completed_urls:
            sub_source_dir = format_sub_source_dir(url)
            sub_output_dir = os.path.join(query_output_dir, "pages", sub_source_dir)
            sub_results_path = f"{sub_output_dir}/search_results.json"
            if os.path.exists(sub_results_path):
                sub_results_data = load_file(sub_results_path)
                if sub_results_data.get("total_tokens", 0) == 0:
                    shutil.rmtree(sub_output_dir, ignore_errors=True)
                    logger.info(
                        f"Removed {sub_output_dir} due to zero total tokens during final cleanup."
                    )

        save_file(
            {
                "expected_order": urls,
                "started_urls": all_started_urls,
                "searched_urls": all_searched_urls,
                "high_score_urls": create_url_dict_list(
                    all_urls_with_high_scores,
                    search_results,
                    high_quality_score=high_quality_score,
                    medium_quality_score=medium_quality_score,
                ),
            },
            f"{query_output_dir}/_scraped_url_order_logs.json",
        )

        search_results = sorted(search_results, key=lambda x: x["score"], reverse=True)
        for i, result in enumerate(search_results, 1):
            result["rank"] = i

        save_file(
            {"query": query, "count": len(header_docs), "documents": header_docs},
            f"{query_output_dir}/docs.json",
        )

        url_stats = defaultdict(
            lambda: {
                "high_score_tokens": 0,
                "medium_score_tokens": 0,
                "header_count": 0,
            }
        )
        for result in search_results:
            url = result["metadata"].get("source", "Unknown")
            if result["score"] >= high_quality_score:
                url_stats[url]["high_score_tokens"] += result["metadata"].get(
                    "num_tokens", 0
                )
                url_stats[url]["header_count"] += 1
            elif result["score"] >= medium_quality_score:
                url_stats[url]["medium_score_tokens"] += result["metadata"].get(
                    "num_tokens", 0
                )
                url_stats[url]["header_count"] += 1

        sorted_urls = [
            {
                "url": url,
                "high_score_tokens": stats["high_score_tokens"],
                "medium_score_tokens": stats["medium_score_tokens"],
                "header_count": stats["header_count"],
            }
            for url, stats in sorted(
                url_stats.items(),
                key=lambda x: (x[1]["high_score_tokens"], x[1]["medium_score_tokens"]),
                reverse=True,
            )
            if stats["high_score_tokens"] > 0 or stats["medium_score_tokens"] > 0
        ]

        save_file(
            {
                "query": query,
                "count": len(search_results),
                "max_score": max((r["score"] for r in search_results), default=0.0),
                "min_score": min((r["score"] for r in search_results), default=0.0),
                "total_tokens": headers_total_tokens,
                "high_score_tokens": headers_high_score_tokens,
                "medium_score_tokens": headers_medium_score_tokens,
                "mtld_score_average": headers_mtld_score_average,
                "settings": {
                    "urls_limit": urls_limit,
                    "model": resolve_model_value(embed_model),
                    "chunk_size": chunk_size,
                    "overlap": chunk_overlap,
                },
                "urls": sorted_urls,
                "results": search_results,
            },
            f"{query_output_dir}/search_results.json",
        )

        sorted_urls = sort_urls_by_high_and_medium_score_tokens(
            search_results, medium_quality_score=medium_quality_score
        )
        sorted_results = sort_search_results_by_url_and_category(
            search_results,
            sorted_urls,
            high_quality_score=high_quality_score,
            medium_quality_score=medium_quality_score,
        )
        total_tokens = sum(r["metadata"].get("num_tokens", 0) for r in sorted_results)

        save_file(
            {
                "query": query,
                "count": len(sorted_results),
                "total_tokens": total_tokens,
                "results": sorted_results,
            },
            f"{query_output_dir}/sorted_search_results.json",
        )

        # Token-limited filtering
        current_tokens = 0
        filtered_results = []
        for result in sorted_results:
            content = f"{result['header']}\n{result['content']}"
            tokens = count_tokens(content, model=llm_model)
            if current_tokens + tokens > max_tokens:
                break
            filtered_results.append(result)
            current_tokens += tokens

        filtered_url_stats = defaultdict(
            lambda: {
                "high_score_tokens": 0,
                "medium_score_tokens": 0,
                "header_count": 0,
            }
        )
        for result in filtered_results:
            url = result["metadata"]["source"]
            if result["score"] >= high_quality_score:
                filtered_url_stats[url]["high_score_tokens"] += result["metadata"].get(
                    "num_tokens", 0
                )
                filtered_url_stats[url]["header_count"] += 1
            elif result["score"] >= medium_quality_score:
                filtered_url_stats[url]["medium_score_tokens"] += result[
                    "metadata"
                ].get("num_tokens", 0)
                filtered_url_stats[url]["header_count"] += 1

        filtered_urls = [
            {
                "url": url,
                "high_score_tokens": stats["high_score_tokens"],
                "medium_score_tokens": stats["medium_score_tokens"],
                "header_count": stats["header_count"],
            }
            for url, stats in sorted(
                filtered_url_stats.items(),
                key=lambda x: (x[1]["high_score_tokens"], x[1]["medium_score_tokens"]),
                reverse=True,
            )
        ]

        save_file(
            {
                "query": query,
                "count": len(filtered_results),
                "total_tokens": current_tokens,
                "urls": filtered_urls,
                "results": filtered_results,
            },
            f"{query_output_dir}/contexts.json",
        )

        # Stage 3: Context Assembly
        with tool_span(
            name="tool.assemble_context",
            tool_name="group_results_by_source",
            parameters={"result_count": len(filtered_results)},
        ) as ctx_span:
            context = group_results_by_source_for_llm_context(
                filtered_results,
                high_quality_score=high_quality_score,
                medium_quality_score=medium_quality_score,
            )
            ctx_span.set_attribute("context_length", len(context))

        save_file(context, f"{query_output_dir}/context.md")

        # Stage 4: LLM Generation using observed achat
        prompt = PROMPT_TEMPLATE.format(query=query, context=context)
        messages = [{"role": "user", "content": prompt}]
        save_file(messages, f"{query_output_dir}/messages.json")

        with llm_span(
            name="llm.generate_answer",
            model_name=resolve_model_value(llm_model),
            messages=messages,
            invocation_params={"temperature": 0.3, "stream": True},
            provider="llama_cpp",
        ) as llm_trace_span:
            result_obj = await achat(
                prompt_or_messages=messages,
                model=llm_model,
                temperature=0.3,
                project_name=PROJECT_NAME,
                capture_content=True,
                session_id=session_id,
            )
            llm_response = result_obj.content

            input_tokens = count_tokens(prompt, model=llm_model)
            output_tokens = count_tokens(llm_response, model=llm_model)

            llm_trace_span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT, input_tokens
            )
            llm_trace_span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, output_tokens
            )
            llm_trace_span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL, input_tokens + output_tokens
            )
            llm_trace_span.set_attribute(
                SpanAttributes.LLM_OUTPUT_MESSAGES,
                json.dumps(
                    [{"role": "assistant", "content": redact(llm_response[:2000])}]
                ),
            )

        save_file(llm_response, f"{query_output_dir}/response.md")
        save_file(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            f"{query_output_dir}/tokens_info.json",
        )

        # 1. Set Status to OK (or ERROR if an exception occurred)
        from opentelemetry.trace import Status, StatusCode

        root_span.set_status(Status(StatusCode.OK))

        # 2. Populate Output Value (redacted LLM response)
        root_span.set_attribute(
            SpanAttributes.OUTPUT_VALUE,
            redact(llm_response[:4000]),  # Truncate to avoid OTLP payload limits
        )
        root_span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "text/plain")

        # 3. Optional: Add pipeline-level metrics to the root span
        elapsed = time.time() - start_time
        root_span.set_attribute("pipeline.elapsed_seconds", round(elapsed, 4))
        root_span.set_attribute("pipeline.total_tokens_processed", headers_total_tokens)
        root_span.set_attribute("pipeline.urls_scraped", len(all_completed_urls))
        root_span.set_attribute("pipeline.final_context_tokens", current_tokens)

        logger.info(f"Pipeline completed in {elapsed:.2f}s")


if __name__ == "__main__":
    asyncio.run(main(**get_args()))
