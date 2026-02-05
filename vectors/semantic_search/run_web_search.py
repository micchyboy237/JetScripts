# ──────────────────────────────────────────────────────────────────────────────
#  hybrid_search.py
# ──────────────────────────────────────────────────────────────────────────────


import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, TypedDict

from jet.adapters.llama_cpp.llm import LlamacppLLM
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS, LLAMACPP_LLM_KEYS
from jet.code.markdown_types.markdown_parsed_types import HeaderDoc, HeaderSearchResult
from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.code.markdown_utils._markdown_parser import (
    derive_by_header_hierarchy,
)
from jet.code.markdown_utils._preprocessors import link_to_text_ratio
from jet.models.tokenizer.base import count_tokens
from jet.scrapers.hrequests_utils import ScrapeStatus, scrape_urls
from jet.scrapers.utils import search_data
from jet.vectors.semantic_search.header_vector_search import search_headers
from jet.vectors.semantic_search.web_search import (
    HIGH_QUALITY_SCORE,
    MEDIUM_QUALITY_SCORE,
    PROMPT_TEMPLATE,
    TARGET_HIGH_SCORE_TOKENS,
    TARGET_MEDIUM_SCORE_TOKENS,
    group_results_by_source_for_llm_context,
)
from jet.wordnet.analyzers.text_analysis import calculate_mtld, calculate_mtld_category

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem


class SearchStats(TypedDict):
    total_tokens: int
    high_score_tokens: int
    medium_score_tokens: int
    mtld_score_average: float
    urls_with_high_scores: list[str]
    urls_with_low_scores: list[str]


class UrlTokenStat(TypedDict):
    url: str
    high_score_tokens: int
    medium_score_tokens: int
    header_count: int


class HtmlStatusItem(TypedDict):
    status: ScrapeStatus
    html: str


HtmlStatus = dict[str, HtmlStatusItem]


class HybridSearchResult(TypedDict):
    query: str
    search_engine_results: list[dict]
    collected_urls: list[str]
    header_docs: list[HeaderDoc]
    all_search_results: list[HeaderSearchResult]  # before final filtering
    filtered_results: list[HeaderSearchResult]  # after token budget
    grouped_context: str
    llm_messages: list[dict]
    llm_response: str
    token_counts: dict[Literal["input", "output", "total"], int]
    stats: SearchStats
    url_stats: list[UrlTokenStat]
    settings: dict[str, Any]
    all_htmls_with_status: HtmlStatus


async def hybrid_search(
    query: str, llm_log_dir: Path | str | None = None
) -> HybridSearchResult:
    """Perform hybrid search and return structured results without side-effects."""

    # ── Configuration ────────────────────────────────────────────────────────
    embed_model: LLAMACPP_EMBED_KEYS = "nomic-embed-text"
    llm_model: LLAMACPP_LLM_KEYS = "qwen3-instruct-2507:4b"
    max_tokens = 4000
    use_cache = True
    urls_limit = 10

    top_k = None
    threshold = 0.0
    chunk_size = 200
    chunk_overlap = 50
    merge_chunks = False

    # We no longer create directories here unless explicitly requested
    # (caller responsibility)

    search_engine_results = search_data(query, use_cache=use_cache)

    urls = [r["url"] for r in search_engine_results][:urls_limit]

    html_list: list[str] = []
    header_docs: list[HeaderDoc] = []
    search_results: list[HeaderSearchResult] = []

    headers_total_tokens = 0
    headers_high_score_tokens = 0
    headers_medium_score_tokens = 0
    headers_mtld_scores: list[float] = []

    all_started_urls = []
    all_completed_urls = []
    all_searched_urls = []
    all_urls_with_high_scores = []
    all_urls_with_low_scores = []
    all_htmls_with_status: HtmlStatus = {}

    async for url, status, html in scrape_urls(urls, show_progress=True):
        if html:
            all_htmls_with_status[url] = {
                "status": status,
                "html": html,
            }

        if status == "started":
            all_started_urls.append(url)
            continue

        if status != "completed" or not html:
            continue

        all_completed_urls.append(url)
        html_list.append(html)
        all_searched_urls.append(url)

        # ── Per-page saving removed ───────────────────────────────────────
        # sub_source_dir = ...
        # sub_output_dir = ...
        # save_file(html, ...)
        # save_file(preprocess_html(html), ...)
        # save_file(links, ...)
        # save_file(doc_markdown, ...)
        # save_file(doc_analysis, ...)
        # save_file(doc_markdown_tokens, ...)
        # save_file(original_docs, ...)

        original_docs: list[HeaderDoc] = derive_by_header_hierarchy(
            convert_html_to_markdown(html, ignore_links=True), ignore_links=True
        )

        for doc in original_docs:
            doc["source"] = url  # type: ignore

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

        filtered_sub_results = []
        for result in sub_results:
            ltr = link_to_text_ratio(result["content"])
            mtld = calculate_mtld(result["content"])
            mtld_cat = calculate_mtld_category(mtld)

            result["metadata"]["ltr_ratio"] = ltr
            result["metadata"]["mtld"] = mtld
            result["metadata"]["mtld_category"] = mtld_cat

            if result["score"] >= MEDIUM_QUALITY_SCORE and mtld_cat != "very_low":
                filtered_sub_results.append(result)

        sub_total_tokens = sum(
            r["metadata"].get("num_tokens", 0) for r in filtered_sub_results
        )
        sub_high = sum(
            r["metadata"].get("num_tokens", 0)
            for r in filtered_sub_results
            if r["score"] >= HIGH_QUALITY_SCORE
        )
        sub_medium = sum(
            r["metadata"].get("num_tokens", 0)
            for r in filtered_sub_results
            if MEDIUM_QUALITY_SCORE <= r["score"] < HIGH_QUALITY_SCORE
        )

        sub_mtld_values = [
            r["metadata"]["mtld"]
            for r in filtered_sub_results
            if r["score"] >= HIGH_QUALITY_SCORE
        ]
        sub_mtld_avg = (
            sum(sub_mtld_values) / len(sub_mtld_values) if sub_mtld_values else 0.0
        )

        header_docs.extend(original_docs)
        search_results.extend(filtered_sub_results)

        if sub_high > 0:
            all_urls_with_high_scores.append(url)
        else:
            all_urls_with_low_scores.append(url)

        headers_total_tokens += sub_total_tokens
        headers_high_score_tokens += sub_high
        headers_medium_score_tokens += sub_medium
        headers_mtld_scores.append(sub_mtld_avg)

        if (
            headers_high_score_tokens >= TARGET_HIGH_SCORE_TOKENS
            or (headers_high_score_tokens + headers_medium_score_tokens)
            >= TARGET_MEDIUM_SCORE_TOKENS
        ):
            logger.info(f"Early stop after {url} – token targets reached")
            break

    # ── Final aggregation & sorting ──────────────────────────────────────────

    # Sort & rank
    search_results = sorted(search_results, key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(search_results, 1):
        r["rank"] = i

    # URL stats
    url_stats = defaultdict(lambda: {"high": 0, "medium": 0, "count": 0})
    for r in search_results:
        src = r["metadata"].get("source", "unknown")
        tok = r["metadata"].get("num_tokens", 0)
        if r["score"] >= HIGH_QUALITY_SCORE:
            url_stats[src]["high"] += tok
            url_stats[src]["count"] += 1
        elif r["score"] >= MEDIUM_QUALITY_SCORE:
            url_stats[src]["medium"] += tok
            url_stats[src]["count"] += 1

    sorted_url_stats: list[UrlTokenStat] = sorted(
        (
            {
                "url": url,
                "high_score_tokens": s["high"],
                "medium_score_tokens": s["medium"],
                "header_count": s["count"],
            }
            for url, s in url_stats.items()
            if s["high"] > 0 or s["medium"] > 0
        ),
        key=lambda x: (x["high_score_tokens"], x["medium_score_tokens"]),
        reverse=True,
    )

    # Final token-limited filtering
    current_tokens = 0
    filtered_results = []
    for result in search_results:  # already sorted by score
        content = f"{result['header']}\n{result['content']}"
        tokens = count_tokens(llm_model, content)
        if current_tokens + tokens > max_tokens:
            break
        filtered_results.append(result)
        current_tokens += tokens

    # Build final context
    grouped_context = group_results_by_source_for_llm_context(
        filtered_results, llm_model
    )

    # LLM call
    llm = LlamacppLLM(
        model=llm_model,
        base_url="http://shawn-pc.local:8080/v1",
        verbose=True,
        log_dir=str(llm_log_dir) if llm_log_dir else None,
    )
    prompt = PROMPT_TEMPLATE.format(query=query, context=grouped_context)
    messages = [{"role": "user", "content": prompt}]
    llm_response = llm.chat(messages, temperature=0.3)

    input_tokens = count_tokens(llm_model, prompt)
    output_tokens = count_tokens(llm_model, llm_response)

    return {
        "query": query,
        "search_engine_results": search_engine_results,
        "collected_urls": all_completed_urls,
        "header_docs": header_docs,
        "all_search_results": search_results,
        "filtered_results": filtered_results,
        "grouped_context": grouped_context,
        "llm_messages": messages,
        "llm_response": llm_response,
        "token_counts": {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
        },
        "stats": {
            "total_tokens": headers_total_tokens,
            "high_score_tokens": headers_high_score_tokens,
            "medium_score_tokens": headers_medium_score_tokens,
            "mtld_score_average": round(
                sum(headers_mtld_scores) / len(headers_mtld_scores), 2
            )
            if headers_mtld_scores
            else 0.0,
            "urls_with_high_scores": all_urls_with_high_scores,
            "urls_with_low_scores": all_urls_with_low_scores,
        },
        "url_stats": sorted_url_stats,
        "settings": {
            "urls_limit": urls_limit,
            "embed_model": embed_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "max_tokens": max_tokens,
        },
        "all_htmls_with_status": all_htmls_with_status,
    }


if __name__ == "__main__":
    import argparse
    import asyncio
    import shutil

    from jet.code.html_utils import preprocess_html
    from jet.code.markdown_utils._converters import convert_html_to_markdown
    from jet.code.markdown_utils._markdown_analyzer import analyze_markdown
    from jet.code.markdown_utils._markdown_parser import base_parse_markdown
    from jet.file.utils import save_file
    from jet.scrapers.utils import scrape_links
    from jet.utils.text import format_sub_dir, format_sub_source_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="Top 10 isekai anime 2026")
    args = parser.parse_args()

    query_output_dir = f"{OUTPUT_DIR}/{format_sub_dir(args.query)}"
    shutil.rmtree(query_output_dir, ignore_errors=True)

    llm_log_dir = Path(query_output_dir) / "llm_calls"

    result = asyncio.run(hybrid_search(args.query, llm_log_dir=llm_log_dir))

    print(f"Found {len(result['filtered_results'])} relevant chunks")
    print(f"LLM response length: {len(result['llm_response'])} chars")

    search_engine_results = result.pop("search_engine_results")
    header_docs = result.pop("header_docs")
    all_search_results = result.pop("all_search_results")
    filtered_results = result.pop("filtered_results")
    url_stats = result.pop("url_stats")
    stats = result.pop("stats")
    grouped_context = result.pop("grouped_context")
    llm_messages = result.pop("llm_messages")
    llm_response = result.pop("llm_response")
    token_counts = result.pop("token_counts")
    all_htmls_with_status = result.pop("all_htmls_with_status")

    save_file(result, f"{query_output_dir}/result_meta.json")
    save_file(search_engine_results, f"{query_output_dir}/search_engine_results.json")
    save_file(header_docs, f"{query_output_dir}/header_docs.json")
    save_file(all_search_results, f"{query_output_dir}/all_search_results.json")
    save_file(filtered_results, f"{query_output_dir}/filtered_results.json")
    save_file(url_stats, f"{query_output_dir}/url_stats.json")
    save_file(stats, f"{query_output_dir}/stats.json")
    save_file(grouped_context, f"{query_output_dir}/grouped_context.md")
    save_file(llm_messages, f"{query_output_dir}/llm_messages.json")
    save_file(llm_response, f"{query_output_dir}/llm_response.md")
    save_file(token_counts, f"{query_output_dir}/token_counts.json")

    for url, info in all_htmls_with_status.items():
        status = info["status"]
        html = info["html"]

        sub_source_dir = format_sub_source_dir(url)
        sub_output_dir = Path(query_output_dir) / "pages" / sub_source_dir
        save_file(html, f"{sub_output_dir}/page.html")
        save_file(preprocess_html(html), f"{sub_output_dir}/page_preprocessed.html")

        links = set(scrape_links(html, url))
        links = [
            link
            for link in links
            if (link != url if isinstance(link, str) else link["url"] != url)
        ]
        save_file(links, sub_output_dir / "links.json")

        doc_markdown = convert_html_to_markdown(html, ignore_links=False)
        save_file(doc_markdown, f"{sub_output_dir}/page.md")

        doc_analysis = analyze_markdown(doc_markdown)
        save_file(doc_analysis, sub_output_dir / "analysis.json")
        doc_markdown_tokens = base_parse_markdown(doc_markdown)
        save_file(doc_markdown_tokens, sub_output_dir / "markdown_tokens.json")

        original_docs: list[HeaderDoc] = derive_by_header_hierarchy(
            doc_markdown, ignore_links=True
        )

        save_file(original_docs, sub_output_dir / "docs.json")
