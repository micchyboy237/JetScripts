# JetScripts/search/playwright/run_playwright_extract.py

import os
import shutil
from typing import List, Optional

# --- UPDATED IMPORTS ---
from jet.adapters.llama_cpp.hybrid_utils import HybridSearchResult, hybrid_search
from jet.code.markdown_types import HeaderSearchResult
from jet.code.markdown_utils._markdown_parser import derive_by_header_hierarchy
from jet.code.markdown_utils._preprocessors import clean_markdown_links
from jet.file.utils import save_file
from jet.logger import logger
from jet.scrapers.utils import search_data
from jet.search.playwright.playwright_extract import (
    PlaywrightExtract,
    convert_html_to_markdown,
)
from jet.utils.text import format_sub_dir

# -----------------------

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "generated",
    os.path.splitext(os.path.basename(__file__))[0],
)

# ... [ContextItem, Topic, SearchResult TypedDicts remain unchanged] ...


def extract_doc_chunks(
    html: str, url: str, chunk_size: int = 200, chunk_overlap: int = 50
) -> List[dict]:
    """Extract structured chunks from HTML content."""
    md_content = convert_html_to_markdown(html, ignore_links=True)
    headings = derive_by_header_hierarchy(md_content, ignore_links=True)
    # Return format compatible with hybrid_search documents list
    return [
        {"id": f"{url}#{i}", "content": f"{header['header']}\n{header['content']}"}
        for i, header in enumerate(headings)
        if header.get("content")
    ]


# ... [extract_topics and _fallback_topic_extraction remain unchanged as no direct adapter exists yet] ...


def search_contexts(query: str, html: str, url: str) -> List[HeaderSearchResult]:
    """
    Search for relevant contexts using Hybrid Search (Vector + Rerank).
    Replaces legacy semantic_search with jet.adapters.llama_cpp.hybrid_utils.
    """
    chunks = extract_doc_chunks(html, url)
    if not chunks:
        return []

    texts = [chunk["content"] for chunk in chunks]

    logger.info(
        f"Running hybrid search for '{query}' on {len(texts)} chunks from {url}"
    )

    # Use hybrid_search which handles embedding, vector retrieval, AND reranking
    results: List[HybridSearchResult] = hybrid_search(
        query=query,
        documents=texts,
        top_n=5,  # Limit to top 5 most relevant sections
        vector_score_threshold=0.3,  # Filter out low-relevance candidates early
        normalize_scores=True,  # Return 0-1 scores for easier interpretation
    )

    # Map HybridSearchResult back to HeaderSearchResult format
    search_results: List[HeaderSearchResult] = []
    for r in results:
        search_results.append(
            {
                "id": chunks[r["index"]]["id"],
                "rank": r["rank"],
                "score": r["score"],  # Normalized rerank score
                "text": r["text"],
                "metadata": {
                    "vector_score": r["vector_score"],
                    "raw_rerank_score": r["rerank_score_raw"],
                },
            }
        )

    return search_results


def scrape_urls_data(
    query: str, urls: List[str], use_cache: bool = True, url_limit: Optional[int] = None
):
    """
    Scrape and process URLs.
    NOTE: Removed 'model' param - embedding/rerank models are now configured
    via jet.adapters.llama_cpp.config or environment variables.
    """
    sub_dir_query = format_sub_dir(query)
    base_output_dir = f"{OUTPUT_DIR}/{sub_dir_query}"
    shutil.rmtree(base_output_dir, ignore_errors=True)

    if not urls:
        search_engine_results = search_data(query, use_cache=use_cache)
        urls = [r["url"] for r in search_engine_results]
        save_file(
            search_engine_results, f"{base_output_dir}/search_engine_results.json"
        )
        save_file(urls, f"{base_output_dir}/urls.json")

    extractor = PlaywrightExtract()
    result_stream = extractor._stream(
        urls=urls,
        extract_depth="advanced",
        include_images=True,
        include_favicon=True,
        format="text",
        url_limit=url_limit,
    )

    print("\nAdvanced extract results stream:")
    count = 0
    all_headers = {}

    for result in result_stream:
        count += 1
        meta = result.copy().pop("meta")
        chunks = extract_doc_chunks(meta["html"], result["url"])
        documents = [doc["content"] for doc in chunks]

        # Updated: No longer passing model explicitly
        search_results = search_contexts(query, meta["html"], result["url"])

        sub_dir_url = format_sub_dir(result["url"])
        print(
            f"URL: {sub_dir_url} (Images: {len(result['images'])}, Favicon: {result['favicon']})"
        )

        # Save artifacts
        save_file(
            {"query": query, "count": len(chunks), "chunks": chunks},
            f"{base_output_dir}/{sub_dir_url}/chunks.json",
        )
        save_file(
            {"query": query, "count": len(documents), "documents": documents},
            f"{base_output_dir}/{sub_dir_url}/documents.json",
        )

        md_content = convert_html_to_markdown(meta["html"], ignore_links=True)
        headers = derive_by_header_hierarchy(md_content, ignore_links=True)
        save_file(
            {"query": query, "count": len(headers), "headers": headers},
            f"{base_output_dir}/{sub_dir_url}/headers.json",
        )
        all_headers[result["url"]] = headers

        save_file(result, f"{base_output_dir}/{sub_dir_url}/results.json")
        save_file(
            search_results, f"{base_output_dir}/{sub_dir_url}/search_results.json"
        )
        save_file(
            {"url": result["url"], "tokens": meta["tokens"]},
            f"{base_output_dir}/{sub_dir_url}/info.json",
        )
        save_file(meta["analysis"], f"{base_output_dir}/{sub_dir_url}/analysis.json")
        save_file(
            meta["text_links"], f"{base_output_dir}/{sub_dir_url}/text_links.json"
        )
        save_file(
            meta["image_links"], f"{base_output_dir}/{sub_dir_url}/image_links.json"
        )
        save_file(meta["html"], f"{base_output_dir}/{sub_dir_url}/page.html")
        save_file(meta["markdown"], f"{base_output_dir}/{sub_dir_url}/markdown.md")
        save_file(
            clean_markdown_links(meta["markdown"]),
            f"{base_output_dir}/{sub_dir_url}/markdown_no_links.md",
        )
        save_file(meta["md_tokens"], f"{base_output_dir}/{sub_dir_url}/md_tokens.json")
        save_file(meta["screenshot"], f"{base_output_dir}/{sub_dir_url}/screenshot.png")

    return all_headers


if __name__ == "__main__":
    urls = []
    query = "Top RAG context engineering tips 2026 reddit"
    url_limit = 20
    use_cache = True

    # Updated: Removed model argument
    sub_dir_query = format_sub_dir(query)
    all_headers = scrape_urls_data(
        query, urls, use_cache=use_cache, url_limit=url_limit
    )
    save_file(all_headers, f"{OUTPUT_DIR}/{sub_dir_query}/all_headers.json")

    all_contexts = []
    for url, headers in all_headers.items():
        context = f"<!-- Source: {url} -->\n"
        combined_headers_string = "\n".join(
            [f"{header['header']}\n{header['content']}" for header in headers]
        )
        context += combined_headers_string
        all_contexts.append(context)

    all_contexts_str = "\n\n".join(all_contexts)
    save_file(all_contexts_str, f"{OUTPUT_DIR}/{sub_dir_query}/all_contexts.md")
