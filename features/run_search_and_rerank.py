import argparse
from collections import defaultdict
import json
import math
import os
import re
import shutil
from typing import Dict, List, Optional, Tuple, TypedDict
from datetime import datetime
import asyncio
from urllib.parse import unquote, urlparse
from jet.code.markdown_utils import analyze_markdown, parse_markdown
from jet.data.sample_diverse_headers import sample_diverse_headers
from jet.scrapers.preprocessor import convert_html_to_markdown
from jet.code.markdown_utils import convert_html_to_markdown as convert_html_to_markdownify
from jet.features.nltk_search import get_pos_tag, search_by_pos
from jet.llm.mlx.helpers.base import get_system_date_prompt
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.logger import logger
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.scrapers.hrequests_utils import scrape_urls
from jet.transformers.link_formatters import LinkFormatter, format_links_for_embedding
from jet.utils.url_utils import rerank_urls_bm25_plus
from jet.vectors.document_types import HeaderDocument, HeaderDocumentWithScore
from jet.vectors.search_with_clustering import search_documents
from jet.wordnet.analyzers.text_analysis import ReadabilityResult, analyze_readability, analyze_text
from jet.code.splitter_markdown_utils import get_md_header_docs, get_header_level
from jet.file.utils import save_file
from jet.llm.mlx.base import MLX
from jet.models.tokenizer.base import count_tokens, get_tokenizer_fn
from jet.scrapers.browser.playwright_utils import scrape_multiple_urls

from jet.scrapers.utils import scrape_links, scrape_published_date, search_data
from jet.models.tasks.hybrid_search_docs_with_bm25 import search_docs
from jet.llm.evaluators.context_relevancy_evaluator import evaluate_context_relevancy
from jet.llm.evaluators.response_relevancy_evaluator import evaluate_response_relevancy
# from jet.llm.mlx.tasks.eval.evaluate_context_relevance import evaluate_context_relevance
# from jet.llm.mlx.tasks.eval.evaluate_response_relevance import evaluate_response_relevance
from jet.wordnet.similarity import group_similar_headers
from jet.wordnet.text_chunker import chunk_headers
from jet.wordnet.words import count_words
from jet.search.searxng import SearchResult as BrowserSearchResult


class StepBackQueryResponse(TypedDict):
    original_query: str
    broader_query: List[str]


class ContextEntry(TypedDict):
    rank: int
    doc_index: int
    chunk_index: int
    tokens: int
    score: float
    rerank_score: float
    source_url: str
    parent_header: str
    header: str
    content: str


class ContextInfo(TypedDict):
    model: str
    total_tokens: int
    contexts: list[ContextEntry]


def get_header_stats(text: str) -> Dict:
    """Analyze text and return header statistics."""
    logger.debug("Analyzing text for header statistics")
    analysis = analyze_text(text)
    logger.info(
        f"Header stats computed: MTLD={analysis['mtld']}, Difficulty={analysis['overall_difficulty']}")
    return {
        "mtld": analysis["mtld"],
        "mtld_category": analysis["mtld_category"],
        "overall_difficulty": analysis["overall_difficulty"],
        "overall_difficulty_category": analysis["overall_difficulty_category"],
    }


def format_sub_dir(text: str) -> str:
    return text.lower().strip('.,!?').replace(' ', '_').replace(
        '.', '_').replace(',', '_').replace('!', '_').replace('?', '_').strip()


def initialize_output_directory(script_path: str, query: str) -> str:
    """Create and return the output directory path."""
    logger.debug(f"Initializing output directory for script: {script_path}")
    script_dir = os.path.dirname(os.path.abspath(script_path))
    output_dir = os.path.join(script_dir, "generated", os.path.splitext(
        os.path.basename(script_path))[0])
    query_sub_dir = format_sub_dir(query)
    output_dir = os.path.join(output_dir, query_sub_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory initialized: {output_dir}")
    return output_dir


def format_sub_url_dir(url: str) -> str:
    """Format a URL into a lowercase directory name, replacing hyphens and spaces with underscores."""
    clean_url = re.sub(r'^(https?://|www\.)|(\?.*)', '', url)
    formatted = re.sub(r'[- ]+', '_', clean_url).lower()
    formatted = re.sub(r'[^\w./]', '_', formatted)
    formatted = re.sub(r'_+', '_', formatted)
    return formatted.strip('_')


def initialize_search_components(
    llm_model: LLMModelType,
    embed_model: EmbedModelType,
    seed: int
) -> Tuple[MLX, callable]:
    """Initialize MLX model and tokenizer."""
    logger.debug(
        f"Initializing search components with LLM={llm_model}, Embed={embed_model}, Seed={seed}")
    mlx = MLXModelRegistry.load_model(llm_model, seed=seed)
    tokenize = get_tokenizer_fn(embed_model)
    logger.info("Search components initialized successfully")
    return mlx, tokenize


async def fetch_search_results(query: str, output_dir: str, use_cache: bool = False) -> List[BrowserSearchResult]:
    """Fetch search results and save them."""
    logger.info(
        f"Fetching search results for query: {query}, use_cache={use_cache}")
    browser_search_results = search_data(query, use_cache=use_cache)
    logger.debug(f"Fetched {len(browser_search_results)} search results")
    save_file(
        {"query": query, "count": len(
            browser_search_results), "results": browser_search_results},
        os.path.join(output_dir, "browser_search_results.json")
    )
    return browser_search_results


async def process_search_results(
    browser_search_results: List[BrowserSearchResult],
    query: str,
    output_dir: str,
    top_k: Optional[int] = None
) -> List[Tuple[str, str, Optional[str]]]:
    """Process search results and extract links, ensuring top 5 URLs are always included."""
    if not top_k:
        top_k = 10

    logger.info(
        f"Processing {len(browser_search_results)} search results for query: {query}")
    guaranteed_top_n = min(5, len(browser_search_results))
    top_urls = [item["url"]
                for item in browser_search_results[:guaranteed_top_n]]
    logger.debug(f"Guaranteed top {guaranteed_top_n} URLs: {top_urls}")
    browser_search_docs = [
        HeaderDocument(
            id=result["id"],
            text=f"{result['title']}\n{result['content']}",
            metadata={
                "source_url": result["url"],
                "header": result["title"],
                "content": result["content"],
            }
        )
        for result in browser_search_results
        if result.get("title") and result.get("content")
    ]
    browser_search_doc_results = search_docs(
        query=query,
        documents=browser_search_docs,
        ids=[doc["id"] for doc in browser_search_docs],
        top_k=None,
        filter_by_headers_enabled=False
    )
    save_file(browser_search_doc_results,
              f"{output_dir}/browser_search_doc_results.json")
    filtered_ids = {result["id"] for result in browser_search_doc_results}
    selected_urls = top_urls + [
        doc["source_url"] for doc in browser_search_docs
        if doc["id"] in filtered_ids and doc["source_url"] not in top_urls
    ]
    logger.debug(f"Selected {len(selected_urls)} URLs: {selected_urls}")
    url_to_result = {r["url"]: r for r in browser_search_results}
    all_url_html_date_tuples = []
    all_links = []
    async for url, status, html in scrape_urls(selected_urls, num_parallel=10, limit=10, show_progress=True):
        if status == "completed" and html:
            docs = get_md_header_docs(
                html, ignore_links=True, metadata={"source_url": url})
            if len(docs) == 0:
                logger.debug(f"No headers found for {url}, skipping")
                continue
            sub_url_dir = format_sub_url_dir(url)
            sub_output_dir = os.path.join(output_dir, "pages", sub_url_dir)
            save_file(html, f"{sub_output_dir}/page.html")
            md_content = convert_html_to_markdown(html)
            save_file(md_content, f"{sub_output_dir}/md_content.md")
            md_content_markdownify = convert_html_to_markdownify(html)
            save_file(
                md_content_markdownify, f"{sub_output_dir}/md_content_markdownify.md")
            parse_markdown_results = parse_markdown(md_content)
            save_file(
                parse_markdown_results, f"{sub_output_dir}/parse_markdown_results.json")
            analyze_markdown_results = analyze_markdown(md_content)
            save_file(
                analyze_markdown_results, f"{sub_output_dir}/analyze_markdown_results.json")

            save_file({
                "query": query,
                "from_reranked_link": False,
                "count": len(docs),
                "source_url": url,
                "headers": {
                    f"h{i}": sum(1 for doc in docs if doc.metadata["header_level"] == i)
                    for i in range(1, 7)
                },
                "documents": docs
            }, f"{sub_output_dir}/docs.json")
            headers = [doc["header"] for doc in docs]
            save_file(headers, f"{sub_output_dir}/headers.json")
            docs_text = "\n\n".join(doc.text for doc in docs)
            readability_overall = analyze_readability(docs_text)
            save_file(readability_overall,
                      f"{sub_output_dir}/readability_overall.json")
            readability_docs = [analyze_readability(doc.text) for doc in docs]
            save_file(readability_docs,
                      f"{sub_output_dir}/readability_docs.json")
            all_url_html_date_tuples.append(
                (url, html, docs, readability_overall))
            result = url_to_result.get(url)
            if not result.get("publishedDate"):
                published_date = scrape_published_date(html)
                result["publishedDate"] = published_date if published_date else None
                logger.debug(
                    f"Scraped published date for {url}: {published_date}")
            links = set(scrape_links(html, url))
            links = [link for link in links if (
                link != url if isinstance(link, str) else link["url"] != url)]
            all_links.extend(links)
            logger.debug(f"Extracted {len(links)} links from {url}")
    all_links = list(set(all_links))
    all_links = [link for link in all_links if (link not in selected_urls if isinstance(
        link, str) else link["url"] not in selected_urls)]
    save_file(all_links, os.path.join(output_dir, "links.json"))
    logger.debug(f"Total unique links extracted: {len(all_links)}")
    reranked_links = rerank_urls_bm25_plus(all_links, query, threshold=0.7)
    logger.debug(f"Reranked to {len(reranked_links)} links")
    save_file(reranked_links, os.path.join(output_dir, "reranked_links.json"))
    remaining_k = top_k - len(all_url_html_date_tuples)
    if remaining_k > 0:
        logger.info(f"Scraping {len(reranked_links)} reranked links...")
        async for url, status, html in scrape_urls(reranked_links, num_parallel=10, show_progress=True):
            if status == "completed" and html:
                docs = get_md_header_docs(
                    html, ignore_links=True, metadata={"source_url": url})
                if len(docs) == 0:
                    logger.debug(f"No headers found for {url}, skipping")
                    continue
                sub_url_dir = format_sub_url_dir(url)
                sub_output_dir = os.path.join(output_dir, sub_url_dir)
                save_file(html, f"{sub_output_dir}/page.html")
                save_file({
                    "query": query,
                    "from_reranked_link": True,
                    "count": len(docs),
                    "source_url": url,
                    "headers": {
                        f"h{i}": sum(1 for doc in docs if doc.metadata["header_level"] == i)
                        for i in range(1, 7)
                    },
                    "documents": docs
                }, f"{sub_output_dir}/docs.json")
                headers = [doc["header"] for doc in docs]
                save_file(headers, f"{sub_output_dir}/headers.json")
                docs_text = "\n\n".join(doc.text for doc in docs)
                readability_overall = analyze_readability(docs_text)
                save_file(readability_overall,
                          f"{sub_output_dir}/readability_overall.json")
                readability_docs = [
                    analyze_readability(doc.text) for doc in docs]
                save_file(readability_docs,
                          f"{sub_output_dir}/readability_docs.json")
                published_date = scrape_published_date(html)
                all_url_html_date_tuples.append(
                    (url, html, docs, readability_overall))
                logger.debug(f"Scraped HTML and date for reranked URL: {url}")
                if len(all_url_html_date_tuples) == top_k:
                    break
    logger.info(
        f"Processed {len(all_url_html_date_tuples)} URL-HTML-date tuples")
    return all_url_html_date_tuples


def process_documents(
    url_html_date_tuples: List[Tuple[str, str, List[HeaderDocument], Optional[str]]],
    query: str,
    embed_model: EmbedModelType,
    output_dir: str,
) -> List[HeaderDocument]:
    """Process documents and extract headers, assigning doc_index based on list order."""
    logger.info(f"Processing {len(url_html_date_tuples)} documents")

    all_docs: List[HeaderDocument] = []
    headers = []
    doc_index = 0  # Initialize doc_index counter
    for url, html_str, docs, readability in url_html_date_tuples:
        logger.debug(f"Processing documents for URL: {url}")
        for doc in docs:
            if not doc.metadata["content"].strip():
                continue

            doc.metadata["source_url"] = url
            doc.metadata["readability"] = readability
            # Assign sequential doc_index
            doc.metadata["doc_index"] = doc_index
            headers.append({
                "doc_index": doc_index,
                "source_url": doc["source_url"],
                "parent_header": doc["parent_header"],
                "header": doc["header"],
            })
            all_docs.append(doc)
            doc_index += 1  # Increment doc_index for the next document
    save_file({
        "query": query,
        "count": len(all_docs),
        "source_urls": {doc.metadata["source_url"]: {
            f"h{i}": sum(1 for d in all_docs if d.metadata["source_url"] == doc.metadata["source_url"] and d.metadata["header_level"] == i)
            for i in range(1, 7)
        } for doc in all_docs},
        "headers": {
            f"h{i}": sum(1 for doc in all_docs if doc.metadata["header_level"] == i)
            for i in range(1, 7)
        },
        "documents": all_docs
    }, os.path.join(output_dir, "docs.json"))
    grouped_docs = group_similar_headers(all_docs, model_name=embed_model)
    save_file({"query": query, "results": grouped_docs},
              os.path.join(output_dir, "grouped_docs.json"))
    save_file(headers, os.path.join(output_dir, "headers.json"))
    return all_docs


def search_and_group_documents(
    query: str,
    all_docs: List[HeaderDocument],
    embed_model: EmbedModelType,
    llm_model: LLMModelType,
    output_dir: str,
    chunk_size: int,
    top_k: Optional[int] = None,
    min_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> Tuple[List[Dict], str]:
    """Search documents and group results with source_url at the top of each group, optionally limiting total tokens to max_tokens and filtering contexts with fewer than min_tokens."""
    # Validate that at least one of top_k or max_tokens is provided
    if top_k is None and max_tokens is None:
        raise ValueError(
            "At least one of top_k or max_tokens must be provided")

    logger.info(
        f"Searching {len(all_docs)} documents for query: {query}, top_k={top_k}, max_tokens={max_tokens}, min_tokens={min_tokens}")

    chunked_docs = chunk_headers(
        all_docs, max_tokens=chunk_size, model=embed_model)
    save_file({"query": query, "count": len(chunked_docs), "results": chunked_docs},
              os.path.join(output_dir, "chunked_docs.json"))

    # Filter out header level 1 documents and those without content or below min_tokens
    docs_to_search = [
        doc for doc in chunked_docs
        if doc.metadata["header_level"] != 1
        and doc.metadata.get("content", "").strip()
        and (min_tokens is None or count_tokens(llm_model, [doc.metadata.get("content", "")], prevent_total=True)[0] >= min_tokens)
    ]
    logger.debug(
        f"Filtered to {len(docs_to_search)} documents for search (excluding header level 1, empty content, and below min_tokens)")

    # Perform search
    search_doc_results = search_docs(
        query=query,
        documents=docs_to_search,
        ids=[doc.id_ for doc in docs_to_search],
        model=embed_model,
        top_k=top_k,  # Pass top_k directly to search_docs
    )
    save_file(
        {"query": query, "count": len(
            search_doc_results), "results": search_doc_results},
        os.path.join(output_dir, "search_doc_results.json")
    )
    logger.info(
        f"Saved {len(search_doc_results)} search results to {output_dir}/search_doc_results.json")

    grouped_similar_headers = group_similar_headers(
        search_doc_results, threshold=0.7, model_name=embed_model)

    save_file(
        {"query": query, "count": len(
            grouped_similar_headers), "results": grouped_similar_headers},
        os.path.join(output_dir, "grouped_similar_headers.json")
    )

    # Merge each group by doc["text"], splitting into chunks that do not exceed chunk_size tokens
    # Use max_tokens to limit the total number of tokens across all merged similar headers
    merged_similar_headers: List[HeaderDocumentWithScore] = []
    total_merged_tokens = 0  # Track total tokens across all merged headers

    for group in grouped_similar_headers:
        docs = group.get("documents", [])
        average_score = group.get("average_score", None)
        # Sort docs by doc["score"] in descending order
        docs = sorted(docs, key=lambda d: getattr(d, "score", 0), reverse=True)
        current_chunk_texts = []
        current_chunk_docs = []
        current_chunk_tokens = 0

        def flush_chunk():
            nonlocal total_merged_tokens, current_chunk_tokens
            if current_chunk_texts:
                # Only flush if adding this chunk does not exceed max_tokens (if set)
                if max_tokens is not None and total_merged_tokens + current_chunk_tokens > max_tokens:
                    return False  # Do not flush, would exceed max_tokens
                # Create a HeaderDocument for the node field
                node = HeaderDocument(
                    id_=f"merged_{len(merged_similar_headers)}",
                    text="\n\n".join(current_chunk_texts),
                    metadata={
                        "source_url": current_chunk_docs[0].metadata.get("source_url", "") if current_chunk_docs else "",
                        "parent_header": current_chunk_docs[0].metadata.get("parent_header", "") if current_chunk_docs else "",
                        "header": current_chunk_docs[0].metadata.get("header", "") if current_chunk_docs else "",
                        "doc_index": min(d.metadata.get("doc_index", 0) for d in current_chunk_docs) if current_chunk_docs else 0,
                        "header_level": min(d.metadata.get("header_level", 0) for d in current_chunk_docs) if current_chunk_docs else 0,
                        "content": "\n\n".join(current_chunk_texts),
                    }
                )
                # Create a HeaderDocumentWithScore object
                merged_doc = HeaderDocumentWithScore(
                    node=node,
                    score=average_score,
                    doc_index=min(d.metadata.get("doc_index", 0)
                                  for d in current_chunk_docs) if current_chunk_docs else 0,
                    headers=group.get("headers", []),
                    matches=[],
                    highlighted_text="\n\n".join(current_chunk_texts),
                )
                merged_similar_headers.append(merged_doc)
                total_merged_tokens += current_chunk_tokens
                return True
            return False

        for doc in docs:
            # Try to get doc["text"], fallback to doc["metadata"]["content"]
            doc_text = getattr(doc, "text", None)
            if doc_text is None and hasattr(doc, "metadata"):
                doc_text = doc.metadata.get("content", None)
            if not doc_text:
                continue
            tokens_list = count_tokens(
                llm_model, [doc_text], prevent_total=True)
            tokens = tokens_list[0] if tokens_list else 0

            # Apply min_tokens filter
            if min_tokens is not None and tokens < min_tokens:
                continue

            # If chunk_size is set, flush chunk if adding this doc would exceed chunk_size
            if chunk_size is not None and current_chunk_tokens + tokens > chunk_size:
                flushed = flush_chunk()
                if not flushed and max_tokens is not None and total_merged_tokens >= max_tokens:
                    break  # Stop processing if max_tokens reached
                current_chunk_texts = []
                current_chunk_docs = []
                current_chunk_tokens = 0

            # If max_tokens is set, check if adding this doc would exceed max_tokens
            if max_tokens is not None and total_merged_tokens + current_chunk_tokens + tokens > max_tokens:
                break  # Stop processing this group if max_tokens would be exceeded

            current_chunk_texts.append(doc_text)
            current_chunk_docs.append(doc)
            current_chunk_tokens += tokens

        # Flush any remaining chunk, respecting max_tokens
        flush_chunk()

    save_file(
        {"query": query, "count": len(
            merged_similar_headers), "results": merged_similar_headers},
        os.path.join(output_dir, "merged_similar_headers.json")
    )

    # Apply top_k limit only if specified
    if top_k is not None:
        search_doc_results = merged_similar_headers[:top_k]
    else:
        search_doc_results = merged_similar_headers

    search_doc_results = sample_diverse_headers(search_doc_results)

    # Initialize results and token tracking
    result_texts = [result.text for result in search_doc_results]
    context_tokens: List[int] = count_tokens(
        llm_model, result_texts, prevent_total=True)
    limited_results = search_doc_results
    limited_context_tokens = context_tokens
    total_tokens = sum(context_tokens)

    # Limit results by max_tokens if specified
    if max_tokens is not None:
        total_tokens = 0
        limited_results = []
        limited_context_tokens = []
        for result, tokens in zip(search_doc_results, context_tokens):
            if total_tokens + tokens > max_tokens:
                break
            limited_results.append(result)
            limited_context_tokens.append(tokens)
            total_tokens += tokens

    # Group contexts by source_url
    contexts: List[str] = []
    current_url: str | None = None
    url_to_docs = defaultdict(list)
    for doc in limited_results:
        source_url = doc.metadata["source_url"]
        url_to_docs[source_url].append(doc)

    # Sort each group by doc_index and build contexts
    contexts: List[str] = []
    for source_url in url_to_docs:
        # Sort documents by doc_index
        sorted_docs = sorted(
            url_to_docs[source_url],
            key=lambda x: x.metadata.get("doc_index", 0)
        )
        contexts.append(f"<!-- Source: {source_url} -->")
        contexts.extend(doc.text for doc in sorted_docs)
        logger.debug(
            f"Added source_url header with {len(sorted_docs)} sorted documents: {source_url}")

    # Join contexts with double newlines
    context = "\n\n".join(contexts)
    save_file(context, os.path.join(output_dir, "context.md"))
    logger.debug(f"Generated context with {len(contexts)} segments")

    # Save context metadata (update to reflect sorted order)
    save_file(
        {
            "query": query,
            "total_tokens": total_tokens,
            "count": len(limited_results),
            "urls_info": {
                result.metadata["source_url"]: len(
                    [r for r in limited_results if r.metadata["source_url"] == result.metadata["source_url"]])
                for result in limited_results
            },
            "contexts": [
                {
                    "doc_index": result.metadata.get("doc_index", 0),
                    "score": result.score,
                    "tokens": tokens,
                    "source_url": result.metadata["source_url"],
                    "parent_header": result.metadata["parent_header"],
                    "header": result.metadata["header"],
                    "text": result.text
                }
                for result, tokens in zip(limited_results, limited_context_tokens)
            ]
        },
        os.path.join(output_dir, "contexts.json")
    )
    logger.info(
        f"Saved context with {sum(limited_context_tokens)} tokens to {output_dir}/contexts.json")

    return limited_results, context


def generate_response(
    query: str,
    context: str,
    mlx: MLX,
    output_dir: str
) -> str:
    """Generate and save LLM response."""
    logger.info(
        f"Generating response for query: {query} with model: {mlx.model}")
    PROMPT_TEMPLATE = """\
Context information is below.
---------------------
{context}
---------------------

Given the context information, answer the query.

Query: {query}
"""
    prompt = PROMPT_TEMPLATE.format(query=query, context=context)
    logger.debug("Prompt prepared for LLM")
    response = ""
    for chunk in mlx.stream_chat(
        prompt,
        system_prompt=get_system_date_prompt(),
        temperature=0.7,
        verbose=True,
        max_tokens=10000
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content

    save_file(response, os.path.join(output_dir, "response.md"))
    save_file(
        {"query": query, "context": context, "response": response},
        os.path.join(output_dir, "chat_result.json")
    )
    try:
        logger.success(f"Successfully generated response for query: {query}")
    except AttributeError:
        logger.info(f"Successfully generated response for query: {query}")
    return response


def evaluate_results(
    query: str,
    context: str,
    response: str,
    llm_model: LLMModelType,
    output_dir: str
) -> None:
    """Evaluate context and response relevance."""
    logger.info(f"Evaluating context relevance for query: {query}")
    os.makedirs(os.path.join(output_dir, "eval"), exist_ok=True)
    eval_context_result = evaluate_context_relevancy(query, context, llm_model)
    save_file(
        eval_context_result,
        os.path.join(output_dir, "eval",
                     "evaluate_context_relevance_result.json")
    )
    logger.info(f"Evaluating response relevance for query: {query}")
    eval_response_result = evaluate_response_relevancy(
        query, response, llm_model)
    save_file(
        eval_response_result,
        os.path.join(output_dir, "eval",
                     "evaluate_response_relevance_result.json")
    )
    try:
        logger.success("Evaluation completed successfully")
    except AttributeError:
        logger.info("Evaluation completed successfully")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run semantic search and processing pipeline.")
    p.add_argument("-q", "--query", type=str,
                   default="Top isekai anime 2025.", help="Search query to process")
    p.add_argument("-k", "--top_k", type=int, default=None,
                   help="Number of top documents to consider")
    p.add_argument("-m", "--llm_model", type=str,
                   default="qwen3-1.7b-4bit", help="LLM model to use")
    p.add_argument("-e", "--embed_model", type=str,
                   default="static-retrieval-mrl-en-v1", help="Embedding model to use")
    p.add_argument("-min", "--min_tokens", type=int, default=None,
                   help="Maximum number of tokens for final context")
    p.add_argument("-max", "--max_tokens", type=int, default=2000,
                   help="Maximum number of tokens for final context")
    p.add_argument("-s", "--chunk_size", type=int, default=300,
                   help="Maximum number of tokens per context")
    p.add_argument("-c", "--use_cache", action="store_true",
                   default=True, help="Use cached search results if available")
    p.add_argument("--seed", type=int, default=45,
                   help="Random seed for reproducibility")
    return p.parse_args()


async def main():
    args = parse_args()
    output_dir = initialize_output_directory(__file__, args.query)
    mlx, tokenize = initialize_search_components(
        args.llm_model, args.embed_model, args.seed)

    browser_results = await fetch_search_results(args.query, output_dir, use_cache=args.use_cache)
    url_html_docs = await process_search_results(browser_results, args.query, output_dir, top_k=args.top_k)
    all_docs = process_documents(
        url_html_docs, args.query, args.embed_model, output_dir)
    search_results, context_md = search_and_group_documents(
        query=args.query,
        all_docs=all_docs,
        embed_model=args.embed_model,
        llm_model=args.llm_model,
        output_dir=output_dir,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )
    response = generate_response(
        args.query, context_md, mlx, output_dir)
    evaluate_results(args.query, context_md, response,
                     args.llm_model, output_dir)
    if hasattr(logger, "success"):
        logger.success("Search engine execution completed")
    else:
        logger.info("Search engine execution completed")

if __name__ == "__main__":
    logger.info("Starting search engine script")
    asyncio.run(main())
    logger.info("Search engine script finished")
