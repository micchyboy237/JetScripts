# search_engine.py
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

from jet.features.nltk_search import get_pos_tag, search_by_pos
from jet.llm.mlx.helpers.base import get_system_date_prompt
from jet.llm.mlx.mlx_types import EmbedModelType, LLMModelType
from jet.logger import logger
from jet.scrapers.hrequests_utils import scrape_urls
from jet.transformers.link_formatters import LinkFormatter, format_links_for_embedding
from jet.utils.url_utils import rerank_urls_bm25_plus
from jet.wordnet.text_chunker import truncate_texts
from jet.vectors.document_types import HeaderDocument, HeaderDocumentWithScore
from jet.vectors.search_with_clustering import search_documents
from jet.wordnet.analyzers.text_analysis import ReadabilityResult, analyze_readability, analyze_text
from jet.code.splitter_markdown_utils import get_md_header_docs, get_header_level
from jet.file.utils import save_file
from jet.llm.mlx.base import MLX
from jet.models.tokenizer.base import count_tokens, get_tokenizer_fn
from jet.scrapers.browser.playwright_utils import scrape_multiple_urls
from jet.scrapers.preprocessor import html_to_markdown
from jet.scrapers.utils import scrape_links, scrape_published_date, search_data
# from jet.llm.utils.search_docs import search_docs
from jet.models.tasks.hybrid_search_docs_with_bm25 import search_docs
from jet.llm.mlx.tasks.eval.evaluate_context_relevance import evaluate_context_relevance
from jet.llm.mlx.tasks.eval.evaluate_response_relevance import evaluate_response_relevance
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


# def filter_htmls_with_best_combined_mtld(
#     url_html_date_tuples: List[Tuple[str, str, Optional[str]]],
#     limit: Optional[int] = None,
#     min_mtld: float = 100.0
# ) -> List[Tuple[str, str, List[HeaderDocument], ReadabilityResult]]:
#     """Filter HTMLs based on MTLD score and header count."""
#     logger.info(
#         f"Filtering {len(url_html_date_tuples)} HTMLs with min MTLD={min_mtld} and limit={limit}")
#     if not url_html_date_tuples:
#         logger.debug("No HTMLs to filter")
#         return []

#     doc_scores = []
#     for url, html, _ in url_html_date_tuples:
#         try:
#             logger.debug(f"Processing HTML for URL: {url}")
#             docs = get_md_header_docs(html, ignore_links=True)
#             header_count = len(docs)
#             logger.debug(f"Found {header_count} headers for {url}")
#             if header_count == 0:
#                 logger.warning(
#                     f"Skipping {url}: no headers found")
#                 continue

#             docs_text = "\n\n".join(doc.text for doc in docs)
#             readability = analyze_readability(docs_text)
#             mtld_score = readability['mtld']
#             logger.debug(f"MTLD score for {url}: {mtld_score}")

#             if header_count > 5 or mtld_score >= min_mtld:
#                 doc_scores.append((url, html, docs, readability, mtld_score))
#                 logger.debug(
#                     f"Added {url} to candidates with MTLD={mtld_score}")
#         except (ValueError, KeyError, AttributeError) as e:
#             logger.debug(f"Error processing {url}: {str(e)}")
#             continue

#     doc_scores.sort(key=lambda x: x[4], reverse=True)
#     filtered = [(url, html, docs, readability)
#                 for url, html, docs, readability, _ in doc_scores[:limit]]
#     logger.info(f"Filtered to {len(filtered)} HTMLs with highest MTLD scores")
#     return filtered


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
    mlx = MLX(llm_model, seed=seed)
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
    top_k: int = 10
) -> List[Tuple[str, str, Optional[str]]]:
    """Process search results and extract links, ensuring top 5 URLs are always included."""
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

    # Store URL-to-HTML mapping
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
            sub_output_dir = os.path.join(output_dir, sub_url_dir)
            save_file(html, f"{sub_output_dir}/page.html")
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

            # Get published date and links
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
    url_html_date_tuples: List[Tuple[str, str, Optional[str]]],
    query: str,
    output_dir: str
) -> List[HeaderDocument]:
    """Process documents and extract headers."""
    logger.info(f"Processing {len(url_html_date_tuples)} documents")
    # all_url_docs_tuples = filter_htmls_with_best_combined_mtld(
    #     url_html_date_tuples)
    all_docs = []
    headers = []

    for url, html_str, docs, readability in url_html_date_tuples:
        logger.debug(f"Processing documents for URL: {url}")
        for doc in docs:
            doc.metadata["source_url"] = url
            doc.metadata["readability"] = readability
            headers.append({
                "doc_index": doc["doc_index"],
                "source_url": doc["source_url"],
                "parent_header": doc["parent_header"],
                "header": doc["header"],
            })
        all_docs.extend(docs)

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
    save_file(headers, os.path.join(output_dir, "headers.json"))
    return all_docs


def search_and_group_documents(
    query: str,
    all_docs: List[HeaderDocument],
    embed_model: str,
    llm_model: str,
    output_dir: str,
    top_k: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> Tuple[List[Dict], str, List[Dict]]:
    logger.info(
        f"Searching {len(all_docs)} documents for query: {query}, top_k={top_k}, max_tokens={max_tokens}")
    search_doc_results = search_docs(
        query=query,
        documents=all_docs,
        ids=[doc.id_ for doc in all_docs],
        model=embed_model,
        top_k=top_k,
    )
    save_file(
        {"query": query, "count": len(
            search_doc_results), "results": search_doc_results},
        os.path.join(output_dir, "search_doc_results.json")
    )
    logger.info(
        f"Saved {len(search_doc_results)} search results to {output_dir}/search_doc_results.json")
    parent_groups: dict[str, List[HeaderDocumentWithScore]] = defaultdict(list)
    doc_id_to_result = {result["id"]: result for result in search_doc_results}
    for doc in all_docs:
        if doc.id_ in doc_id_to_result:
            parent_header = doc.metadata.get("parent_header", "")
            parent_groups[parent_header].append(doc_id_to_result[doc.id_])
    parent_scores = []
    for parent_header, docs in parent_groups.items():
        if not docs:
            logger.debug(f"No documents found for parent_header: {parent_header}, skipping")
            continue
        parent_doc = next(
            (d for d in all_docs if d.metadata["header"] == parent_header), None)
        parent_score = doc_id_to_result.get(parent_doc.id_, {}).get("score", 0.0) if parent_doc else 0.0
        num_children = len(docs)
        avg_child_score = sum(doc["score"] for doc in docs) / num_children if num_children > 0 else 0.0
        combined_score = parent_score + (num_children * 0.2) + (avg_child_score * 0.5)
        child_headers = [{
            "doc_index": doc["metadata"]["doc_index"],
            "score": doc["score"],
            "text": doc["text"],
        } for doc in docs]
        parent_scores.append({
            "parent_header": parent_header,
            "source_url": docs[0]["metadata"]["source_url"] if docs else "",
            "parent_score": parent_score,
            "num_children": num_children,
            "avg_child_score": avg_child_score,
            "combined_score": combined_score,
            "child_headers": child_headers
        })
    sorted_parent_headers = sorted(
        parent_scores, key=lambda x: x["combined_score"], reverse=True)
    save_file(
        {"query": query, "count": len(
            sorted_parent_headers), "sorted_parent_headers": sorted_parent_headers},
        os.path.join(output_dir, "sorted_parent_headers.json")
    )
    logger.info(
        f"Saved {len(sorted_parent_headers)} sorted parent headers to {output_dir}/sorted_parent_headers.json")
    contexts: List[str] = []
    texts_to_tokenize: List[str] = []
    current_url: str | None = None
    total_tokens = 0
    selected_docs = []
    for parent in sorted_parent_headers:
        parent_header = parent["parent_header"]
        source_url = parent["source_url"]
        if source_url != current_url:
            source_line = f"<!-- Source: {source_url} -->"
            texts_to_tokenize.append(source_line)
            contexts.append(source_line)
            current_url = source_url
            logger.debug(f"Added source_url header: {source_url}")
        parent_doc = next(
            (d for d in all_docs if d.metadata["header"] == parent_header and d.metadata["header_level"] == 1), None)
        if parent_doc:
            parent_text = parent_doc.get_recursive_text()
            texts_to_tokenize.append(parent_text)
            contexts.append(parent_text)
            selected_docs.append({
                "id": parent_doc.id_,
                "metadata": parent_doc.metadata,
                "text": parent_text,
                "score": parent["parent_score"],
                "doc_index": parent_doc.metadata["doc_index"]
            })
        for child in parent["child_headers"]:
            child_text = child["text"]
            child_doc_result = next(
                (result for result in doc_id_to_result.values()
                 if result["metadata"]["doc_index"] == child["doc_index"]),
                None
            )
            if not child_doc_result:
                logger.warning(
                    f"No document found for doc_index: {child['doc_index']}")
                continue
            texts_to_tokenize.append(child_text)
            contexts.append(child_text)
            selected_docs.append({
                "id": child_doc_result["id"],
                "metadata": child_doc_result["metadata"],
                "text": child_text,
                "score": child["score"],
                "doc_index": child["doc_index"]
            })
    context_tokens = count_tokens(
        llm_model, texts_to_tokenize, prevent_total=True)
    total_tokens = sum(context_tokens)
    if max_tokens is not None:
        valid_indices = []
        running_tokens = 0
        for i, tokens in enumerate(context_tokens):
            if running_tokens + tokens > max_tokens:
                break
            running_tokens += tokens
            valid_indices.append(i)
        contexts = [contexts[i] for i in valid_indices]
        selected_docs = [selected_docs[i] for i in valid_indices]
        context_tokens = [context_tokens[i] for i in valid_indices]
        total_tokens = sum(context_tokens)
    context = "\n\n".join(contexts)
    save_file(context, os.path.join(output_dir, "context.md"))
    logger.debug(
        f"Generated context with {len(contexts)} segments, {total_tokens} tokens")
    save_file(
        {
            "total_tokens": total_tokens,
            "count": len(selected_docs),
            "urls_info": {
                doc["metadata"]["source_url"]: len(
                    [d for d in selected_docs if d["metadata"]
                        ["source_url"] == doc["metadata"]["source_url"]]
                )
                for doc in selected_docs
            },
            "contexts": [
                {
                    "doc_index": doc["doc_index"],
                    "score": doc["score"],
                    "tokens": tokens,
                    "source_url": doc["metadata"]["source_url"],
                    "parent_header": doc["metadata"]["parent_header"],
                    "header": doc["metadata"]["header"],
                    "text": doc["text"]
                }
                for doc, tokens in zip(selected_docs, context_tokens)
            ]
        },
        os.path.join(output_dir, "contexts.json")
    )
    logger.info(
        f"Saved context with {total_tokens} tokens to {output_dir}/contexts.json")
    return search_doc_results, context, sorted_parent_headers
    

def generate_response(
    query: str,
    context: str,
    llm_model: str,
    mlx: MLX,
    output_dir: str
) -> str:
    """Generate and save LLM response."""
    logger.info(
        f"Generating response for query: {query} with model: {llm_model}")
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
    llm_model: str,
    output_dir: str
) -> None:
    """Evaluate context and response relevance."""
    logger.info(f"Evaluating context relevance for query: {query}")
    os.makedirs(os.path.join(output_dir, "eval"), exist_ok=True)
    eval_context_result = evaluate_context_relevance(query, context, llm_model)
    save_file(
        eval_context_result,
        os.path.join(output_dir, "eval",
                     "evaluate_context_relevance_result.json")
    )
    logger.info(f"Evaluating response relevance for query: {query}")
    eval_response_result = evaluate_response_relevance(
        query, context, response, llm_model)
    save_file(
        eval_response_result,
        os.path.join(output_dir, "eval",
                     "evaluate_response_relevance_result.json")
    )
    try:
        logger.success("Evaluation completed successfully")
    except AttributeError:
        logger.info("Evaluation completed successfully")


async def main():
    """Main function to orchestrate the search and response generation."""
    query = "Top isekai anime 2025."
    top_k = None
    max_tokens = 2000
    embed_model = "static-retrieval-mrl-en-v1"
    llm_model = "llama-3.2-1b-instruct-4bit"
    seed = 45
    use_cache = False

    logger.info(f"Starting search engine with query: {query}")
    output_dir = initialize_output_directory(__file__, query)
    mlx, _ = initialize_search_components(llm_model, embed_model, seed)
    # query = rewrite_query(query, llm_model)
    browser_search_results = await fetch_search_results(query, output_dir, use_cache=use_cache)
    url_html_date_tuples = await process_search_results(browser_search_results, query, output_dir)
    # url_html_date_tuples.sort(key=lambda x: x[2] or "", reverse=True)
    all_docs = process_documents(url_html_date_tuples, query, output_dir)
    sorted_doc_results, context, sorted_parent_headers = search_and_group_documents(
        query, all_docs, embed_model, llm_model, output_dir, top_k=top_k, max_tokens=max_tokens)
    response = generate_response(query, context, llm_model, mlx, output_dir)
    evaluate_results(query, context, response, llm_model, output_dir)
    try:
        logger.success("Search engine execution completed")
    except AttributeError:
        logger.info("Search engine execution completed")


if __name__ == "__main__":
    logger.info("Starting search engine script")
    asyncio.run(main())
    logger.info("Search engine script finished")
