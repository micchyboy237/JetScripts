# search_engine.py
from collections import defaultdict
import json
import os
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
from jet.utils.url_utils import rerank_bm25_plus
from jet.wordnet.text_chunker import truncate_texts
from jet.vectors.document_types import HeaderDocument
from jet.vectors.search_with_clustering import search_documents
from jet.wordnet.analyzers.text_analysis import ReadabilityResult, analyze_readability, analyze_text
from jet.code.splitter_markdown_utils import get_md_header_docs
from jet.file.utils import save_file
from jet.llm.mlx.base import MLX
from jet.llm.mlx.token_utils import count_tokens, get_tokenizer_fn
from jet.scrapers.browser.playwright_utils import scrape_multiple_urls
from jet.scrapers.preprocessor import html_to_markdown
from jet.scrapers.utils import scrape_links, scrape_published_date, search_data
from jet.llm.utils.search_docs import SimilarityResult, search_docs
from jet.llm.mlx.tasks.eval.evaluate_context_relevance import evaluate_context_relevance
from jet.llm.mlx.tasks.eval.evaluate_response_relevance import evaluate_response_relevance
from jet.wordnet.words import count_words
from jet.search.searxng import SearchResult


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


def filter_htmls_with_best_combined_mtld(
    url_html_date_tuples: List[Tuple[str, str, Optional[str]]],
    limit: int = 3,
    min_mtld: float = 100.0
) -> List[Tuple[str, str, List[HeaderDocument], ReadabilityResult]]:
    """Filter HTMLs based on MTLD score and header count."""
    logger.info(
        f"Filtering {len(url_html_date_tuples)} HTMLs with min MTLD={min_mtld} and limit={limit}")
    if not url_html_date_tuples or limit <= 0:
        logger.debug("No HTMLs to filter or invalid limit")
        return []

    doc_scores = []
    for url, html, _ in url_html_date_tuples:
        try:
            logger.debug(f"Processing HTML for URL: {url}")
            docs = get_md_header_docs(html, ignore_links=False)
            h2_count = sum(
                1 for doc in docs if doc.metadata['header_level'] == 2)
            logger.debug(f"Found {h2_count} H2 headers for {url}")
            if h2_count < 5:
                logger.debug(
                    f"Skipping {url}: insufficient H2 headers ({h2_count} < 5)")
                continue

            docs_text = "\n\n".join(doc.text for doc in docs)
            readability = analyze_readability(docs_text)
            mtld_score = readability['mtld']
            logger.debug(f"MTLD score for {url}: {mtld_score}")

            if mtld_score >= min_mtld:
                doc_scores.append((url, html, docs, readability, mtld_score))
                logger.debug(
                    f"Added {url} to candidates with MTLD={mtld_score}")
        except (ValueError, KeyError, AttributeError) as e:
            logger.debug(f"Error processing {url}: {str(e)}")
            continue

    doc_scores.sort(key=lambda x: x[4], reverse=True)
    filtered = [(url, html, docs, readability)
                for url, html, docs, readability, _ in doc_scores[:limit]]
    logger.info(f"Filtered to {len(filtered)} HTMLs with highest MTLD scores")
    return filtered


def initialize_output_directory(script_path: str) -> str:
    """Create and return the output directory path."""
    logger.debug(f"Initializing output directory for script: {script_path}")
    script_dir = os.path.dirname(os.path.abspath(script_path))
    output_dir = os.path.join(script_dir, "generated", os.path.splitext(
        os.path.basename(script_path))[0])
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory initialized: {output_dir}")
    return output_dir


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


async def fetch_search_results(query: str, output_dir: str, use_cache: bool = False) -> List[SearchResult]:
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
    browser_search_results: List[Dict],
    query: str,
    output_dir: str
) -> List[Tuple[str, str, Optional[str]]]:
    """Process search results and extract links."""
    logger.info(
        f"Processing {len(browser_search_results)} search results for query: {query}")
    urls = [item["url"] for item in browser_search_results]
    logger.debug(f"Scraping {len(urls)} URLs")
    html_list = await scrape_urls(urls, num_parallel=5)

    all_url_html_date_tuples = []
    all_links = []

    for result, html_str in zip(browser_search_results, html_list):
        url = result["url"]
        if not html_str:
            logger.debug(f"No HTML content for {url}, skipping")
            continue

        if not result.get("publishedDate"):
            published_date = scrape_published_date(html_str)
            result["publishedDate"] = published_date if published_date else None
            logger.debug(f"Scraped published date for {url}: {published_date}")

        links = set(scrape_links(html_str, url))
        links = [link for link in links if (
            link != url if isinstance(link, str) else link["url"] != url)]
        all_links.extend(links)
        logger.debug(f"Extracted {len(links)} links from {url}")

        all_url_html_date_tuples.append(
            (url, html_str, result.get("publishedDate")))

    all_links = list(set(all_links))
    save_file(all_links, os.path.join(output_dir, "links.json"))
    logger.debug(f"Total unique links extracted: {len(all_links)}")
    reranked_links = rerank_bm25_plus(all_links, query, 3)
    logger.debug(f"Reranked to {len(reranked_links)} links")
    save_file(reranked_links, os.path.join(output_dir, "reranked_links.json"))

    logger.info(f"Scraping {len(reranked_links)} reranked links...")
    reranked_html_list = await scrape_urls(reranked_links, num_parallel=5)
    for url, html_str in zip(reranked_links, reranked_html_list):
        if html_str:
            published_date = scrape_published_date(html_str)
            all_url_html_date_tuples.append((url, html_str, published_date))
            logger.debug(f"Scraped HTML and date for reranked URL: {url}")

    logger.info(
        f"Processed {len(all_url_html_date_tuples)} URL-HTML-date tuples")
    return all_url_html_date_tuples


def process_documents(
    url_html_date_tuples: List[Tuple[str, str, Optional[str]]],
    output_dir: str
) -> List[HeaderDocument]:
    """Process documents and extract headers."""
    logger.info(f"Processing {len(url_html_date_tuples)} documents")
    all_url_docs_tuples = filter_htmls_with_best_combined_mtld(
        url_html_date_tuples)
    all_docs = []
    headers = []

    for url, html_str, docs, readability in all_url_docs_tuples:
        logger.debug(f"Processing documents for URL: {url}")
        for doc in docs:
            doc.metadata["source_url"] = url
            doc.metadata["readability"] = readability
            headers.append({**doc.metadata, "text": doc.text})
        all_docs.extend(docs)

    save_file(all_docs, os.path.join(output_dir, "docs.json"))
    save_file(headers, os.path.join(output_dir, "headers.json"))
    return all_docs


def map_search_results_to_docs(search_results: List[SimilarityResult], docs: List[HeaderDocument]) -> List[HeaderDocument]:
    """Map search results back to original documents.

    Args:
        search_results: List of search results with document IDs
        docs: Original list of documents to map back to

    Returns:
        List of documents that matched the search results
    """
    result_ids = {result["id"] for result in search_results}
    return [doc for doc in docs if doc.id_ in result_ids]


def search_and_group_documents(
    query: str,
    all_docs: List[HeaderDocument],
    embed_model: str,
    llm_model: str,
    top_k: int,
    output_dir: str
) -> Tuple[List[Dict], str]:
    """Search documents and group results."""
    logger.info(
        f"Searching {len(all_docs)} documents for query: {query}, top_k={top_k}")
    docs_to_search = [
        doc for doc in all_docs if doc.metadata["header_level"] != 1]
    parent_headers = [
        doc.metadata["parent_header"].lstrip('#').strip() for doc in docs_to_search
    ]
    headers = [
        doc.metadata["header"].lstrip('#').strip() for doc in docs_to_search
    ]
    contents = [
        doc.metadata["content"].lstrip('#').strip() for doc in docs_to_search
    ]
    logger.debug(
        f"Filtered to {len(docs_to_search)} documents for search (excluding header level 1)")

    # Search through parent headers
    parent_headers_search_doc_results = search_docs(
        query=query,
        documents=parent_headers,
        ids=[doc.id_ for doc in docs_to_search],
        model=embed_model,
        top_k=40,
    )
    mapped_parent_headers_search_docs = map_search_results_to_docs(
        parent_headers_search_doc_results, docs_to_search)

    # Search through headers
    headers_search_doc_results = search_docs(
        query=query,
        documents=headers,
        ids=[doc.id_ for doc in mapped_parent_headers_search_docs],
        model=embed_model,
        top_k=20,
    )
    mapped_headers_search_docs = map_search_results_to_docs(
        headers_search_doc_results, mapped_parent_headers_search_docs)

    # Final search through document contents
    search_doc_results = search_docs(
        query=query,
        documents=[doc.text for doc in mapped_headers_search_docs],
        ids=[doc.id_ for doc in mapped_headers_search_docs],
        model=embed_model,
        top_k=10,
    )

    save_file(
        {"query": query, "count": len(
            search_doc_results), "results": search_doc_results},
        os.path.join(output_dir, "search_doc_results.json")
    )
    logger.info(
        f"Saved {len(search_doc_results)} search results to {output_dir}/search_doc_results.json")

    search_result_dict = {result["id"]: result for result in search_doc_results}
    sorted_doc_results = []
    for doc in all_docs:
        if doc.metadata["header_level"] != 1 and count_words(doc.text) >= 10:
            sorted_doc_results.append({
                **doc.metadata,
                "text": doc.text,
                "is_top": doc.id_ in search_result_dict
            })

    grouped_by_source_and_parent = defaultdict(list)
    for result in sorted_doc_results:
        key = (result["source_url"], result.get(
            "parent_header", ""), result["is_top"])
        grouped_by_source_and_parent[key].append(result)

    contexts = [doc["text"]
                for doc in sorted_doc_results if doc["is_top"]][:top_k]
    context = "\n\n".join(contexts)
    save_file(context, os.path.join(output_dir, "context.md"))
    logger.debug(f"Generated context with {len(contexts)} segments")

    context_tokens = count_tokens(llm_model, context, prevent_total=True)
    save_file({
        "total_tokens": context_tokens,
        "contexts": contexts
    }, os.path.join(output_dir, "contexts.json"))
    logger.info(
        f"Saved context with {context_tokens} tokens to {output_dir}/contexts.json")
    return sorted_doc_results, context


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

    save_file(
        {"query": query, "context": context, "response": response},
        os.path.join(output_dir, "chat_response.json")
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
    query = "List all ongoing and upcoming isekai anime 2025."
    top_k = 10
    embed_model = "all-MiniLM-L12-v2"
    llm_model = "llama-3.2-3b-instruct-4bit"
    seed = 45
    use_cache = False

    logger.info(f"Starting search engine with query: {query}")
    output_dir = initialize_output_directory(__file__)
    mlx, _ = initialize_search_components(llm_model, embed_model, seed)
    # query = rewrite_query(query, llm_model)
    browser_search_results = await fetch_search_results(query, output_dir, use_cache=use_cache)
    url_html_date_tuples = await process_search_results(browser_search_results, query, output_dir)
    url_html_date_tuples.sort(key=lambda x: x[2] or "", reverse=True)
    all_docs = process_documents(url_html_date_tuples, output_dir)
    sorted_doc_results, context = search_and_group_documents(
        query, all_docs, embed_model, llm_model, top_k, output_dir)
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
