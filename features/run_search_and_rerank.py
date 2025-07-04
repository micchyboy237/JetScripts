import argparse
from collections import defaultdict
import json
import math
import os
import re
import shutil
import string
from typing import Dict, List, Optional, Tuple, TypedDict
from datetime import datetime
import asyncio
from urllib.parse import unquote, urlparse
from jet.code.html_utils import preprocess_html
from jet.code.markdown_utils import analyze_markdown, parse_markdown
from jet.data.header_docs import HeaderDocs
from jet.data.header_types import NodeWithScore
from jet.data.header_utils._prepare_for_rag import prepare_for_rag
from jet.data.header_utils._search_headers import search_headers
from jet.data.sample_diverse_headers import sample_diverse_headers
# from jet.scrapers.preprocessor import convert_html_to_markdown
from jet.code.markdown_utils import convert_html_to_markdown
from jet.features.nltk_search import get_pos_tag, search_by_pos
from jet.llm.mlx.helpers.base import get_system_date_prompt
from jet.models.embeddings.chunking import chunk_headers_by_hierarchy
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.logger import logger
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.scrapers.hrequests_utils import scrape_urls
from jet.transformers.link_formatters import LinkFormatter, format_links_for_embedding
from jet.utils.url_utils import rerank_urls_bm25_plus
from jet.vectors.document_types import HeaderDocument, HeaderDocumentWithScore
from jet.vectors.document_utils import get_leaf_documents
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
from jet.wordnet.similarity import group_similar_headers
from jet.wordnet.text_chunker import chunk_headers
from jet.wordnet.words import count_words
from jet.search.searxng import SearchResult as BrowserSearchResult
from jet.llm.rag.mlx.classification import MLXRAGClassifier
import time
import numpy as np

PROMPT_TEMPLATE = """\
Context information is below.
---------------------
{context}
---------------------

Given the context information, answer the query.

Query: {query}
"""


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
    """
    Format a URL into a lowercase directory name, replacing all punctuation with underscores.
    """
    clean_url = re.sub(r'^(https?://|www\.)|(\?.*)', '', url)
    trans_table = str.maketrans({p: '_' for p in string.punctuation})
    formatted = clean_url.translate(trans_table).lower()
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


async def process_url_content(
    url: str,
    html: str,
    query: str,
    output_dir: str,
    from_reranked_link: bool,
    url_to_result: Optional[Dict[str, Dict]] = None
) -> Optional[Tuple[str, str, List[HeaderDocument], Dict]]:
    docs = get_md_header_docs(html, ignore_links=True,
                              metadata={"source_url": url})
    if len(docs) == 0:
        logger.debug(f"No headers found for {url}, skipping")
        return None
    # # Filter out docs with readability["mtld_category"] == "very_low"
    # filtered_docs = []
    # for doc in docs:
    #     readability = analyze_readability(doc.text)
    #     mtld_category = readability.get("mtld_category")
    #     if mtld_category not in ["very_low", "low"]:
    #         filtered_docs.append(doc)
    # docs = filtered_docs
    # if len(docs) == 0:
    #     logger.debug(
    #         f"All docs for {url} filtered out due to mtld_category == 'very_low'")
    #     return None

    sub_url_dir = format_sub_url_dir(url)
    sub_output_dir = os.path.join(output_dir, "pages", sub_url_dir)
    os.makedirs(sub_output_dir, exist_ok=True)
    save_file(html, f"{sub_output_dir}/page.html")
    md_content = convert_html_to_markdown(html)
    save_file(md_content, f"{sub_output_dir}/md_content.md")
    # md_content_markdownify = convert_html_to_markdownify(html)
    # save_file(md_content_markdownify,
    #           f"{sub_output_dir}/md_content_markdownify.md")
    markdown_parsed = parse_markdown(md_content)
    save_file(markdown_parsed, f"{sub_output_dir}/markdown_parsed.json")
    markdown_analysis = analyze_markdown(md_content)
    save_file(markdown_analysis, f"{sub_output_dir}/markdown_analysis.json")
    save_file(
        {
            "query": query,
            "from_reranked_link": from_reranked_link,
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

    text_stats_all = analyze_text(docs_text)
    save_file(text_stats_all,
              f"{sub_output_dir}/text_stats_all.json")
    text_stats_docs = [
        {
            "id": getattr(doc, "id", None),
            "doc_index": idx,
            "text": doc.text,
            "text_stats": analyze_text(doc.text)
        }
        for idx, doc in enumerate(docs)
    ]
    # Sort by text_stats["mtld"] in descending order
    text_stats_docs_sorted = sorted(
        text_stats_docs,
        key=lambda x: x["text_stats"].get("mtld", float('-inf')),
        reverse=True
    )
    save_file(text_stats_docs_sorted, f"{sub_output_dir}/text_stats_docs.json")

    readability_overall = analyze_readability(docs_text)
    save_file(readability_overall,
              f"{sub_output_dir}/readability_overall.json")
    readability_docs = [
        {
            "id": getattr(doc, "id", None),
            "doc_index": idx,
            "text": doc.text,
            "readability": analyze_readability(doc.text)
        }
        for idx, doc in enumerate(docs)
    ]
    # Sort by readability["mtld"] in descending order
    readability_docs_sorted = sorted(
        readability_docs,
        key=lambda x: x["readability"].get("mtld", float('-inf')),
        reverse=True
    )
    save_file(readability_docs_sorted,
              f"{sub_output_dir}/readability_docs.json")

    if url_to_result and url in url_to_result:
        result = url_to_result[url]
        if not result.get("publishedDate"):
            published_date = scrape_published_date(html)
            result["publishedDate"] = published_date if published_date else None
            logger.debug(f"Scraped published date for {url}: {published_date}")
    return url, html, docs, readability_overall


# async def process_search_results(
#     browser_search_results: List[BrowserSearchResult],
#     query: str,
#     output_dir: str,
#     top_n: int = 10
# ) -> List[Tuple[str, str, Optional[str]]]:
#     """Process search results and extract links from the top N URLs."""
#     logger.info(
#         f"Processing {len(browser_search_results)} search results for query: {query}")
#     selected_urls = [item["url"] for item in browser_search_results[:top_n]]
#     logger.debug(f"Selected {len(selected_urls)} URLs: {selected_urls}")
#     url_to_result = {r["url"]: r for r in browser_search_results}
#     all_url_html_date_tuples = []
#     all_links = []
#     async for url, status, html in scrape_urls(selected_urls, num_parallel=top_n, limit=top_n, show_progress=True):
#         if status == "completed" and html:
#             result = await process_url_content(
#                 url=url,
#                 html=html,
#                 query=query,
#                 output_dir=output_dir,
#                 from_reranked_link=False,
#                 url_to_result=url_to_result
#             )
#             if result:
#                 all_url_html_date_tuples.append(result)
#                 links = set(scrape_links(html, url))
#                 links = [link for link in links if (
#                     link != url if isinstance(link, str) else link["url"] != url)]
#                 all_links.extend(links)
#                 logger.debug(f"Extracted {len(links)} links from {url}")
#     all_links = list(set(all_links))
#     all_links = [link for link in all_links if (
#         link not in selected_urls if isinstance(link, str) else link["url"] not in selected_urls)]
#     save_file(all_links, os.path.join(output_dir, "links.json"))
#     logger.debug(f"Total unique links extracted: {len(all_links)}")
#     logger.info(
#         f"Processed {len(all_url_html_date_tuples)} URL-HTML-date tuples")
#     return all_url_html_date_tuples


async def process_search_results(
    browser_search_results: List[BrowserSearchResult],
    query: str,
    output_dir: str,
    top_n: int = 10,
    embed_model: EmbedModelType = "all-MiniLM-L6-v2",
    chunk_size: int = 200,
    chunk_overlap: int = 40,
    max_length: int = 2000,
) -> List[NodeWithScore]:
    """Process search results and extract links from the top N URLs."""
    logger.info(
        f"Processing {len(browser_search_results)} search results for query: {query}")
    selected_urls = [item["url"] for item in browser_search_results[:top_n]]
    logger.debug(f"Selected {len(selected_urls)} URLs: {selected_urls}")

    SentenceTransformerRegistry.load_model(embed_model)
    tokenizer = SentenceTransformerRegistry.get_tokenizer()
    url_to_result = {r["url"]: r for r in browser_search_results}
    all_url_html_date_tuples = []
    all_links = []
    all_search_results: List[NodeWithScore] = []
    async for url, status, html in scrape_urls(selected_urls, num_parallel=top_n, limit=top_n, show_progress=True):
        if status == "completed" and html:
            # result = await process_url_content(
            #     url=url,
            #     html=html,
            #     query=query,
            #     output_dir=output_dir,
            #     from_reranked_link=False,
            #     url_to_result=url_to_result
            # )
            # if result:
            #     all_url_html_date_tuples.append(result)
            #     links = set(scrape_links(html, url))
            #     links = [link for link in links if (
            #         link != url if isinstance(link, str) else link["url"] != url)]
            #     all_links.extend(links)
            #     logger.debug(f"Extracted {len(links)} links from {url}")

            # Extract links from html
            links = set(scrape_links(html, url))
            links = [link for link in links if (
                link != url if isinstance(link, str) else link["url"] != url)]
            all_links.extend(links)
            logger.debug(f"Extracted {len(links)} links from {url}")

            # RAG Search
            sub_url_dir = format_sub_url_dir(url)
            sub_output_dir = os.path.join(output_dir, "pages", sub_url_dir)

            html = preprocess_html(html)
            save_file(html, f"{sub_output_dir}/page.html")

            doc_markdown_tokens = parse_markdown(html, ignore_links=True)
            doc_markdown = "\n\n".join([item["content"]
                                        for item in doc_markdown_tokens])
            save_file(doc_markdown, f"{output_dir}/doc_markdown.md")

            chunked_docs = chunk_headers_by_hierarchy(
                doc_markdown, chunk_size, tokenizer)
            save_file(chunked_docs, f"{output_dir}/chunked_docs.json")

            md_content = convert_html_to_markdown(html)
            save_file(md_content, f"{sub_output_dir}/md_content.md")

            parsed_md = parse_markdown(md_content)
            save_file(parsed_md, f"{sub_output_dir}/parsed_md.json")

            analysis = analyze_markdown(md_content)
            save_file(analysis, f"{sub_output_dir}/analysis.json")

            tokens = parse_markdown(md_content, ignore_links=True)
            save_file(tokens, f"{sub_output_dir}/markdown_tokens.json")

            header_docs = HeaderDocs.from_tokens(tokens)
            save_file(header_docs, f"{sub_output_dir}/header_docs.json")

            header_docs.calculate_num_tokens(embed_model)
            all_nodes = header_docs.as_nodes()
            save_file(all_nodes, f"{sub_output_dir}/all_nodes.json")

            vector_store = prepare_for_rag(
                all_nodes, model=embed_model, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            save_file(vector_store.get_nodes(),
                      f"{sub_output_dir}/prepared_nodes.json")

            search_results = search_headers(
                query, vector_store, model=embed_model, top_k=None, threshold=0.7)
            for result in search_results:
                result.metadata.update({
                    "source_url": url
                })
            save_file(search_results, f"{sub_output_dir}/search_results.json")
            all_search_results.extend(search_results)

    all_links = list(set(all_links))
    all_links = [link for link in all_links if (
        link not in selected_urls if isinstance(link, str) else link["url"] not in selected_urls)]
    save_file(all_links, os.path.join(output_dir, "links.json"))
    logger.debug(f"Total unique links extracted: {len(all_links)}")
    logger.info(
        f"Processed {len(all_url_html_date_tuples)} URL-HTML-date tuples")

    save_file(all_search_results, os.path.join(
        output_dir, "all_search_results.json"))

    sorted_search_results = sorted(
        all_search_results, key=lambda x: x.score, reverse=True)
    save_file(sorted_search_results, os.path.join(
        output_dir, "sorted_search_results.json"))

    # INSERT_YOUR_CODE
    # Filter all_search_results by num_tokens up to max_length
    filtered_search_results = []
    total_tokens = 0

    for result in sorted_search_results:
        num_tokens = getattr(result, "num_tokens", None)
        if num_tokens is None:
            # Try to get from metadata if not present directly
            num_tokens = result.num_tokens
        if total_tokens + num_tokens > max_length:
            break
        filtered_search_results.append(result)
        total_tokens += num_tokens

    return filtered_search_results


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
    doc_index = 0
    for url, html_str, docs, readability in url_html_date_tuples:
        logger.debug(f"Processing documents for URL: {url}")
        for doc in docs:
            if not doc.metadata["content"].strip():
                continue
            doc.metadata["source_url"] = url
            doc.metadata["readability"] = readability
            doc.metadata["doc_index"] = doc_index
            headers.append({
                "doc_index": doc_index,
                "source_url": doc["source_url"],
                "parent_header": doc["parent_header"],
                "header": doc["header"],
            })
            all_docs.append(doc)
            doc_index += 1
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
    if top_k is None and max_tokens is None:
        raise ValueError(
            "At least one of top_k or max_tokens must be provided")
    logger.info(
        f"Searching {len(all_docs)} documents for query: {query}, top_k={top_k}, max_tokens={max_tokens}, min_tokens={min_tokens}")

    # Filter documents
    docs = get_leaf_documents(all_docs)
    docs_to_search = [doc for doc in docs if doc.metadata["content"].strip()]
    logger.debug(
        f"Filtered to {len(docs_to_search)} documents for search (excluding empty content)")

    # Search documents
    search_results = search_docs(
        query=query,
        documents=docs_to_search,
        ids=[doc.id_ for doc in docs_to_search],
        model=embed_model,
        top_k=None,
        threshold=0.5
    )
    logger.debug(f"Found {len(search_results)} search results")

    # Count tokens
    result_texts = [result["text"] for result in search_results]
    context_tokens = count_tokens(llm_model, result_texts, prevent_total=True)
    total_tokens = sum(context_tokens)

    # Apply token filters
    limited_results = search_results
    limited_context_tokens = context_tokens
    if min_tokens is not None:
        limited_results = [
            result for result, tokens in zip(search_results, context_tokens)
            if tokens >= min_tokens
        ]
        limited_context_tokens = [
            tokens for tokens in context_tokens
            if tokens >= min_tokens
        ]
        total_tokens = sum(limited_context_tokens)
    if max_tokens is not None:
        current_tokens = 0
        temp_results = []
        temp_tokens = []
        for result, tokens in zip(limited_results, limited_context_tokens):
            if current_tokens + tokens > max_tokens:
                break
            temp_results.append(result)
            temp_tokens.append(tokens)
            current_tokens += tokens
        limited_results = temp_results
        limited_context_tokens = temp_tokens
        total_tokens = sum(limited_context_tokens)
    if top_k is not None:
        limited_results = limited_results[:top_k]
        limited_context_tokens = limited_context_tokens[:top_k]
        total_tokens = sum(limited_context_tokens)

    # Group by source URL
    contexts: List[str] = []
    current_url = None
    for result in limited_results:
        source_url = result["metadata"]["source_url"]
        if source_url != current_url:
            contexts.append(f"<!-- Source: {source_url} -->")
            current_url = source_url
        contexts.append(result["text"])
    context = "\n".join(contexts)
    save_file(context, os.path.join(output_dir, "context.md"))
    logger.debug(f"Generated context with {len(contexts)} segments")

    # Save results
    output_path = os.path.join(output_dir, "contexts.json")
    save_file(
        {
            "query": query,
            "total_tokens": total_tokens,
            "count": len(limited_results),
            "urls_info": {
                result["metadata"]["source_url"]: len(
                    [r for r in limited_results if r["metadata"]
                        ["source_url"] == result["metadata"]["source_url"]]
                )
                for result in limited_results
            },
            "contexts": [
                {
                    "rank": result["rank"],
                    "doc_index": result["doc_index"],
                    "chunk_index": result["chunk_index"],
                    "score": result["score"],
                    "tokens": tokens,
                    "source_url": result["metadata"]["source_url"],
                    "parent_header": result["metadata"]["parent_header"],
                    "header": result["metadata"]["header"],
                    "text": result["text"]
                }
                for result, tokens in zip(limited_results, limited_context_tokens)
            ]
        },
        output_path
    )
    logger.info(f"Saved context with {total_tokens} tokens to {output_path}")

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


def group_search_results_by_source_url_for_context(search_results: List[NodeWithScore]) -> str:
    """
    Group search results by their source URL and format as a context string.

    Args:
        search_results (List[NodeWithScore]): List of search result nodes, each with metadata.

    Returns:
        str: Markdown-formatted string grouping content by source URL.
    """
    from collections import defaultdict

    grouped = defaultdict(list)
    for node in search_results:
        url = node.metadata.get("source_url", "Unknown Source")
        grouped[url].append(node)

    context_blocks = []
    for url, nodes in grouped.items():
        block = f"<!-- Source: {url} -->\n\n"
        nodes = sorted(nodes, key=lambda n: (
            getattr(n, "doc_index", 0), getattr(n, "chunk_index", 0)))
        for i, node in enumerate(nodes, 1):
            # Try to get a title or header if available
            block += node.get_text() + "\n\n"
        context_blocks.append(block.strip())

    return "\n\n".join(context_blocks)


async def main():
    args = parse_args()
    output_dir = initialize_output_directory(__file__, args.query)
    mlx, tokenize = initialize_search_components(
        args.llm_model, args.embed_model, args.seed)
    save_file(args.query, os.path.join(output_dir, "query.md"))
    browser_results = await fetch_search_results(args.query, output_dir, use_cache=args.use_cache)
    # url_html_docs = await process_search_results(browser_results, args.query, output_dir)
    search_results = await process_search_results(browser_results, args.query, output_dir, max_length=1500)
    save_file(search_results, os.path.join(output_dir, "contexts.json"))
    context_md = group_search_results_by_source_url_for_context(search_results)
    save_file(context_md, os.path.join(output_dir, "context.md"))
    save_file({
        "context_tokens": count_tokens(args.llm_model, context_md),
    }, os.path.join(output_dir, "context_info.json"))
    parsed_md = parse_markdown(context_md)
    save_file(parsed_md, f"{output_dir}/parsed_md.json")
    analysis = analyze_markdown(context_md)
    save_file(analysis, f"{output_dir}/analysis.json")

    # all_docs = process_documznts(
    #     url_html_docs, args.query, args.embed_model, output_dir)
    # search_results, context_md = search_and_group_documents(
    #     query=args.query,
    #     all_docs=all_docs,
    #     embed_model=args.embed_model,
    #     llm_model=args.llm_model,
    #     output_dir=output_dir,
    #     top_k=args.top_k,
    #     chunk_size=args.chunk_size,
    #     min_tokens=args.min_tokens,
    #     max_tokens=args.max_tokens,
    # )
    response = generate_response(
        args.query, context_md, mlx, output_dir)
    evaluate_results(args.query, context_md, response,
                     args.llm_model, output_dir)
    if hasattr(logger, "success"):
        logger.success("Search engine execution completed")
    else:
        logger.info("Search engine execution completed")


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
                   default="all-MiniLM-L6-v2", help="Embedding model to use")
    p.add_argument("-min", "--min_tokens", type=int, default=50,
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


if __name__ == "__main__":
    logger.info("Starting search engine script")
    asyncio.run(main())
    logger.info("Search engine script finished")
