from collections import defaultdict
import json
import re
import time
import asyncio
import os
import shutil
from typing import Dict, Generator, List, Optional, Tuple, Union, TypedDict
from datetime import datetime
from urllib.parse import unquote, urlparse
from jet.features.nltk_search import get_pos_tag, search_by_pos
from jet.transformers.link_formatters import LinkFormatter, format_links_for_embedding
from jet.wordnet.text_chunker import truncate_texts
from jet.vectors.document_types import HeaderDocument
from jet.vectors.search_with_clustering import search_documents
from tqdm import tqdm
from mlx_lm import load
from jet.wordnet.analyzers.text_analysis import ReadabilityResult, analyze_readability, analyze_text
from jet.code.splitter_markdown_utils import Header, extract_md_header_contents, get_header_level, get_md_header_contents, get_md_header_docs
from jet.file.utils import save_file
from jet.llm.mlx.base import MLX
from jet.llm.mlx.helpers import decompose_query, get_system_date_prompt, rewrite_query
from jet.llm.mlx.token_utils import count_tokens
from jet.llm.embeddings.sentence_embedding import get_tokenizer_fn
from jet.llm.mlx.mlx_types import LLMModelType
from jet.logger import logger
from jet.scrapers.browser.playwright_utils import scrape_multiple_urls
from jet.scrapers.preprocessor import html_to_markdown
from jet.scrapers.utils import extract_texts_by_hierarchy, merge_texts_by_hierarchy, safe_path_from_url, scrape_links, scrape_metadata, scrape_published_date, scrape_title_and_metadata, search_data
from jet.scrapers.hrequests_utils import scrape_urls
from jet.transformers.formatters import format_html, format_json
from jet.utils.url_utils import normalize_url
from jet.vectors.hybrid_search_engine import HybridSearchEngine
# from jet.wordnet.similarity import compute_info, query_similarity_scores
from jet.llm.utils.search_docs import search_docs
from jet.llm.mlx.tasks.eval.evaluate_context_relevance import evaluate_context_relevance
from jet.llm.mlx.tasks.eval.evaluate_response_relevance import evaluate_response_relevance
from jet.wordnet.words import count_words


class StepBackQueryResponse(TypedDict):
    """TypedDict for the structured response of the step-back query."""
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


def get_header_stats(text: str):
    analysis = analyze_text(text)
    stats = {
        "mtld": analysis["mtld"],
        "mtld_category": analysis["mtld_category"],
        "overall_difficulty": analysis["overall_difficulty"],
        "overall_difficulty_category": analysis["overall_difficulty_category"],
    }
    return stats


def filter_htmls_with_best_combined_mtld(
    url_html_date_tuples: List[Tuple[str, str, Optional[str]]],
    limit: int = 3,
    min_mtld: float = 100.0
) -> List[Tuple[str, str, List[HeaderDocument], ReadabilityResult]]:
    """
    Filters a list of (url, html, date) tuples to return the top <limit> items with the highest combined MTLD scores,
    excluding any items with MTLD score below min_mtld and fewer than or equal to 5 h2 headers.

    Args:
        url_html_date_tuples: List of tuples containing (url, html_content, published_date).
        limit: Maximum number of items to return (default: 3).
        min_mtld: Minimum MTLD score required to include an item (default: 100.0).

    Returns:
        List of tuples, each containing the URL, HTML string, its corresponding HeaderDocument list,
        and the readability result dictionary, sorted by highest MTLD scores, up to the specified limit.
    """
    if not url_html_date_tuples or limit <= 0:
        return []

    doc_scores = []
    for url, html, _ in url_html_date_tuples:
        try:
            docs = get_md_header_docs(html, ignore_links=False)
            # Count h2 headers
            h2_count = sum(
                1 for doc in docs if doc.metadata['header_level'] == 2)
            if h2_count < 5:
                continue

            docs_text = "\n\n".join(doc.text for doc in docs)
            readability_result = analyze_readability(docs_text)
            mtld_score = readability_result['mtld']
            if mtld_score >= min_mtld:
                doc_scores.append(
                    (url, html, docs, readability_result, mtld_score))
        except (ValueError, KeyError, AttributeError):
            continue

    doc_scores.sort(key=lambda x: x[4], reverse=True)
    return [(url, html, docs, readability_result) for url, html, docs, readability_result, _ in doc_scores[:min(limit, len(doc_scores))]]


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "generated",
                              os.path.splitext(os.path.basename(__file__))[0])

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    query = f"List trending isekai anime 2025."
    # query = "Tips and links to 2025 online registration steps for TikTok live selling in the Philippines."
    top_k = 10
    # top_k = None

    model_path = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    # embed_models = ["mxbai-embed-large"]
    embed_model = "all-MiniLM-L12-v2"
    llm_model = "llama-3.2-3b-instruct-4bit"
    rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    tokenize = get_tokenizer_fn(embed_model)

    logger.info("Initializing MLX and embedding function")
    seed = 45
    mlx = MLX(llm_model, seed=seed)

    # Generate broader query
    query = rewrite_query(query, llm_model)

    # Search web engine
    browser_search_results = search_data(query)

    # Sort search results by latest first
    # browser_search_results = sorted(
    #     browser_search_results, key=lambda x: x.get("published_date", ""), reverse=True)
    save_file({"query": query, "count": len(browser_search_results), "results": browser_search_results}, os.path.join(
        output_dir, "browser_search_results.json"))

    # # Filter search results
    # browser_docs = [
    #     HeaderDocument(
    #         text=f"Title: {search_result["title"]}\nContent: {search_result["content"]}",
    #         url=search_result["url"],
    #         score=search_result["score"]
    #     )
    #     for search_result in browser_search_results
    #     if search_result.get("title")
    # ]
    # search_browser_doc_results = search_docs(
    #     query=query,
    #     documents=[doc.text for doc in browser_docs],
    #     ids=[doc.id_ for doc in browser_docs],
    #     # headers=splitted_docs,
    #     model=embed_model,
    #     # rerank_model=rerank_model,
    #     top_k=None
    # )
    # search_browser_doc_result_ids = [result["id"]
    #                                  for result in search_browser_doc_results]
    # browser_search_result_docs = [
    #     doc for doc in browser_docs
    #     if doc.id_ in search_browser_doc_result_ids
    # ]
    # save_file({
    #     "query": query,
    #     "count": len(browser_search_result_docs),
    #     "results": browser_search_result_docs
    # }, os.path.join(output_dir, "browser_search_result_docs.json"))

    # Scrape htmls from search result urls
    urls = [item["url"] for item in browser_search_results]
    html_list = asyncio.run(scrape_urls(urls, num_parallel=5))
    # Extract published date if not exists
    all_results_html_tuples = list(zip(browser_search_results, html_list))
    all_url_html_date_tuples = []
    all_links = []
    for result, html_str in all_results_html_tuples:
        url = result["url"]

        if not result.get("publishedDate"):
            published_date = scrape_published_date(html_str)

            if not published_date:
                logger.info("No published date:")
                metadata = scrape_metadata(html_str)
                logger.info("Metadata:")
                logger.debug(format_json(metadata))
            else:
                logger.info("Scraped published date:")
                logger.debug(published_date)
                result["publishedDate"] = published_date

        if html_str:
            # Collect initial scraped URLs
            links = set(scrape_links(html_str, url))

            # # Extract URLs from header docs, normalize all to strings
            # docs = get_md_header_docs(html_str)
            # header_links = [
            #     link["url"] if not link["is_heading"] else link
            #     for doc in docs
            #     for link in doc["links"]
            # ]

            # # Convert to strings (in case headings are strings, not dicts)
            # normalized_links = set(str(link) for link in header_links)

            # # Merge sets to remove duplicates
            # links.update(normalized_links)

            # # Convert back to list if needed
            # links = list(links)

            # Filter out base url
            links = [link for link in links
                     if (link != url if isinstance(link, str) else link == link["url"])]

            all_links.extend(links)

        all_url_html_date_tuples.append(
            (url, html_str, result.get("publishedDate")))

    all_links = list(set(all_links))
    save_file(all_links, os.path.join(output_dir, "links.json"))

    # # Initialize formatter
    # formatter = LinkFormatter()
    # # Format links
    # formatted_links = formatter.format_links_for_embedding(all_links)
    # # Save formatted list
    # save_file(formatted_links, os.path.join(
    #     output_dir, "formatted-links.json"))
    # search_links_results = search_docs(
    #     query=query,
    #     documents=formatted_links,
    #     model=embed_model,
    #     top_k=None
    # )
    # # Step 4: Enrich with original link using mapping
    # enriched_results = []
    # for i, result in enumerate(search_links_results):
    #     formatted = formatted_links[i]
    #     enriched_results.append({
    #         **result,
    #         "formatted_link": formatted,
    #         "link": formatter.formatted_to_original_map.get(formatted, "")
    #     })
    # # Save enriched search results
    # save_file({
    #     "query": query,
    #     "results": enriched_results
    # }, os.path.join(output_dir, "search_links_results.json"))

    # Sort by publishedDate in descending order (newest first)
    all_url_html_date_tuples = sorted(
        all_url_html_date_tuples,
        key=lambda x: x[2] or "",  # fallback to "" if date is None
        reverse=True
    )

    # queries = [*sub_queries]
    # combined_query = "\n".join(queries)

    # Convert html to docs
    all_url_docs_tuples = filter_htmls_with_best_combined_mtld(
        all_url_html_date_tuples)

    all_urls = []
    all_docs = []
    headers = []
    for url, html_str, docs, readability_result in all_url_docs_tuples:
        all_urls.append(url)

        for doc in docs:
            doc.metadata["source_url"] = url

            headers.append({
                **doc.metadata,
                "text": doc.text,
            })

        all_docs.extend(docs)

    save_file(all_docs, os.path.join(output_dir, "docs.json"))
    save_file(headers, os.path.join(output_dir, "headers.json"))

    # splitted_docs = split_headers(
    #     all_docs, embed_model, chunk_size=200, chunk_overlap=20)
    # save_file(splitted_docs, os.path.join(output_dir, "splitted_docs.json"))

    # Search headers
    docs_to_search = [doc for doc in all_docs if doc["header_level"] != 1]
    search_doc_results = search_docs(
        query=query,
        documents=[doc.text for doc in docs_to_search],
        ids=[doc.id_ for doc in docs_to_search],
        # headers=splitted_docs,
        model=embed_model,
        # rerank_model=rerank_model,
        top_k=top_k,
    )

    save_file({
        "query": query,
        "count": len(search_doc_results),
        "results": search_doc_results
    }, os.path.join(output_dir, "search_doc_results.json"))

    # Run LLM response
    PROMPT_TEMPLATE = """\
Context information is below.
---------------------
{context}
---------------------

Given the context information, answer the query.

Query: {query}
"""

    # Filter results with positive scores
    # filtered_doc_results = [r for r in search_doc_results if r["score"] > 0]

    # Map search result to ids
    search_result_dict = {result["id"]: result for result in search_doc_results}
    sorted_doc_results = []
    for doc in all_docs:
        if doc["header_level"] != 1:
            if count_words(doc["content"]) < 10:
                continue

            is_top = doc.id_ in search_result_dict
            text = doc.text
            sorted_doc_results.append(
                {**doc.metadata, "text": text, "is_top": is_top})

    # Group results by source_url, parent_header, and is_top
    grouped_by_source_and_parent: dict[tuple[str,
                                             str, bool], List[dict]] = defaultdict(list)
    for result in sorted_doc_results:
        parent_header = result.get("parent_header", None)
        key = (result["source_url"],
               parent_header if parent_header is not None else "", result["is_top"])
        grouped_by_source_and_parent[key].append(result)

    # # Create table of contents with top results section per source_url
    # toc = []
    # seen_source_urls = set()
    # seen_headers = set()

    # for source_url in sorted(set(result["source_url"] for result in sorted_doc_results)):
    #     if source_url:
    #         toc.append(f"<!-- Source: {source_url} -->")
    #         toc.append("# Table of Contents")
    #         seen_source_urls.add(source_url)

    #         # Collect headers for this source_url
    #         headers_by_parent = defaultdict(list)
    #         for result in sorted_doc_results:
    #             if result["source_url"] == source_url:
    #                 parent_header = result.get("parent_header", None)
    #                 header = result.get("header", "").strip()
    #                 header_level = result.get("header_level", 1)
    #                 headers_by_parent[parent_header if parent_header else ""].append(
    #                     (header, header_level))

    #         # Add headers to TOC
    #         for parent_header in sorted(headers_by_parent.keys(), key=lambda x: "" if x is None else x):
    #             if parent_header and parent_header not in seen_headers:
    #                 parent_header_level = min(
    #                     (entry[1] - 1 for entry in headers_by_parent[parent_header]), default=1)
    #                 parent_indent = "\t" * (max(1, parent_header_level) - 1)
    #                 toc.append(
    #                     f"{parent_indent}- {parent_header.lstrip('#').strip()}")
    #                 seen_headers.add(parent_header)

    #             for header, header_level in sorted(headers_by_parent[parent_header], key=lambda x: x[1]):
    #                 if header and header != parent_header and header not in seen_headers:
    #                     indent = "\t" * (header_level - 1)
    #                     toc.append(f"{indent}- {header.lstrip('#').strip()}")
    #                     seen_headers.add(header)

    #         # Add Top Results section for this source_url
    #         top_results = [
    #             result["text"] for (url, _, is_top), results in grouped_by_source_and_parent.items()
    #             if url == source_url and is_top
    #             for result in sorted(results, key=lambda x: x.get("header_level", 1))
    #         ]
    #         if top_results:
    #             toc.append("# Top Results")
    #             toc.extend(top_results)

    # # Save formatted context as markdown file
    # context = "\n\n".join(toc)
    # save_file(context, os.path.join(output_dir, "context.md"))

    # contexts = truncate_texts([doc["text"] for doc in sorted_doc_results], 100)
    contexts = [doc["text"] for doc in sorted_doc_results if doc["is_top"]]

    context = "\n\n".join(contexts)
    save_file(context, os.path.join(output_dir, "context.md"))

    context_tokens: int = count_tokens(
        llm_model, context, prevent_total=True)
    save_file({
        "total_tokens": context_tokens,
        "contexts": contexts
    }, os.path.join(output_dir, "contexts.json"))

    response = ""
    prompt = PROMPT_TEMPLATE.format(query=query, context=context)
    for chunk in mlx.stream_chat(
        prompt,
        system_prompt=get_system_date_prompt(),
        temperature=0.7,
        verbose=True,
        max_tokens=10000
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content

    save_file({"query": query, "context": context, "response": response},
              os.path.join(output_dir, "chat_response.json"))

    # Evaluate context
    evaluate_context_relevance_result = evaluate_context_relevance(
        query, context, llm_model)
    save_file(evaluate_context_relevance_result, os.path.join(
        output_dir, "eval", "evaluate_context_relevance_result.json"))

    # Evaluate context and response
    evaluate_response_relevance_result = evaluate_response_relevance(
        query, context, response, llm_model)
    save_file(evaluate_response_relevance_result, os.path.join(
        output_dir, "eval", "evaluate_response_relevance_result.json"))
