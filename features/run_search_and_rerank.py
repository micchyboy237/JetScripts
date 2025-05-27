from collections import defaultdict
import json
import time
import asyncio
import os
import shutil
from typing import Dict, Generator, List, Optional, Tuple, Union, TypedDict
from datetime import datetime
from jet.features.nltk_search import get_pos_tag, search_by_pos
from jet.token.token_utils import merge_headers, split_headers
from jet.vectors.document_types import HeaderDocument
from jet.vectors.search_with_clustering import search_documents
from tqdm import tqdm
from mlx_lm import load
from jet.wordnet.analyzers.text_analysis import ReadabilityResult, analyze_readability, analyze_text
from jet.code.splitter_markdown_utils import Header, extract_md_header_contents, get_header_level, get_md_header_contents, get_md_header_docs
from jet.file.utils import save_file
from jet.llm.mlx.base import MLX
from jet.llm.mlx.helpers import decompose_query, get_system_date_prompt
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
from jet.llm.utils.transformer_embeddings import SimilarityResult, get_embedding_function, search_docs
from jet.llm.mlx.tasks.eval.evaluate_context_relevance import evaluate_context_relevance
from jet.llm.mlx.tasks.eval.evaluate_response_relevance import evaluate_response_relevance

logger.info("Initializing MLX and embedding function")
seed = 45
DEFAULT_MODEL = "llama-3.2-3b-instruct-4bit"
mlx = MLX(DEFAULT_MODEL, seed=seed)


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

    query = f"List trending isekai reincarnation anime this year."
    # query = "Tips and links to 2025 online registration steps for TikTok live selling in the Philippines."
    top_k = 20

    model_path = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    # embed_models = ["mxbai-embed-large"]
    embed_model = "all-mpnet-base-v2"
    rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    tokenize = get_tokenizer_fn(embed_model)

    # Search web engine
    search_results = search_data(query)
    # Scrape htmls from search result urls
    urls = [item["url"] for item in search_results]
    html_list = asyncio.run(scrape_urls(urls, num_parallel=5))
    # Extract published date if not exists
    all_results_html_tuples = list(zip(search_results, html_list))
    all_url_html_date_tuples = []
    for result, html_str in all_results_html_tuples:
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

        # docs = get_md_header_docs(html_str, ignore_links=True)

        all_url_html_date_tuples.append(
            (result["url"], html_str, result.get("publishedDate")))

    # Sort by publishedDate in descending order (newest first)
    all_url_html_date_tuples = sorted(
        all_url_html_date_tuples,
        key=lambda x: x[2] or "",  # fallback to "" if date is None
        reverse=True
    )

    # Sort search results by latest first
    search_results = sorted(
        search_results, key=lambda x: x.get("published_date", ""), reverse=True)
    save_file({"query": query, "results": search_results}, os.path.join(
        output_dir, "search_results.json"))

    # Generate broader query
    # transformed_query = generate_step_back_query(query, mlx)
    # transformed_query = "Popular and trending isekai anime series released in 2023 or later, along with their genres, ratings, and a brief summary of each show."
    # sub_queries = [transformed_query]
    # Decompose query to sub-queries
    # sub_queries = decompose_query(query)
    # save_file({"query": query, "sub_queries": sub_queries}, os.path.join(
    #     output_dir, "queries.json"))

    # Format docs
    search_result_docs = [
        f"Title: {result['title']}\nContent: {result['content']}"
        for result in search_results
        if result.get("title")
    ]
    top_n = len(search_result_docs)

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

    # search_doc_results = search_documents(
    #     query=query,
    #     headers=docs,
    #     # headers=splitted_docs,
    #     model_name=embed_model,
    #     # rerank_model=rerank_model,
    #     top_k=top_k,
    #     # lambda_param=0.5,
    #     min_header_level=2,
    #     max_header_level=3
    # )
    search_doc_results = search_docs(
        query=query,
        documents=[doc.text for doc in docs],
        ids=[doc.id_ for doc in docs],
        # headers=splitted_docs,
        model=embed_model,
        # rerank_model=rerank_model,
        top_k=top_k,
    )
    # results_dir = os.path.join(output_dir, "results")
    # save_file(search_doc_results["merge_results"],
    #           os.path.join(results_dir, "merge_results.json"))
    # save_file(search_doc_results["embed_results"],
    #           os.path.join(results_dir, "embed_results.json"))
    # save_file(search_doc_results["rerank_results"],
    #           os.path.join(results_dir, "rerank_results.json"))
    # save_file(search_doc_results,
    #           os.path.join(results_dir, "results.json"))

    # # Extract headers from all_docs, excluding level 1 headers
    # docs = [header["text"]
    #         for header in all_docs if header["header_level"] != 1]

    # # Perform search using the extracted header texts
    # search_by_pos_results = search_by_pos(query, docs)
    # # Get query POS tags
    # query_pos = get_pos_tag(query)
    # # Calculate document counts for each query lemma
    # lemma_doc_counts: Dict[str, int] = {
    #     pos_tag['lemma']: 0 for pos_tag in query_pos}
    # for result in search_by_pos_results:
    #     matched_lemmas = {pos_tag['lemma']
    #                       for pos_tag in result['matching_words_with_pos_and_lemma']}
    #     for lemma in lemma_doc_counts:
    #         if lemma in matched_lemmas:
    #             lemma_doc_counts[lemma] += 1
    # total_docs = len(docs)
    # save_file({
    #     "query_pos": sorted([
    #         {
    #             **pos_tag,
    #             "document_count": lemma_doc_counts[pos_tag["lemma"]],
    #             "document_percentage": (
    #                 round(
    #                     (lemma_doc_counts[pos_tag["lemma"]] / total_docs * 100), 2)
    #                 if total_docs > 0 else 0.0
    #             )
    #         }
    #         for pos_tag in query_pos
    #     ], key=lambda x: x["word"]),
    #     "documents_pos": [
    #         {
    #             "doc_index": result["doc_index"],
    #             "matching_words_count": result["matching_words_count"],
    #             "matching_words": ", ".join(sorted(
    #                 set(item["lemma"]
    #                     for item in result["matching_words_with_pos_and_lemma"])
    #             )),
    #             "text": result["text"],
    #         } for result in search_by_pos_results
    #     ],
    # }, f"{output_dir}/search_by_pos_results.json")
    # # Map search results back to the original headers in all_docs
    # search_doc_results = [
    #     header for header in all_docs
    #     if header["header_level"] != 1 and header["text"] in [result["text"] for result in search_doc_results]
    # ]

    # Remove "embedding" prop
    # search_doc_results = [
    #     {k: v for k, v in result.items() if k != "embedding"}
    #     for result in search_doc_results
    # ]

    save_file({
        "query": query,
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

    # Map search results back to the original headers in all_docs
    search_doc_results = [
        doc for doc in all_docs
        if get_header_level(doc["text"]) != 1 and doc["doc_index"] in [result["doc_index"] for result in search_doc_results]
    ]

    # Sort results by source_url, parent_header, doc_index, and chunk_index
    sorted_doc_results = sorted(
        search_doc_results,
        key=lambda r: (r["source_url"], r["parent_header"],
                       r["doc_index"], r["chunk_index"])
    )

    # Group results by source_url and parent_header
    grouped_by_source_and_parent: dict[tuple[str,
                                             str], List[dict]] = defaultdict(list)
    for result in sorted_doc_results:
        key = (result["source_url"], result["parent_header"])
        grouped_by_source_and_parent[key].append(result)

    # Format markdown context with no duplicate source_urls, parent_headers, or headers
    formatted_context_blocks = []
    seen_source_urls = set()

    for (source_url, parent_header), group in grouped_by_source_and_parent.items():
        # Add source URL as a comment only if not already seen
        if source_url not in seen_source_urls:
            formatted_context_blocks.append(f"<!-- Source: {source_url} -->")
            seen_source_urls.add(source_url)

        # Add parent header only if not already seen and non-empty
        if parent_header.strip() and parent_header not in formatted_context_blocks:
            formatted_context_blocks.append(parent_header)

        # Add entries under the parent header
        for entry in group:
            header = entry.get("header", "").strip()

            # Add header only if distinct from parent_header and not already seen in this group
            if header:
                formatted_context_blocks.append(header)

            # Add content
            formatted_context_blocks.append(entry["content"])

    # Save formatted context as JSON and markdown files
    save_file(formatted_context_blocks, os.path.join(
        output_dir, "contexts.json"))
    context = "\n\n".join(formatted_context_blocks)
    save_file(context, os.path.join(output_dir, "context.md"))

    # # Create structured context entries
    # context_entries: List[ContextEntry] = [
    #     ContextEntry(**result) for result in sorted_doc_results
    # ]

    # # Compile final context information
    # context_info: ContextInfo = {
    #     "model": DEFAULT_MODEL,
    #     "total_tokens": sum(entry["tokens"] for entry in context_entries),
    #     "contexts": context_entries,
    # }

    # # Save to JSON file
    # save_file(context_info, os.path.join(output_dir, "context_info.json"))

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
        query, context, DEFAULT_MODEL)
    save_file(evaluate_context_relevance_result, os.path.join(
        output_dir, "eval", "evaluate_context_relevance_result.json"))

    # Evaluate context and response
    evaluate_response_relevance_result = evaluate_response_relevance(
        query, context, response, DEFAULT_MODEL)
    save_file(evaluate_response_relevance_result, os.path.join(
        output_dir, "eval", "evaluate_response_relevance_result.json"))
