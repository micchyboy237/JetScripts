import json
import time
import asyncio
import os
import shutil
from typing import Generator, List, Optional, Tuple, Union, TypedDict
from datetime import datetime
from jet.token.token_utils import merge_headers, split_headers
from jet.vectors.document_types import HeaderDocument
from jet.vectors.hybrid_reranker import search_documents
from jet.vectors.search_with_mmr import search_diverse_context
from tqdm import tqdm
from mlx_lm import load
from jet.wordnet.analyzers.text_analysis import ReadabilityResult, analyze_readability, analyze_text
from jet.code.splitter_markdown_utils import Header, extract_md_header_contents, get_md_header_contents, get_md_header_docs
from jet.file.utils import save_file
from jet.llm.mlx.base import MLX
from jet.llm.mlx.helpers import decompose_query
from jet.llm.mlx.token_utils import count_tokens
from jet.llm.embeddings.sentence_embedding import get_tokenizer_fn
from jet.llm.mlx.mlx_types import LLMModelType
from jet.logger import logger
from jet.scrapers.browser.playwright_utils import scrape_multiple_urls
from jet.scrapers.preprocessor import html_to_markdown
from jet.scrapers.utils import extract_texts_by_hierarchy, merge_texts_by_hierarchy, safe_path_from_url, scrape_links, scrape_title_and_metadata, search_data
from jet.scrapers.hrequests_utils import scrape_urls
from jet.transformers.formatters import format_html, format_json
from jet.utils.url_utils import normalize_url
from jet.vectors.hybrid_search_engine import HybridSearchEngine
# from jet.wordnet.similarity import compute_info, query_similarity_scores
from jet.llm.utils.transformer_embeddings import SimilarityResult, get_embedding_function, search_docs

logger.info("Initializing MLX and embedding function")
seed = 42
DEFAULT_MODEL = "llama-3.2-3b-instruct-4bit"
mlx = MLX(DEFAULT_MODEL, seed=seed)


def get_url_html_tuples(urls: list[str], top_n: int = 3, num_parallel: int = 3, min_header_count: int = 10, min_avg_word_count: int = 10, output_dir: Optional[str] = None) -> Generator[list[Header], None, None]:
    urls = [normalize_url(url) for url in urls]

    for url, html in scrape_multiple_urls(urls, top_n=top_n, num_parallel=num_parallel, min_header_count=min_header_count, min_avg_word_count=min_avg_word_count):

        headers = get_md_header_contents(html, [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ])

        yield {
            "url": url,
            "headers": headers,
            "html": html,
        }


class StepBackQueryResponse(TypedDict):
    """TypedDict for the structured response of the step-back query."""
    original_query: str
    broader_query: List[str]


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
    htmls: List[str],
    limit: int = 3,
    min_mtld: float = 100.0
) -> List[Tuple[str, List[HeaderDocument], ReadabilityResult]]:
    """
    Filters a list of HTML strings to return the top <limit> items with the highest combined MTLD scores,
    excluding any items with MTLD score below min_mtld.

    Args:
        htmls: List of HTML content strings.
        limit: Maximum number of HTML items to return (default: 3).
        min_mtld: Minimum MTLD score required to include an HTML (default: 100.0).

    Returns:
        List of tuples, each containing an HTML string, its corresponding HeaderDocument list,
        and the readability result dictionary, sorted by highest MTLD scores, up to the specified limit.
    """
    if not htmls or limit <= 0:
        return []

    doc_scores = []
    for html in htmls:
        try:
            docs = get_md_header_docs(html, [
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ])
            docs_text = "\n\n".join(doc.text for doc in docs)

            readability_result = analyze_readability(docs_text)
            mtld_score = readability_result['mtld']
            if mtld_score >= min_mtld:
                doc_scores.append((html, docs, readability_result, mtld_score))
        except (ValueError, KeyError, AttributeError):
            continue

    doc_scores.sort(key=lambda x: x[3], reverse=True)
    return [(html, docs, readability_result) for html, docs, readability_result, _ in doc_scores[:min(limit, len(doc_scores))]]


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "generated",
                              os.path.splitext(os.path.basename(__file__))[0])

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    query = "List trending isekai reincarnation anime this year."
    model_path = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    # embed_models = ["mxbai-embed-large"]
    embed_model = "all-minilm:33m"
    tokenize = get_tokenizer_fn(embed_model)

    # Search web engine
    search_results = search_data(query)
    save_file({"query": query, "results": search_results}, os.path.join(
        output_dir, "search_results.json"))

    # Generate broader query
    # transformed_query = generate_step_back_query(query, mlx)
    # transformed_query = "Popular and trending isekai anime series released in 2023 or later, along with their genres, ratings, and a brief summary of each show."
    # sub_queries = [transformed_query]
    # Decompose query to sub-queries
    sub_queries = decompose_query(query)
    save_file({"query": query, "sub_queries": sub_queries}, os.path.join(
        output_dir, "queries.json"))

    # Rerank docs
    queries = [query, *sub_queries]
    search_result_docs = [
        f"Title: {result["title"]}\nContent: {result["content"]}" for result in search_results]
    top_n = len(search_result_docs)

    combined_query = "\n".join(queries)

    sub_dir = f"{output_dir}/searched_html"

    # Convert html to docs
    urls = [item["url"] for item in search_results]
    html_list = asyncio.run(scrape_urls(urls, num_parallel=5))

    filtered_docs_list = filter_htmls_with_best_combined_mtld(html_list)
    all_url_html_tuples = list(zip(urls, filtered_docs_list))

    all_urls = []
    all_docs = []
    for url, (html_str, docs, readability_result) in tqdm(all_url_html_tuples, desc="Processing"):
        all_urls.append(url)
        all_docs.extend(docs)

    save_file(all_docs, os.path.join(output_dir, "headers.json"))

    search_doc_results = search_diverse_context(
        query=combined_query,
        headers=all_docs,
        model_name="all-MiniLM-L12-v2",
        rerank_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
        top_k=20,
        num_results=10,
        lambda_param=0.5
    )
    save_file({
        "query": combined_query,
        "results": search_doc_results
    }, os.path.join(output_dir, "search_doc_results.json"))

    # splitted_nodes = split_headers(
    #     all_docs, embed_model, chunk_size=200, chunk_overlap=0)

    # merged_nodes = merge_headers(
    #     splitted_nodes, embed_model, chunk_size=200, chunk_overlap=0)

    # contexts = [
    #     f"{(item["parent_header"] or "").strip()}\n{item["header"]}\n{item["content"]}" for item in merged_nodes
    #     if not item["header_level"] == 1
    # ]

    # results_dir = f"{output_dir}/results"
    # save_file({
    #     "urls": all_urls,
    #     "contexts": len(contexts),
    # }, os.path.join(results_dir, "info.json"))
    # splitted_node_token_counts: list[int] = count_tokens(
    #     embed_model, [node.text for node in splitted_nodes], prevent_total=True)
    # for node, token_count in zip(splitted_nodes, splitted_node_token_counts):
    #     node.metadata["tokens"] = token_count
    # save_file(splitted_nodes, os.path.join(results_dir, "splitted_nodes.json"))
    # merged_node_token_counts: list[int] = count_tokens(
    #     embed_model, [node.text for node in merged_nodes], prevent_total=True)
    # for node, token_count in zip(merged_nodes, merged_node_token_counts):
    #     node.metadata["tokens"] = token_count
    # save_file(merged_nodes, os.path.join(results_dir, "merged_nodes.json"))
    # save_file(contexts, os.path.join(results_dir, "contexts.json"))

    # # Search contexts
    # top_k = 10
    # hybrid_search_doc_results = search_documents(
    #     combined_query, contexts, k=top_k)
    # save_file({
    #     "query": combined_query,
    #     "results": hybrid_search_doc_results
    # }, os.path.join(results_dir, "hybrid_search_doc_results.json"))
    # search_doc_results = search_docs(combined_query, contexts, top_k=top_k)
    # save_file({
    #     "query": combined_query,
    #     "results": search_doc_results
    # }, os.path.join(results_dir, "search_doc_results.json"))
