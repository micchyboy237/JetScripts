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
    embed_model = "all-mpnet-base-v2"
    rerank_model = "cross-encoder/ms-marco-MiniLM-L-12-v2"
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
    # sub_queries = decompose_query(query)
    # save_file({"query": query, "sub_queries": sub_queries}, os.path.join(
    #     output_dir, "queries.json"))

    # Rerank docs
    search_result_docs = [
        f"Title: {result["title"]}\nContent: {result["content"]}" for result in search_results]
    top_n = len(search_result_docs)

    # queries = [*sub_queries]
    # combined_query = "\n".join(queries)

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

    headers_without_h1 = [doc for doc in all_docs if doc["header_level"] != 1]

    # Search headers

    # search_doc_results = search_diverse_context(
    #     query=combined_query,
    #     headers=headers_without_h1,
    #     model_name=embed_model,
    #     rerank_model=rerank_model,
    #     top_k=20,
    #     num_results=10,
    #     lambda_param=0.5
    # )

    # # Remove embedding attribute from each result before saving
    # results_without_embeddings = [
    #     {k: v for k, v in result.items() if k != 'embedding'} for result in search_doc_results]
    # save_file({
    #     "query": combined_query,
    #     "results": results_without_embeddings
    # }, os.path.join(output_dir, "search_doc_results.json"))

    # Extract headers from all_docs, excluding level 1 headers
    top_k = 10
    docs = [header["text"]
            for header in all_docs if header["header_level"] != 1]

    # Perform search using the extracted header texts
    search_by_pos_results = search_by_pos(query, docs)
    # Get query POS tags
    query_pos = get_pos_tag(query)
    # Calculate document counts for each query lemma
    lemma_doc_counts: Dict[str, int] = {
        pos_tag['lemma']: 0 for pos_tag in query_pos}
    for result in search_by_pos_results:
        matched_lemmas = {pos_tag['lemma']
                          for pos_tag in result['matching_words_with_pos_and_lemma']}
        for lemma in lemma_doc_counts:
            if lemma in matched_lemmas:
                lemma_doc_counts[lemma] += 1
    total_docs = len(docs)
    save_file({
        "query_pos": [
            {
                **pos_tag,
                "document_count": lemma_doc_counts[pos_tag['lemma']],
                "document_percentage": round((lemma_doc_counts[pos_tag['lemma']] / total_docs * 100), 2) if total_docs > 0 else 0.0
            } for pos_tag in query_pos
        ],
        "documents_pos": [
            {
                "doc_index": result["doc_index"],
                "matching_words_count": result["matching_words_count"],
                "matching_words": ", ".join(item["lemma"] for item in result["matching_words_with_pos_and_lemma"]),
                "text": result["text"],
            } for result in search_by_pos_results
        ],
    }, f"{output_dir}/search_by_pos_results.json")

    # Map search results back to the original headers in all_docs
    search_doc_results = [
        header for header in all_docs
        if header["header_level"] != 1 and header["text"] in [result["text"] for result in search_by_pos_results[:top_k]]
    ]
    save_file({
        "query": query,
        "results": search_doc_results
    }, os.path.join(output_dir, "search_doc_results.json"))

    # Sort by doc_index
    sorted_results = sorted(
        search_doc_results, key=lambda x: x["doc_index"], reverse=True)
    contexts = [
        # f"{result["header"]}\n{result["content"]}" for result in sorted_results
        f"{result["text"]}" for result in sorted_results
    ]

    # Run LLM response
    PROMPT_TEMPLATE = """\
Context information is below.
---------------------
{context}
---------------------

Given the context information, answer the query.

Query: {query}
"""

    contexts = contexts[:5]
    context = "\n\n".join(contexts)
    save_file(contexts, os.path.join(output_dir, "contexts.json"))

    response = ""
    prompt = PROMPT_TEMPLATE.format(query=query, context=context)
    for chunk in mlx.stream_chat(
        prompt,
        temperature=0.3,
        verbose=True
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content

    save_file({"query": query, "context": context, "response": response},
              os.path.join(output_dir, "chat_response.json"))
