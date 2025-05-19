import json
import time
import asyncio
import os
import shutil
from typing import Generator, List, Optional, Union, TypedDict
from datetime import datetime
from jet.token.token_utils import split_headers
from tqdm import tqdm
from mlx_lm import load
from jet.wordnet.analyzers.text_analysis import analyze_readability, analyze_text
from jet.code.splitter_markdown_utils import Header, extract_md_header_contents, get_md_header_contents, get_md_header_docs
from jet.file.utils import save_file
from jet.llm.mlx.base import MLX
from jet.llm.mlx.helpers import decompose_query
from jet.llm.mlx.token_utils import merge_texts
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


def filter_htmls_with_best_combined_mtld(htmls: List[str], limit: int = 5, min_mtld: int = 100) -> List[str]:
    """
    Filters a list of HTML strings to return the top <limit> items with the highest combined MTLD scores,
    excluding any items with MTLD score below min_mtld.

    Args:
        htmls: List of HTML content strings
        limit: Maximum number of HTML items to return
        min_mtld: Minimum MTLD score required to include an HTML

    Returns:
        List of HTML strings with the highest combined MTLD scores, up to the specified limit
    """
    if not htmls or limit <= 0:
        return []

    html_scores = []
    for html in htmls:
        try:
            readability_result = analyze_readability(html)
            mtld_score = readability_result['mtld']
            if mtld_score >= min_mtld:
                html_scores.append((html, mtld_score))
        except ValueError:
            continue

    html_scores.sort(key=lambda x: x[1], reverse=True)
    return [html for html, _ in html_scores[:min(limit, len(html_scores))]]


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "generated",
                              os.path.splitext(os.path.basename(__file__))[0])

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    query = "List trending isekai anime this year."
    model_path = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    # embed_models = ["mxbai-embed-large"]
    embed_model = "all-minilm:33m"

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

    # query_scores = query_similarity_scores(
    #     queries, search_result_docs, model=embed_models)
    # save_file({"queries": queries, "results": query_scores},
    #           os.path.join(output_dir, "search_query_scores.json"))
    combined_query = "\n".join(queries)
    search_docs_results = search_docs(
        combined_query, search_result_docs, top_k=10)
    save_file({"combined_query": combined_query, "results": search_docs_results},
              os.path.join(output_dir, "search_docs_results.json"))

    # Use hybrid search
    # engine = HybridSearchEngine()
    # engine.fit(search_result_docs)

    # print("\n🔎 Hybrid Search Results:\n")
    # simple_results = engine.search(
    #     query, top_n=top_n, alpha=0.5, use_mmr=False)
    # for r in simple_results:
    #     print(f"Score: {r['score']:.4f} | Document: {r['document'][:100]}")
    # save_file(simple_results, os.path.join(output_dir, "hybrid_search.json"))

    # print("\n🔎 Hybrid Search Results w/ MMR Diversity:\n")
    # mmr_results = engine.search(
    #     query, top_n=top_n, alpha=0.5, use_mmr=True, lambda_param=0.7)
    # for r in mmr_results:
    #     print(f"Score: {r['score']:.4f} | Document: {r['document'][:100]}")
    # save_file(mmr_results, os.path.join(
    #     output_dir, "hybrid_search_with_diversity.json"))

    sub_dir = f"{output_dir}/searched_html"

    # Convert html to docs
    urls = [item["url"] for item in search_results]
    html_list = asyncio.run(scrape_urls(urls, num_parallel=5))

    filtered_html_list = filter_htmls_with_best_combined_mtld(html_list)
    all_url_html_tuples = list(zip(urls, filtered_html_list))

    selected_url_html_tuples = []
    for url, html_str in tqdm(all_url_html_tuples, desc="Processing"):
        if html_str:
            logger.debug(f"Scraped {url}")

            selected_url_html_tuples.append((url, html_str))

            output_dir_url = safe_path_from_url(url, sub_dir)

            save_file(format_html(html_str), os.path.join(
                output_dir_url, "doc.html"))

            # docs = extract_texts_by_hierarchy(html_str)
            docs = get_md_header_docs(html_str, [
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ])
            save_file(docs, f"{output_dir_url}/headers.json")

            splitted_docs = split_headers(
                docs, embed_model, chunk_size=200, chunk_overlap=20)
            save_file(splitted_docs, f"{output_dir_url}/splitted-headers.json")

            headers = [item["header"] for item in splitted_docs]
            logger.debug(f"Headers (all) length: {len(headers)}")
            save_file("\n".join(headers), os.path.join(
                output_dir_url, "headers.md"))

            all_links = scrape_links(html_str, base_url=url)
            save_file(all_links, os.path.join(
                output_dir_url, "links.json"))

            # Load the model and tokenizer
            model, tokenizer = load(model_path)

            # # Merge docs with max tokens
            # max_tokens = 300

            # def _tokenizer(text: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
            #     if isinstance(text, str):
            #         token_ids = tokenizer.encode(
            #             text, add_special_tokens=False)
            #         return tokenizer.convert_ids_to_tokens(token_ids)
            #     else:
            #         token_ids_list = tokenizer.batch_encode_plus(
            #             text, add_special_tokens=False)["input_ids"]
            #         return [tokenizer.convert_ids_to_tokens(ids) for ids in token_ids_list]

            # merged_docs = merge_texts_by_hierarchy(
            #     html_str, _tokenizer, max_tokens)

            # token_counts = [item["token_count"] for item in merged_docs]

            # Analyze headings
            header_stats = []
            for item in tqdm(splitted_docs, desc="Analyzing headers"):
                text = item["text"]
                header = f"{(item["parent_header"] or "").strip()}\n{item["header"]}"
                stats = get_header_stats(text)
                header_stats.append(
                    {"stats": stats, "header": header, "text": text})
                save_file(header_stats, f"{output_dir_url}/header_stats.json")

            context_docs = [
                f"{(item["parent_header"] or "").strip()}\n{item["header"]}\n{item["content"]}" for item in splitted_docs
                if not item["header_level"] == 1
            ]
            md_context = "\n\n".join(context_docs)
            save_file(md_context, os.path.join(output_dir_url, "context.md"))

            title_and_metadata = scrape_title_and_metadata(html_str)

            # Analyze doc
            stats = analyze_text(md_context)
            save_file(
                {"url": url, "title": title_and_metadata["title"], "stats": stats}, f"{output_dir_url}/overall_stats.json")

            # Rerank splitted_docs
            # query_scores = query_similarity_scores(
            #     queries, context_docs, model=embed_models)
            # save_file({"queries": queries, "results": query_scores},
            #           os.path.join(output_dir_url, "query_scores.json"))
            search_docs_results = search_docs(combined_query, context_docs)
            save_file({"combined_query": combined_query, "results": search_docs_results},
                      os.path.join(output_dir_url, "search_docs_results.json"))

            headers = [item["header"] for item in header_stats]
            logger.debug(f"Headers (context) length: {len(headers)}")
            save_file("\n".join(headers), os.path.join(
                output_dir_url, "context_headers.md"))

            save_file({
                "url": url,
                "title": title_and_metadata["title"],
                "headers": len(headers),
                # "tokens": {
                #     "min_tokens": min(token_counts),
                #     "max_tokens": max(token_counts),
                #     "ave_tokens": sum(token_counts) / len(token_counts) if token_counts else 0,
                # },
                "metadata": title_and_metadata["metadata"],
                # "text_analysis": compute_info(query_scores),
                "top_header": {
                    "doc_index": search_docs_results[0]
                },
            }, os.path.join(output_dir_url, "context_info.json"))

        else:
            logger.error(f"Failed to fetch {url}")

    logger.success(f"Done scraping urls {len(selected_url_html_tuples)}")

    # Save all search_docs_results -> results as one combined search_docs_results.json under output_dir sorted by score in desc.
    all_search_results: List[SimilarityResult] = []

    for url, html_str in selected_url_html_tuples:
        output_dir_url = safe_path_from_url(url, sub_dir)
        search_results_file = os.path.join(
            output_dir_url, "search_docs_results.json")

        if os.path.exists(search_results_file):
            with open(search_results_file, "r") as f:
                search_results = json.load(f)
                # Add source_url to each result
                for result in search_results["results"]:
                    result["source_url"] = url
                    all_search_results.append(result)

    # Sort all results by score in descending order
    all_search_results.sort(key=lambda x: x["score"], reverse=True)

    # Assign rank based on sorted order
    for idx, result in enumerate(all_search_results, start=1):
        result["rank"] = idx  # 1-based rank

    # Save combined results
    combined_results_file = os.path.join(
        output_dir, "combined_search_results.json")
    save_file({
        "combined_query": combined_query,
        "results": all_search_results
    }, combined_results_file)
