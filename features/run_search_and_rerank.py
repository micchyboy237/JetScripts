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
from jet.scrapers.hrequests_utils import scrape_urls
from jet.transformers.link_formatters import LinkFormatter, format_links_for_embedding
from jet.wordnet.text_chunker import truncate_texts
from jet.vectors.document_types import HeaderDocument
from jet.vectors.search_with_clustering import search_documents
from jet.wordnet.analyzers.text_analysis import ReadabilityResult, analyze_readability, analyze_text
from jet.code.splitter_markdown_utils import get_md_header_docs
from jet.file.utils import save_file
from jet.llm.mlx.base import MLX
from jet.llm.mlx.helpers import decompose_query, get_system_date_prompt, rewrite_query
from jet.llm.mlx.token_utils import count_tokens
from jet.llm.embeddings.sentence_embedding import get_tokenizer_fn
from jet.scrapers.browser.playwright_utils import scrape_multiple_urls
from jet.scrapers.preprocessor import html_to_markdown
from jet.scrapers.utils import scrape_links, scrape_published_date, search_data
from jet.llm.utils.search_docs import search_docs
from jet.llm.mlx.tasks.eval.evaluate_context_relevance import evaluate_context_relevance
from jet.llm.mlx.tasks.eval.evaluate_response_relevance import evaluate_response_relevance
from jet.wordnet.words import count_words


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
    analysis = analyze_text(text)
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
    if not url_html_date_tuples or limit <= 0:
        return []

    doc_scores = []
    for url, html, _ in url_html_date_tuples:
        try:
            docs = get_md_header_docs(html, ignore_links=False)
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
    return [(url, html, docs, readability_result) for url, html, docs, readability_result, _ in doc_scores[:limit]]


def initialize_output_directory(script_path: str) -> str:
    """Create and return the output directory path."""
    script_dir = os.path.dirname(os.path.abspath(script_path))
    output_dir = os.path.join(script_dir, "generated", os.path.splitext(
        os.path.basename(script_path))[0])
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def initialize_search_components(
    llm_model: str,
    embed_model: str,
    seed: int
) -> Tuple[MLX, callable]:
    """Initialize MLX model and tokenizer."""
    mlx = MLX(llm_model, seed=seed)
    tokenize = get_tokenizer_fn(embed_model)
    return mlx, tokenize


async def fetch_search_results(query: str, output_dir: str, use_cache: bool = False) -> Tuple[List[Dict], List[str]]:
    """Fetch search results and save them."""
    browser_search_results = search_data(query, use_cache=use_cache)
    save_file(
        {"query": query, "count": len(
            browser_search_results), "results": browser_search_results},
        os.path.join(output_dir, "browser_search_results.json")
    )
    urls = [item["url"] for item in browser_search_results]
    html_list = await scrape_urls(urls, num_parallel=5)
    return browser_search_results, html_list


def process_search_results(
    browser_search_results: List[Dict],
    html_list: List[str],
    output_dir: str
) -> Tuple[List[Tuple[str, str, Optional[str]]], List[str]]:
    """Process search results and extract links."""
    all_url_html_date_tuples = []
    all_links = []

    for result, html_str in zip(browser_search_results, html_list):
        url = result["url"]
        if not result.get("publishedDate"):
            published_date = scrape_published_date(html_str)
            result["publishedDate"] = published_date if published_date else None

        if html_str:
            links = set(scrape_links(html_str, url))
            links = [link for link in links if (
                link != url if isinstance(link, str) else link["url"] != url)]
            all_links.extend(links)

        all_url_html_date_tuples.append(
            (url, html_str, result.get("publishedDate")))

    all_links = list(set(all_links))
    save_file(all_links, os.path.join(output_dir, "links.json"))
    return all_url_html_date_tuples, all_links


def process_documents(
    url_html_date_tuples: List[Tuple[str, str, Optional[str]]],
    output_dir: str
) -> Tuple[List[HeaderDocument], List[Dict]]:
    """Process documents and extract headers."""
    all_url_docs_tuples = filter_htmls_with_best_combined_mtld(
        url_html_date_tuples)
    all_docs = []
    headers = []

    for url, html_str, docs, readability_result in all_url_docs_tuples:
        for doc in docs:
            doc.metadata["source_url"] = url
            headers.append({**doc.metadata, "text": doc.text})
        all_docs.extend(docs)

    save_file(all_docs, os.path.join(output_dir, "docs.json"))
    save_file(headers, os.path.join(output_dir, "headers.json"))
    return all_docs, headers


def search_and_group_documents(
    query: str,
    all_docs: List[HeaderDocument],
    embed_model: str,
    llm_model: str,
    top_k: int,
    output_dir: str
) -> Tuple[List[Dict], str]:
    """Search documents and group results."""
    docs_to_search = [
        doc for doc in all_docs if doc.metadata["header_level"] != 1]
    search_doc_results = search_docs(
        query=query,
        documents=[doc.text for doc in docs_to_search],
        ids=[doc.id_ for doc in docs_to_search],
        model=embed_model,
        top_k=top_k,
    )

    save_file(
        {"query": query, "count": len(
            search_doc_results), "results": search_doc_results},
        os.path.join(output_dir, "search_doc_results.json")
    )

    search_result_dict = {result["id"]                          : result for result in search_doc_results}
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

    contexts = [doc["text"] for doc in sorted_doc_results if doc["is_top"]]
    context = "\n\n".join(contexts)
    save_file(context, os.path.join(output_dir, "context.md"))

    context_tokens = count_tokens(llm_model, context, prevent_total=True)
    save_file({
        "total_tokens": context_tokens,
        "contexts": contexts
    }, os.path.join(output_dir, "contexts.json"))
    return sorted_doc_results, context


def generate_response(
    query: str,
    context: str,
    llm_model: str,
    mlx: MLX,
    output_dir: str
) -> str:
    """Generate and save LLM response."""
    PROMPT_TEMPLATE = """\
Context information is below.
---------------------
{context}
---------------------

Given the context information, answer the query.

Query: {query}
"""
    prompt = PROMPT_TEMPLATE.format(query=query, context=context)
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
    return response


def evaluate_results(
    query: str,
    context: str,
    response: str,
    llm_model: str,
    output_dir: str
) -> None:
    """Evaluate context and response relevance."""
    os.makedirs(os.path.join(output_dir, "eval"), exist_ok=True)
    eval_context_result = evaluate_context_relevance(query, context, llm_model)
    save_file(
        eval_context_result,
        os.path.join(output_dir, "eval",
                     "evaluate_context_relevance_result.json")
    )
    eval_response_result = evaluate_response_relevance(
        query, context, response, llm_model)
    save_file(
        eval_response_result,
        os.path.join(output_dir, "eval",
                     "evaluate_response_relevance_result.json")
    )


async def main():
    """Main function to orchestrate the search and response generation."""
    query = "List trending isekai anime 2025."
    top_k = 10
    embed_model = "all-MiniLM-L12-v2"
    llm_model = "llama-3.2-3b-instruct-4bit"
    seed = 45
    use_cache = False

    output_dir = initialize_output_directory(__file__)
    mlx, _ = initialize_search_components(llm_model, embed_model, seed)
    query = rewrite_query(query, llm_model)
    browser_search_results, html_list = await fetch_search_results(query, output_dir, use_cache=use_cache)
    url_html_date_tuples, _ = process_search_results(
        browser_search_results, html_list, output_dir)
    url_html_date_tuples.sort(key=lambda x: x[2] or "", reverse=True)
    all_docs, _ = process_documents(url_html_date_tuples, output_dir)
    sorted_doc_results, context = search_and_group_documents(
        query, all_docs, embed_model, llm_model, top_k, output_dir)
    response = generate_response(query, context, llm_model, mlx, output_dir)
    evaluate_results(query, context, response, llm_model, output_dir)

if __name__ == "__main__":
    asyncio.run(main())
