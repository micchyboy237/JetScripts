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
    sub_url_dir = format_sub_url_dir(url)
    sub_output_dir = os.path.join(output_dir, "pages", sub_url_dir)
    os.makedirs(sub_output_dir, exist_ok=True)
    save_file(html, f"{sub_output_dir}/page.html")
    md_content = convert_html_to_markdown(html)
    save_file(md_content, f"{sub_output_dir}/md_content.md")
    md_content_markdownify = convert_html_to_markdownify(html)
    save_file(md_content_markdownify,
              f"{sub_output_dir}/md_content_markdownify.md")
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
    readability_overall = analyze_readability(docs_text)
    save_file(readability_overall,
              f"{sub_output_dir}/readability_overall.json")
    readability_docs = [analyze_readability(doc.text) for doc in docs]
    save_file(readability_docs, f"{sub_output_dir}/readability_docs.json")
    if url_to_result and url in url_to_result:
        result = url_to_result[url]
        if not result.get("publishedDate"):
            published_date = scrape_published_date(html)
            result["publishedDate"] = published_date if published_date else None
            logger.debug(f"Scraped published date for {url}: {published_date}")
    return url, html, docs, readability_overall


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
            result = await process_url_content(
                url=url,
                html=html,
                query=query,
                output_dir=output_dir,
                from_reranked_link=False,
                url_to_result=url_to_result
            )
            if result:
                all_url_html_date_tuples.append(result)
                links = set(scrape_links(html, url))
                links = [link for link in links if (
                    link != url if isinstance(link, str) else link["url"] != url)]
                all_links.extend(links)
                logger.debug(f"Extracted {len(links)} links from {url}")
    all_links = list(set(all_links))
    all_links = [link for link in all_links if (
        link not in selected_urls if isinstance(link, str) else link["url"] not in selected_urls)]
    save_file(all_links, os.path.join(output_dir, "links.json"))
    logger.debug(f"Total unique links extracted: {len(all_links)}")
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
    """Search documents, classify relevance with MLXRAGClassifier, and group results with source_url at the top of each group, optionally limiting total tokens to max_tokens and filtering contexts with fewer than min_tokens."""
    if top_k is None and max_tokens is None:
        raise ValueError(
            "At least one of top_k or max_tokens must be provided")
    logger.info(
        f"Searching {len(all_docs)} documents for query: {query}, top_k={top_k}, max_tokens={max_tokens}, min_tokens={min_tokens}")

    # Chunk documents
    # chunked_docs = chunk_headers(
    #     all_docs, max_tokens=chunk_size, model=embed_model)
    # save_file({"query": query, "count": len(chunked_docs), "results": chunked_docs},
    #           os.path.join(output_dir, "chunked_docs.json"))

    # # Filter documents for search
    # docs_to_search = all_docs
    # logger.debug(
    #     f"Filtered to {len(docs_to_search)} documents for search (excluding header level 1, empty content, and below min_tokens)")

    # # Search documents
    # search_doc_results = search_docs(
    #     query=query,
    #     documents=docs_to_search,
    #     ids=[doc.id_ for doc in docs_to_search],
    #     model=embed_model,
    #     top_k=None,  # Get all results for classification
    #     threshold=0.7
    # )
    # save_file(
    #     {"query": query, "count": len(
    #         search_doc_results), "results": search_doc_results},
    #     os.path.join(output_dir, "search_doc_results.json")
    # )
    # logger.info(
    #     f"Saved {len(search_doc_results)} search results to {output_dir}/search_doc_results.json")
    search_doc_results = all_docs

    # Classify relevance using MLXRAGClassifier
    classifier_query = f"Will this webpage header have a concrete answer to this query?\nQuery: {query}"
    chunks = [doc.metadata["header"] for doc in search_doc_results]
    source_urls = [doc.metadata["source_url"] for doc in search_doc_results]

    start_classify = time.time()
    mlx_classifier = MLXRAGClassifier(
        model_name=llm_model, batch_size=4, show_progress=True)
    logger.info("Generating embeddings for classification")
    embeddings = mlx_classifier.generate_embeddings(
        chunks, group_ids=source_urls)
    logger.info(f"Classifying headers for query: {classifier_query}")

    classification_results = []
    rank = 0
    for label, score, idx in mlx_classifier.stream_generate(
        classifier_query, chunks, embeddings, top_k=len(chunks), relevance_threshold=0.7
    ):
        rank += 1
        classification_results.append({
            "doc_index": search_doc_results[idx]["doc_index"],
            "rank": rank,
            "chunk_index": 0,  # Set to 0 since chunking is not used
            "header_level": search_doc_results[idx]["header_level"],
            "label": label,
            "score": score,
            "source_url": search_doc_results[idx]["source_url"],
            "header": search_doc_results[idx]["header"],
            "content": search_doc_results[idx]["content"],
        })
    end_classify = time.time()
    logger.info(
        f"Classification took {end_classify - start_classify:.2f} seconds")

    # Filter for classification results using header_level and content
    classification_results = [
        cr for cr in classification_results
        if cr["header_level"] > 1 and cr["content"].strip()
    ]

    # Save classification results
    save_file(
        {
            "query": classifier_query,
            "count": len(classification_results),
            "results": classification_results
        },
        os.path.join(output_dir, "classification_results.json")
    )

    # Convert classification results to HeaderDocumentWithScore and filter for relevant documents
    relevant_results: List[HeaderDocumentWithScore] = []
    for cr in classification_results:
        if cr["label"] == "relevant":
            # Find the corresponding document from search_doc_results
            for result in search_doc_results:
                if result["doc_index"] == cr["doc_index"]:
                    # Create HeaderDocumentWithScore with classification score
                    doc_with_score = HeaderDocumentWithScore(
                        id_=result["id"],
                        text=result["text"],
                        metadata=result["metadata"],
                        score=cr["score"]  # Use classification score
                    )
                    relevant_results.append(doc_with_score)
                    break

    relevant_results = relevant_results[:top_k] if top_k else relevant_results
    logger.debug(
        f"Filtered to {len(relevant_results)} relevant documents after classification")

    # Process tokens for relevant results
    result_texts = [result.text for result in relevant_results]
    context_tokens = count_tokens(llm_model, result_texts, prevent_total=True)
    limited_results = relevant_results
    limited_context_tokens = context_tokens
    total_tokens = sum(context_tokens)

    # Apply max_tokens limit
    if max_tokens is not None:
        total_tokens = 0
        limited_results = []
        limited_context_tokens = []
        for result, tokens in zip(relevant_results, context_tokens):
            if total_tokens + tokens > max_tokens:
                break
            limited_results.append(result)
            limited_context_tokens.append(tokens)
            total_tokens += tokens

    # Group and format contexts
    contexts: List[str] = []
    url_to_docs = defaultdict(list)
    for doc in limited_results:
        source_url = doc.metadata["source_url"]
        url_to_docs[source_url].append(doc)

    for source_url in url_to_docs:
        sorted_docs = sorted(
            url_to_docs[source_url],
            key=lambda x: x.metadata.get("doc_index", 0)
        )
        contexts.append(f"<!-- Source: {source_url} -->")
        contexts.extend(doc.text for doc in sorted_docs)
        logger.debug(
            f"Added source_url header with {len(sorted_docs)} sorted documents: {source_url}")

    context = "\n\n".join(contexts)
    save_file(context, os.path.join(output_dir, "context.md"))
    logger.debug(f"Generated context with {len(contexts)} segments")

    # Save final context info
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
