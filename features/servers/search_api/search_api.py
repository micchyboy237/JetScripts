from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
import asyncio
import os
from jet.data.utils import generate_key, generate_unique_hash
from jet.logger import logger
from jet.vectors.document_types import HeaderDocument
from jet.llm.mlx.base import MLX
from jet.llm.mlx.mlx_types import EmbedModelType, LLMModelType
from jet.utils.url_utils import rerank_bm25_plus
from jet.search.searxng import SearchResult
from jet.scrapers.hrequests_utils import scrape_urls
from jet.scrapers.utils import scrape_published_date, scrape_links
from jet.code.splitter_markdown_utils import get_md_header_docs
from jet.file.utils import save_file
from jet.models.tokenizer.base import count_tokens
from jet.models.tasks.hybrid_search_docs_with_bm25 import search_docs
from jet.wordnet.analyzers.text_analysis import analyze_readability
from search_and_rerank import (
    initialize_output_directory,
    initialize_search_components,
    generate_response,
    evaluate_results,
    get_header_stats,
    fetch_search_results,
    ContextInfo
)

app = FastAPI(title="Search API", version="1.0.0")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    top_k: int = Field(
        10, ge=1, le=50, description="Number of top results to return")
    embed_model: str = Field("static-retrieval-mrl-en-v1",
                             description="Embedding model type")
    llm_model: str = Field("llama-3.2-3b-instruct-4bit",
                           description="LLM model type")
    seed: int = Field(45, description="Random seed for reproducibility")
    use_cache: bool = Field(
        False, description="Whether to use cached search results")
    min_mtld: float = Field(
        100.0, ge=0.0, description="Minimum MTLD score for filtering")


class SearchResponse(BaseModel):
    query: str
    context: str
    response: str
    context_info: dict
    headers_stats: dict


async def filter_htmls_with_best_combined_mtld(
    url_html_date_tuples: List[Tuple[str, str, Optional[str]]],
    limit: int = 3,
    min_mtld: float = 100.0
) -> List[Tuple[str, str, List[HeaderDocument]]]:
    logger.info(
        f"Filtering {len(url_html_date_tuples)} HTMLs with min MTLD={min_mtld}")
    doc_scores = []
    for url, html, _ in url_html_date_tuples:
        try:
            logger.debug(f"Processing HTML for URL: {url}")
            docs = get_md_header_docs(html, ignore_links=False)
            if len(docs) < 5:
                logger.warning(
                    f"Skipping {url}: insufficient headers ({len(docs)} < 5)")
                continue
            docs_text = "\n\n".join(doc.text for doc in docs)
            readability = analyze_readability(docs_text)
            mtld_score = readability['mtld']
            if mtld_score >= min_mtld:
                doc_scores.append((url, html, docs, mtld_score))
                logger.debug(
                    f"Added {url} to candidates with MTLD={mtld_score}")
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            continue
    doc_scores.sort(key=lambda x: x[3], reverse=True)
    filtered = [(url, html, docs) for url, html, docs, _ in doc_scores[:limit]]
    logger.info(f"Filtered to {len(filtered)} HTMLs with highest MTLD scores")
    return filtered


async def process_search_results(
    browser_search_results: List[dict],
    query: str,
    output_dir: str
) -> List[Tuple[str, str, Optional[str]]]:
    logger.info(
        f"Processing {len(browser_search_results)} search results for query: {query}")
    urls = [item["url"] for item in browser_search_results]
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
            result["publishedDate"] = published_date
            logger.debug(f"Scraped published date for {url}: {published_date}")
        links = set(scrape_links(html_str, url))
        links = [link for link in links if link != url]
        all_links.extend(links)
        all_url_html_date_tuples.append(
            (url, html_str, result.get("publishedDate")))
    all_links = list(set(all_links))
    save_file(all_links, os.path.join(output_dir, "links.json"))
    reranked_links = rerank_bm25_plus(all_links, query, 3)
    save_file(reranked_links, os.path.join(output_dir, "reranked_links.json"))
    reranked_html_list = await scrape_urls(reranked_links, num_parallel=5)
    for url, html_str in zip(reranked_links, reranked_html_list):
        if html_str:
            published_date = scrape_published_date(html_str)
            all_url_html_date_tuples.append((url, html_str, published_date))
    logger.info(
        f"Processed {len(all_url_html_date_tuples)} URL-HTML-date tuples")
    return all_url_html_date_tuples


async def process_documents(
    url_html_date_tuples: List[Tuple[str, str, Optional[str]]],
    output_dir: str,
    min_mtld: float
) -> List[HeaderDocument]:
    logger.info(f"Processing {len(url_html_date_tuples)} documents")
    all_url_docs_tuples = await filter_htmls_with_best_combined_mtld(
        url_html_date_tuples, min_mtld=min_mtld
    )
    all_docs = []
    headers = []
    for url, _, docs in all_url_docs_tuples:
        for doc in docs:
            doc.metadata["source_url"] = url
            headers.append({
                "doc_index": doc["doc_index"],
                "source_url": url,
                "parent_header": doc["parent_header"],
                "header": doc["header"],
            })
        all_docs.extend(docs)
    save_file(all_docs, os.path.join(output_dir, "docs.json"))
    save_file(headers, os.path.join(output_dir, "headers.json"))
    return all_docs


async def search_and_group_documents(
    query: str,
    all_docs: List[HeaderDocument],
    embed_model: str,
    llm_model: str,
    top_k: int,
    output_dir: str
) -> Tuple[List[dict], str, ContextInfo]:
    logger.info(f"Searching {len(all_docs)} documents for query: {query}")
    docs_to_search = [
        doc for doc in all_docs if doc.metadata["header_level"] != 1]
    search_doc_results = search_docs(
        query=query,
        documents=docs_to_search,
        ids=[doc.id_ for doc in docs_to_search],
        model=embed_model,
        top_k=top_k,
    )
    save_file(
        {"query": query, "count": len(
            search_doc_results), "results": search_doc_results},
        os.path.join(output_dir, "search_doc_results.json")
    )
    sorted_doc_results = sorted(
        search_doc_results,
        key=lambda x: (x["document"]["metadata"]["source_url"], x["doc_index"])
    )
    save_file(
        {"query": query, "count": len(
            sorted_doc_results), "results": sorted_doc_results},
        os.path.join(output_dir, "sorted_doc_results.json")
    )
    # Save context metadata
    logger.info(f"Counting contexts ({len(sorted_doc_results)}) tokens...")
    result_texts = [result["text"] for result in sorted_doc_results]
    context_tokens: List[int] = count_tokens(
        llm_model, result_texts, prevent_total=True)
    total_tokens = sum(context_tokens)
    save_file(
        {
            "total_tokens": total_tokens,
            "contexts": [
                {
                    "doc_index": result["doc_index"],
                    "score": result["score"],
                    "tokens": tokens,
                    "text": result["text"]
                }
                for result, tokens in zip(sorted_doc_results, context_tokens)
            ]
        },
        os.path.join(output_dir, "contexts.json")
    )
    logger.info(
        f"Saved context with {context_tokens} tokens to {output_dir}/contexts.json")

    contexts = []
    context_info: ContextInfo = {
        "model": llm_model, "total_tokens": total_tokens, "contexts": []}
    current_url = None
    for idx, doc in enumerate(sorted_doc_results):
        source_url = doc["document"]["metadata"]["source_url"]
        if source_url != current_url:
            contexts.append(f"<!-- Source: {source_url} -->")
            current_url = source_url
        contexts.append(doc["text"])
        context_info["contexts"].append({
            "rank": doc["rank"],
            "doc_index": doc["doc_index"],
            "chunk_index": doc["document"]["metadata"].get("chunk_index", 0),
            "tokens": context_tokens[idx],
            "score": doc["score"],
            "rerank_score": doc.get("rerank_score", 0.0),
            "source_url": source_url,
            "parent_header": doc["document"]["metadata"]["parent_header"],
            "header": doc["document"]["metadata"]["header"],
            "content": doc["text"]
        })
    context = "\n\n".join(contexts)
    save_file(context, os.path.join(output_dir, "context.md"))
    return sorted_doc_results, context, context_info


@app.post("/search")
async def search_endpoint(request: SearchRequest):
    try:
        shared_dir = generate_key(
            request.query, request.llm_model, request.embed_model)
        sub_dir = generate_unique_hash()
        output_dir = initialize_output_directory(
            __file__, f"{shared_dir}/{sub_dir}")
        mlx, _ = initialize_search_components(
            request.llm_model, request.embed_model, request.seed)

        browser_search_results = await fetch_search_results(
            request.query, output_dir, request.use_cache
        )
        url_html_date_tuples = await process_search_results(
            browser_search_results, request.query, output_dir
        )
        url_html_date_tuples.sort(key=lambda x: x[2] or "", reverse=True)

        all_docs = await process_documents(
            url_html_date_tuples, output_dir, request.min_mtld
        )
        sorted_doc_results, context, context_info = await search_and_group_documents(
            request.query, all_docs, request.embed_model,
            request.llm_model, request.top_k, output_dir
        )
        response = generate_response(
            request.query, context, request.llm_model, mlx, output_dir
        )
        # evaluate_results(
        #     request.query, context, response, request.llm_model, output_dir
        # )

        headers_stats = get_header_stats(context)

        return SearchResponse(
            query=request.query,
            context=context,
            response=response,
            context_info=context_info,
            headers_stats=headers_stats
        )
    except Exception as e:
        logger.error(f"Search endpoint error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search processing failed: {str(e)}"
        )

# Test endpoint for health check


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
