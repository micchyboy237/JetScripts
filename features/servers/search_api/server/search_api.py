from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Iterator, Union, Tuple
import asyncio
import os
import json
from jet.data.utils import generate_key, generate_unique_hash
from jet.logger import logger
from jet.vectors.document_types import HeaderDocument
from jet.llm.mlx.base import MLX
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.utils.url_utils import rerank_urls_bm25_plus
from jet.search.searxng import SearchResult
from jet.scrapers.hrequests_utils import scrape_urls
from jet.scrapers.utils import scrape_published_date, scrape_links
from jet.code.splitter_markdown_utils import get_md_header_docs
from jet.file.utils import save_file
from jet.models.tokenizer.base import count_tokens
from jet.models.tasks.hybrid_search_docs_with_bm25 import search_docs
from jet.wordnet.analyzers.text_analysis import analyze_readability
from .search_models import (
    SearchRequest,
    SearchResponse,
    StreamResponseChunk
)
from .search_and_rerank import (
    initialize_output_directory,
    initialize_search_components,
    generate_response as generate_response_async,
    evaluate_results as evaluate_results_async,
    get_header_stats,
    fetch_search_results,
    process_search_results,
    process_documents,
    search_and_group_documents,
    StreamedStep,
)

app = FastAPI(title="Search API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def setup_output_directory(query: str, llm_model: str, embed_model: str, script_path: str) -> str:
    """Shared helper to initialize output directory."""
    shared_dir = generate_key(query, llm_model, embed_model)
    sub_dir = generate_unique_hash()
    return initialize_output_directory(script_path, f"{shared_dir}/{sub_dir}")


async def stream_data(request: SearchRequest) -> Iterator[bytes]:
    """Stream search steps as NDJSON chunks for POST endpoint."""
    logger.debug(f"Starting streaming data for query: {request.query}")

    async def step_to_json(step: StreamedStep) -> bytes:
        chunk = json.dumps(StreamResponseChunk(**step).dict()) + "\n"
        logger.debug(f"Streaming chunk: {chunk.strip()}")
        return chunk.encode("utf-8")

    yield await step_to_json({
        "step_title": "Starting Search Process",
        "step_result": {"query": request.query}
    })

    try:
        output_dir = setup_output_directory(
            request.query, request.llm_model, request.embed_model, __file__)
        logger.debug(f"Output directory initialized: {output_dir}")
        yield await step_to_json({
            "step_title": "Output Directory Ready",
            "step_result": {"directory": output_dir}
        })

        mlx, _ = initialize_search_components(
            request.llm_model, request.embed_model, request.seed)
        logger.debug("Search components initialized")
        yield await step_to_json({
            "step_title": "Search Components Initialized",
            "step_result": {"message": "Model and tokenizer loaded"}
        })

        browser_search_results = await fetch_search_results(request.query, output_dir, request.use_cache)
        logger.debug(f"Fetched {len(browser_search_results)} search results")
        yield await step_to_json({
            "step_title": "Search Results Fetched",
            "step_result": {"count": len(browser_search_results)}
        })

        url_html_date_tuples = []
        async for result in process_search_results(browser_search_results, request.query, output_dir):
            if isinstance(result, dict):  # StreamedStep
                yield await step_to_json(result)
            else:  # List[Tuple[str, str, Optional[str]]]
                url_html_date_tuples = result
        url_html_date_tuples.sort(key=lambda x: x[2] or "", reverse=True)
        logger.debug(
            f"Processed {len(url_html_date_tuples)} URL-HTML-date tuples")
        yield await step_to_json({
            "step_title": "Search Results Processed",
            "step_result": {"count": len(url_html_date_tuples)}
        })

        all_docs = await process_documents(url_html_date_tuples, output_dir, request.min_mtld)
        logger.debug(f"Processed {len(all_docs)} documents")
        yield await step_to_json({
            "step_title": "Documents Processed",
            "step_result": {"count": len(all_docs)}
        })

        sorted_doc_results, context, context_info = None, None, None
        async for result in search_and_group_documents(
            request.query, all_docs, request.embed_model, request.llm_model, request.top_k, output_dir
        ):
            if isinstance(result, dict):  # StreamedStep
                yield await step_to_json(result)
            else:  # Tuple[List[dict], str, ContextInfo]
                sorted_doc_results, context, context_info = result
        logger.debug(
            f"Search completed with {len(context.split('\n\n'))} context segments")
        yield await step_to_json({
            "step_title": "Document Search and Grouping Completed",
            "step_result": {"context_segments": len(context.split("\n\n"))}
        })

        response = ""
        async for step in generate_response_async(request.query, context, request.llm_model, mlx, output_dir):
            if step["step_title"] == "Response Complete":
                response = step["step_result"]["full_response"]
            logger.debug(f"Yielding response step: {step['step_title']}")
            yield await step_to_json(step)

        async for step in evaluate_results_async(request.query, context, response, request.llm_model, output_dir):
            logger.debug(f"Yielding evaluation step: {step['step_title']}")
            yield await step_to_json(step)

        headers_stats = get_header_stats(context)
        logger.debug("Finalizing search results")
        yield await step_to_json({
            "step_title": "Search Results Finalized",
            "step_result": {
                "query": request.query,
                "context": context,
                "response": response,
                "context_info": context_info,
                "headers_stats": headers_stats
            }
        })

    except Exception as e:
        logger.error(f"Stream search error: {str(e)}")
        yield await step_to_json({
            "step_title": "Search Error",
            "step_result": {"error": str(e)}
        })


@app.post("/search")
async def search_endpoint(request: SearchRequest):
    logger.info(
        f"POST /search received: query={request.query}, stream={request.stream}")
    try:
        if request.stream:
            return StreamingResponse(
                stream_data(request),
                media_type="application/x-ndjson"
            )
        output_dir = setup_output_directory(
            request.query, request.llm_model, request.embed_model, __file__)
        logger.debug(f"Output directory initialized: {output_dir}")

        mlx, _ = initialize_search_components(
            request.llm_model, request.embed_model, request.seed)
        browser_search_results = await fetch_search_results(request.query, output_dir, request.use_cache)
        url_html_date_tuples = []
        async for result in process_search_results(browser_search_results, request.query, output_dir):
            if isinstance(result, dict):  # StreamedStep
                pass  # Discard streaming steps in non-streaming path
            else:  # List[Tuple[str, str, Optional[str]]]
                url_html_date_tuples = result
        url_html_date_tuples.sort(key=lambda x: x[2] or "", reverse=True)
        all_docs = await process_documents(url_html_date_tuples, output_dir, request.min_mtld)
        _, context, context_info = await search_and_group_documents(
            request.query, all_docs, request.embed_model, request.llm_model, request.top_k, output_dir
        )
        response = ""
        async for step in generate_response_async(request.query, context, request.llm_model, mlx, output_dir):
            if step["step_title"] == "Response Complete":
                response = step["step_result"]["full_response"]
        headers_stats = get_header_stats(context)
        return SearchResponse(
            query=request.query,
            context=context,
            response=response,
            context_info=context_info,
            headers_stats=headers_stats
        )
    except Exception as e:
        logger.error(f"POST /search error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search processing failed: {str(e)}"
        )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
