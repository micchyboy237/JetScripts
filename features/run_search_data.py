import asyncio
import shutil
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from jet.transformers.formatters import format_json
from jet.transformers.object import make_serializable
from jet.wordnet.similarity import compute_info
from jet.wordnet.wordnet_types import SimilarityResult
from pydantic import BaseModel
from typing import List, Dict, Any, AsyncGenerator, Optional, Tuple
import os
import json
from llama_index.core.schema import Document, NodeWithScore
from jet.features.search_and_chat import get_nodes_from_docs, search_and_filter_data
from jet.llm.models import OLLAMA_EMBED_MODELS
from jet.scrapers.utils import safe_path_from_url
from jet.llm.ollama.base import Ollama
from jet.features.search_and_chat import compare_html_results, get_docs_from_html, rerank_nodes, group_nodes
from llama_index.core.schema import TextNode
from jet.file.utils import save_file


async def stream_progress(event_type: str, message: Optional[str] = None, data: Any = None) -> str:
    event_data = {}
    if message is not None:
        event_data["message"] = message
    if data is not None:
        event_data["data"] = data
    sse_message = f"event: {event_type}\n"
    sse_message += f"data: {format_json(event_data)}\n\n"
    return sse_message


async def process_and_compare_htmls(
    query: str,
    url_html_tuples: List[Tuple[str, str]],
    embed_models: List[OLLAMA_EMBED_MODELS],
    output_dir: str
) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
    html_results = []
    header_docs_for_all = {}
    sub_dir = os.path.join(output_dir, "searched_html")
    # Reset searched html results
    if os.path.exists(sub_dir):
        shutil.rmtree(sub_dir)

    yield (await stream_progress("html_processing", "Starting HTML processing", {"total_urls": len(url_html_tuples)}), {})

    for idx, (url, html) in enumerate(url_html_tuples, 1):
        output_dir_url = safe_path_from_url(url, sub_dir)
        os.makedirs(output_dir_url, exist_ok=True)

        yield (await stream_progress("html_processing", f"Processing HTML {idx}/{len(url_html_tuples)}: {url}"), {})

        # ✅ Save raw HTML
        save_file(html, os.path.join(output_dir_url, "page.html"))

        header_docs = get_docs_from_html(html)
        save_file("\n\n".join([doc.text for doc in header_docs]), os.path.join(
            output_dir_url, "docs.md"))
        save_file([
            {
                "doc": doc.metadata["doc_index"] + 1,
                "node_id": doc.node_id,
                "text": doc.text,
                "metadata": doc.metadata,
                "relationships": doc.relationships,
            }
            for doc in header_docs
        ], os.path.join(output_dir_url, "docs.json"))

        yield (await stream_progress("html_processing", f"Extracted header docs for {url}", {"header_docs_count": len(header_docs)}), {})

        query_scores, reranked_all_nodes = rerank_nodes(
            query, header_docs, embed_models)
        save_file({"url": url, "query": query, "info": compute_info(query_scores), "results": query_scores}, os.path.join(
            output_dir_url, "query_scores.json"))

        yield (
            await stream_progress(
                "html_processing",
                f"Reranked nodes for {url}",
                {"url": url, "query": query, "info": compute_info(
                    query_scores), "results": query_scores}
            ),
            {}
        )

        save_file({
            "url": url,
            "query": query,

            "results": [
                {
                    "doc": node.metadata["doc_index"] + 1,
                    "rank": rank_idx + 1,
                    "score": node.score,
                    "text": node.text,
                    "metadata": node.metadata,
                }
                for rank_idx, node in enumerate(reranked_all_nodes)
            ]
        }, os.path.join(output_dir_url, "reranked_all_nodes.json"))

        yield (
            await stream_progress(
                "html_processing",
                f"Processed reranked nodes for {url}",
                {"url": url, "query": query, "results": reranked_all_nodes}
            ),
            {}
        )

        html_results.append(
            {"url": url, "query": query, "results": query_scores})
        header_docs_for_all[url] = (
            header_docs, query_scores, reranked_all_nodes)

    yield (await stream_progress("html_processing", "Comparing HTML results"), {})
    comparison_results = compare_html_results(query, html_results)
    save_file(comparison_results, os.path.join(
        output_dir, "comparison_results.json"))

    if not comparison_results:
        yield (await stream_progress("error", "No comparison results available"), {})
        return

    top_result = comparison_results[0]

    yield (
        await stream_progress("html_processing", "Final HTML processing results", top_result),
        {"top_header_docs": header_docs_for_all[top_result["url"]]
            [0], "top_result": top_result}
    )


async def main():
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    llm_model = "llama3.1"
    embed_models: List[OLLAMA_EMBED_MODELS] = [
        "mxbai-embed-large", "paraphrase-multilingual"
    ]

    query = "Top isekai anime 2025"

    # Await async call
    search_filtered_result = await search_and_filter_data(query)
    url_html_tuples = search_filtered_result["url_html_tuples"]
    search_results = search_filtered_result["search_results"]

    save_file(search_results, os.path.join(output_dir, "search_results.json"))

    html_generator = process_and_compare_htmls(
        query, url_html_tuples, embed_models, output_dir)

    top_header_docs, html_results, query_scores, context_nodes = [], [], [], []

    async for sse_message, data in html_generator:
        if data and "top_result" in data:
            top_header_docs = data["top_header_docs"]
            top_result = data["top_result"]
            url = top_result["url"]
            query_scores: list[SimilarityResult] = top_result["results"]

    top_header_docs_dict: dict[str, Document] = {
        doc.node_id: doc for doc in top_header_docs}

    nodes_with_scores = [
        NodeWithScore(
            node=TextNode(
                node_id=result["id"],
                text=str(result["text"]),
                metadata=top_header_docs_dict[result["id"]].metadata
            ),
            score=float(result["score"])
        )
        for result in query_scores
    ]

    # Save top results
    query_scores_texts = "\n\n".join([node["text"] for node in query_scores])
    save_file(query_scores_texts, os.path.join(
        output_dir, "query_scores_texts.md"))

    save_file({"url": url, "query": query, "results": query_scores},
              os.path.join(output_dir, "all_query_scores.json"))

    # Chat LLM
    grouped_header_nodes = group_nodes(nodes_with_scores, llm_model)
    context_nodes = grouped_header_nodes[0]
    context = "\n\n".join([node.text for node in context_nodes])

    save_file(context, os.path.join(output_dir, "context.md"))

    llm = Ollama(temperature=0.3, model=llm_model)
    response = llm.chat(
        query,
        context=context,
        model=llm_model,
    )
    save_file(response, os.path.join(output_dir, "chat_response.md"))


if __name__ == "__main__":
    asyncio.run(main())
