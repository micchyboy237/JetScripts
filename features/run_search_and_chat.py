import os
from typing import Any, Dict, List, Tuple

from jet._token.token_utils import group_nodes
from jet.features.search_and_chat import (
    compare_html_results,
    get_docs_from_html,
    rerank_nodes,
    search_and_filter_data,
)
from jet.file.utils import save_file
from jet.llm.models import OLLAMA_EMBED_MODELS
from jet.llm.ollama.base import Ollama
from jet.scrapers.utils import safe_path_from_url
from llama_index.core.schema import Document as BaseDocument
from llama_index.core.schema import NodeWithScore

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "generated",
    os.path.splitext(os.path.basename(__file__))[0],
)


def process_and_compare_htmls(
    query: str,
    selected_html: List[Tuple[str, str]],
    embed_models: List[OLLAMA_EMBED_MODELS],
    output_dir: str,
) -> Tuple[
    List[BaseDocument],
    List[Tuple[str, str, str]],
    List[Dict[str, Any]],
    List[NodeWithScore],
]:
    """
    Process HTMLs, rerank documents, and compare results to find the best-matching HTML.

    Returns:
        header_docs: The list of header documents from the top result.
        html_results: The full HTML results for potential further use.
        query_scores: Scores from reranking top document headers.
        context_nodes: Token-grouped top-ranked nodes for context usage.
    """
    html_results = []

    header_docs_for_all = {}

    sub_dir = os.path.join(output_dir, "searched_html")

    for url, html in selected_html:
        output_dir_url = safe_path_from_url(url, sub_dir)
        os.makedirs(output_dir_url, exist_ok=True)

        header_docs = get_docs_from_html(html)
        query_scores, reranked_all_nodes = rerank_nodes(
            query, header_docs, embed_models
        )

        # Save intermediate files
        save_file(
            {"query": query, "results": query_scores},
            os.path.join(output_dir_url, "query_scores.json"),
        )

        save_file(
            {
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
                ],
            },
            os.path.join(output_dir_url, "reranked_all_nodes.json"),
        )

        html_results.append((url, output_dir_url, html))
        header_docs_for_all[url] = (header_docs, query_scores, reranked_all_nodes)

    # Compare and get top result
    comparison_results = compare_html_results(query, html_results, top_n=1)

    top_result = comparison_results[0]
    top_url = top_result["url"]

    header_docs, query_scores, reranked_all_nodes = header_docs_for_all[top_url]

    # Group nodes and extract context
    sorted_reranked_nodes = sorted(
        reranked_all_nodes, key=lambda node: node.metadata["doc_index"]
    )
    grouped_reranked_nodes = group_nodes(sorted_reranked_nodes, "llama3.1")
    context_nodes = grouped_reranked_nodes[0] if grouped_reranked_nodes else []

    return header_docs, html_results, query_scores, context_nodes


if __name__ == "__main__":
    output_dir = OUTPUT_DIR
    query = "Philippines tips for online selling 2025"

    llm_model = "llama3.1"
    embed_models: list[OLLAMA_EMBED_MODELS] = [
        "mxbai-embed-large",
        "paraphrase-multilingual",
    ]

    search_results, selected_html = search_and_filter_data(query)
    search_results_file = f"{output_dir}/search_results.json"
    save_file(search_results, search_results_file)

    header_docs, html_results, query_scores, context_nodes = process_and_compare_htmls(
        query, selected_html, embed_models, output_dir
    )

    # Save markdown content from headers
    header_texts = [doc.text for doc in header_docs]
    headers_text = "\n\n".join(header_texts)
    save_file(headers_text, os.path.join(output_dir, "docs.md"))

    # Save query scores
    save_file(
        {"query": query, "results": query_scores},
        os.path.join(output_dir, "query_scores.json"),
    )

    # Save context node details
    group_header_doc_indexes = [node.metadata["doc_index"] for node in context_nodes]

    save_file(
        {
            "query": query,
            "results": [
                {
                    "doc": node.metadata["doc_index"] + 1,
                    "rank": rank_idx + 1,
                    "score": node.score,
                    "text": node.text,
                    "metadata": node.metadata,
                }
                for rank_idx, node in enumerate(context_nodes)
                if node.metadata["doc_index"] in group_header_doc_indexes
            ],
        },
        os.path.join(output_dir, "reranked_context_nodes.json"),
    )

    # Save context markdown
    context = "\n\n".join([node.text for node in context_nodes])
    save_file(context, os.path.join(output_dir, "context.md"))

    # Run LLM response
    llm = Ollama(temperature=0.3, model=llm_model)
    response = llm.chat(
        query,
        context=context,
        model=llm_model,
    )

    save_file(
        {"query": query, "response": response},
        os.path.join(output_dir, "chat_response.json"),
    )
