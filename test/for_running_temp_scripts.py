import json
import shutil
from jet.code.splitter_markdown_utils import extract_md_header_contents, get_md_header_contents
from jet.data.base import convert_json_schema_to_model_instance, convert_json_schema_to_model_type, create_dynamic_model, extract_titles_descriptions
from jet.features.scrape_search_chat import SYSTEM_QUERY_SCHEMA_DOCS, SEARCH_WEB_PROMPT_TEMPLATE, Document, get_all_header_nodes, get_docs_from_html, get_docs_from_html, get_header_tokens_and_update_metadata, get_nodes_parent_mapping, process_document, rerank_nodes
from jet.llm.models import OLLAMA_EMBED_MODELS, OLLAMA_MODEL_NAMES
from jet.llm.ollama.base import Ollama
from jet.logger import logger
from jet.scrapers.preprocessor import html_to_markdown
from jet.scrapers.utils import extract_internal_links, extract_title_and_metadata, scrape_links
from jet.token.token_utils import get_model_max_tokens, group_nodes
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from jet.utils.doc_utils import get_recursive_text
from jet.wordnet.similarity import query_similarity_scores, SimilarityResult
from pydantic import BaseModel, create_model
from typing import Any, Dict, Optional, List, Type, Union
import os
from jet.file.utils import load_file, save_file
from jet.llm.prompt_templates.base import generate_browser_query_context_json_schema, generate_browser_query_json_schema, generate_json_schema_sample, generate_output_class
from jet.validation.json_schema_validator import schema_validate_json
from typing import Any, Dict
from llama_index.core.schema import Document as BaseDocument, NodeRelationship, NodeWithScore, RelatedNodeInfo, TextNode

output_dir = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


if __name__ == "__main__":
    query = "Philippines tips for online selling 2025"

    llm_model = "llama3.1"
    embed_models: list[OLLAMA_EMBED_MODELS] = [
        "mxbai-embed-large",
        "paraphrase-multilingual",
    ]

    html: str = load_file(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/generated/run_anime_scraper/query_philippines_tips_for_online_selling_2025/hqmanila_com/scraped_html.html")
    json_schema = load_file(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/generated/run_anime_scraper/query_philippines_tips_for_online_selling_2025/generated_json_schema.json")

    schema_results = extract_titles_descriptions(json_schema)
    schema_contexts = [item["description"]
                       for item in schema_results if item["description"]]

    header_docs = get_docs_from_html(html)
    headers = [doc.text for doc in header_docs]
    shared_header_doc = header_docs[0]

    context = "\n\n".join(headers)
    save_file(context, os.path.join(output_dir, f"context.md"))

    # rerank_queries = [query]
    rerank_queries = [query, *schema_contexts]

    # header_docs_dict: dict[str, Document] = {
    #     doc.node_id: doc for doc in header_docs}
    # query_scores = Document.rerank_documents(
    #     rerank_queries, header_docs, embed_models)
    # reranked_docs: list[Dict] = [
    #     {
    #         "doc": header_docs_dict[item["id"]].metadata["doc_index"],
    #         **item,
    #         "text": header_docs_dict[item["id"]].text,
    #     }
    #     for item in query_scores
    # ]

    # Remove first h1
    filtered_header_docs = [
        doc for doc in header_docs if doc.metadata["doc_index"] != shared_header_doc.metadata["doc_index"]]
    # Rerank headers
    reranked_header_nodes = rerank_nodes(
        rerank_queries, filtered_header_docs, embed_models)
    all_header_doc_indexes = [
        node.metadata["doc_index"] for node in reranked_header_nodes]
    save_file({
        "query": rerank_queries,
        "results": [
            {
                "doc": node.metadata["doc_index"] + 1,
                "rank": rank_idx + 1,
                "score": node.score,
                "text": node.text,
                "metadata": node.metadata,
            }
            for rank_idx, node in enumerate(reranked_header_nodes)
            if node.metadata["doc_index"] in all_header_doc_indexes
        ]
    }, os.path.join(output_dir, f"reranked_all_nodes.json"))
    # Sort reranked results by doc index
    sorted_header_nodes = sorted(
        reranked_header_nodes, key=lambda node: node.metadata['doc_index'])
    # Split nodes into groups to prevent LLM max tokens issue
    grouped_header_nodes = group_nodes(sorted_header_nodes, llm_model)

    # First group only
    context_nodes = grouped_header_nodes[0]

    group_header_doc_indexes = [
        node.metadata["doc_index"] for node in context_nodes]
    save_file({
        "query": rerank_queries,
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
        ]
    }, os.path.join(output_dir, f"reranked_context_nodes.json"))

    output_cls = create_dynamic_model(json_schema, nested=True)
    llm = Ollama(temperature=0.3, model=llm_model)
    response = llm.chat(
        query,
        # prompt=SEARCH_WEB_PROMPT_TEMPLATE,
        model=llm_model,
        # headers=context,
        context=context,
        # instruction=SYSTEM_QUERY_SCHEMA_DOCS,
        format=output_cls.model_json_schema(),
    )
    save_file({"query": query, "response": response}, os.path.join(
        output_dir, f"chat_response.json"))
