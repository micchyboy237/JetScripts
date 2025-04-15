import json
import shutil
from jet.code.splitter_markdown_utils import get_md_header_contents
from jet.data.base import convert_json_schema_to_model_instance, convert_json_schema_to_model_type, create_dynamic_model
from jet.features.search_and_chat import SYSTEM_QUERY_SCHEMA_DOCS, SEARCH_WEB_PROMPT_TEMPLATE, get_all_header_nodes, get_docs_from_html, get_header_tokens_and_update_metadata, get_nodes_parent_mapping, process_document, rerank_nodes
from jet.llm.models import OLLAMA_EMBED_MODELS, OLLAMA_MODEL_NAMES
from jet.llm.ollama.base import Ollama
from jet.logger import logger
from jet.scrapers.preprocessor import html_to_markdown
from jet.token.token_utils import get_model_max_tokens, group_nodes
from jet.transformers.formatters import format_json
from pydantic import BaseModel, create_model
from typing import Any, Dict, Optional, List, Type, Union
import os
from jet.file.utils import load_file, save_file
from jet.llm.prompt_templates.base import generate_browser_query_context_json_schema, generate_browser_query_json_schema, generate_json_schema_sample, generate_output_class
from jet.validation.json_schema_validator import schema_validate_json
from typing import Any, Dict

output_dir = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


if __name__ == "__main__":
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    llm_model = "llama3.1"
    embed_models: list[OLLAMA_EMBED_MODELS] = [
        "mxbai-embed-large",
        "paraphrase-multilingual",
    ]
    embed_model = embed_models[0]
    embed_model_max_tokens = get_model_max_tokens(embed_model)

    sub_chunk_size = 128
    sub_chunk_overlap = 40

    query = "Philippines tips for online selling 2025"

    html: str = load_file(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/generated/run_anime_scraper/query_philippines_tips_for_online_selling_2025/hqmanila_com/scraped_html.html")
    save_file(html, f"{output_dir}/doc.html")

    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    header_docs = get_docs_from_html(html)
    # for doc in header_docs:
    #     doc.set_content(doc.get_recursive_text())

    save_file("\n\n".join([doc.text for doc in header_docs]),
              f"{output_dir}/headers.md")

    # Generate output model structure
    json_schema = generate_browser_query_json_schema(query)
    save_file(json_schema, f"{output_dir}/json_schema.json")

    output_cls = create_dynamic_model(json_schema, nested=True)

    logger.orange(format_json(output_cls.model_json_schema()))

    shared_header_doc = header_docs[0]

    header_tokens = get_header_tokens_and_update_metadata(
        header_docs, embed_model)

    all_header_nodes = get_all_header_nodes(
        header_docs,
        header_tokens,
        query,
        llm_model,
        embed_models,
        sub_chunk_size,
        sub_chunk_overlap,
    )

    header_parent_map = get_nodes_parent_mapping(all_header_nodes, header_docs)

    # Remove first h1
    filtered_header_nodes = [
        node for node in all_header_nodes if node.metadata["doc_index"] != shared_header_doc.metadata["doc_index"]]
    # Rerank headers
    reranked_header_nodes = rerank_nodes(
        query, filtered_header_nodes, embed_models, header_parent_map)

    # Sort reranked results by doc index
    sorted_header_nodes = sorted(
        reranked_header_nodes, key=lambda node: node.metadata['doc_index'])
    # Split nodes into groups to prevent LLM max tokens issue
    grouped_header_nodes = group_nodes(sorted_header_nodes, llm_model)

    # First group only
    context_nodes = grouped_header_nodes[0]
    header_texts = [node.text for node in context_nodes]
    # Prepend shared context
    header_texts = [shared_header_doc.text] + header_texts
    context = "\n\n".join(header_texts)
    save_file(context, os.path.join(output_dir, f"context.md"))

    group_header_doc_indexes = [
        node.metadata["doc_index"] for node in context_nodes]
    reranked_group_nodes = [
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
    save_file(reranked_group_nodes, os.path.join(
        output_dir, f"reranked_context_nodes.json"))

    reranked_header_nodes = [
        {
            "doc": node.metadata["doc_index"] + 1,
            "rank": rank_idx + 1,
            "score": node.score,
            "text": node.text,
            "metadata": node.metadata,
        }
        for rank_idx, node in enumerate(reranked_header_nodes)
        if node.metadata["doc_index"] in group_header_doc_indexes
    ]
    save_file(reranked_header_nodes, os.path.join(
        output_dir, f"reranked_header_nodes.json"))

    # json_schema = generate_browser_query_context_json_schema(
    #     query, context, llm_model)
    # output_cls = convert_json_schema_to_model_type(json_schema)
    # json_schema_file = f"{output_dir}/browser_query_context_json_schema.json"
    # save_file(json_schema, json_schema_file)

    llm = Ollama(temperature=0.3, model=llm_model)
    response = llm.structured_predict(
        output_cls,
        prompt=SEARCH_WEB_PROMPT_TEMPLATE,
        model=llm_model,
        headers=context,
        instruction=SYSTEM_QUERY_SCHEMA_DOCS,
        query=query,
        schema=json.dumps(output_cls.model_json_schema(), indent=2),
    )
    save_file(response, os.path.join(
        output_dir, f"chat_response.json"))
