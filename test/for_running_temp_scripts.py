import json
import shutil
from jet.code.splitter_markdown_utils import get_md_header_contents
from jet.data.base import convert_json_schema_to_model_instance, convert_json_schema_to_model_type
from jet.features.scrape_search_chat import SYSTEM_QUERY_SCHEMA_DOCS, SEARCH_WEB_PROMPT_TEMPLATE, get_all_header_nodes, get_docs_from_html, get_header_tokens_and_update_metadata, get_nodes_parent_mapping, process_document, rerank_nodes
from jet.llm.models import OLLAMA_EMBED_MODELS, OLLAMA_MODEL_NAMES
from jet.llm.ollama.base import Ollama
from jet.scrapers.preprocessor import html_to_markdown
from jet.token.token_utils import get_model_max_tokens, group_nodes
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

    data_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/prompts/generated/run_generate_browser_query_context_json_schema"

    result = extract_by_heading_hierarchy(sample_html)

    # Generate output model structure
    json_schema = load_file(os.path.join(data_dir, "json_schema.json"))
    output_cls = convert_json_schema_to_model_type(json_schema)

    context = load_file(os.path.join(data_dir, f"context.md"))

    query = "Philippines TikTok online seller for live selling registration steps 2025"

    llm = Ollama(temperature=0.3, model=llm_model)
    response = llm.structured_predict(
        output_cls,
        prompt=SEARCH_WEB_PROMPT_TEMPLATE,
        model=llm_model,
        context=context,
        system=SYSTEM_QUERY_SCHEMA_DOCS,
        query=query,
    )
    save_file(response, os.path.join(
        output_dir, f"chat_response.json"))
