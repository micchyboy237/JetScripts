import json
import shutil
from jet.code.splitter_markdown_utils import extract_md_header_contents, get_md_header_contents
from jet.data.base import convert_json_schema_to_model_instance, convert_json_schema_to_model_type
from jet.features.scrape_search_chat import SYSTEM_QUERY_SCHEMA_DOCS, SEARCH_WEB_PROMPT_TEMPLATE, get_all_header_nodes, get_docs_from_html, get_docs_from_html, get_header_tokens_and_update_metadata, get_nodes_parent_mapping, process_document, rerank_nodes
from jet.llm.models import OLLAMA_EMBED_MODELS, OLLAMA_MODEL_NAMES
from jet.llm.ollama.base import Ollama
from jet.logger import logger
from jet.scrapers.preprocessor import html_to_markdown
from jet.token.token_utils import get_model_max_tokens, group_nodes
from jet.transformers.formatters import format_json
from jet.utils.doc_utils import get_recursive_text
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
    html: str = load_file("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/generated/run_anime_scraper/query_philippines_tiktok_online_seller_for_live_selling_registration_steps_2025/sitegiant_ph/scraped_html.html")

    llm_model = "llama3.1"
    model_max_tokens = get_model_max_tokens(llm_model)

    llm = Ollama(llm_model)

    docs = get_docs_from_html(html)
    md_text = "\n\n".join(doc.text for doc in docs)
    header_contents = extract_md_header_contents(
        md_text, min_tokens_per_chunk=256, max_tokens_per_chunk=1000)

    content = docs[1].get_recursive_text()

    logger.success(content)
