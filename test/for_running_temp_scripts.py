import json
import shutil
from jet.code.splitter_markdown_utils import extract_md_header_contents, get_md_header_contents
from jet.data.base import convert_json_schema_to_model_instance, convert_json_schema_to_model_type, extract_titles_descriptions
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
from jet.wordnet.similarity import query_similarity_scores
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
    schema_contexts = [
        f"Title: {item["title"]}, Description: {item["description"]}" for item in schema_results]

    header_docs = get_docs_from_html(html)

    queries = [query, *schema_contexts]
    query_scores = Document.rerank_documents(
        queries, header_docs, embed_models)

    logger.success(format_json(query_scores))

    save_file({"queries": queries, "results": query_scores}, os.path.join(
        output_dir, f"reranked_docs.json"))
