import json
import os
import re
from typing import Optional, TypedDict
from urllib.parse import urlparse

from jet.features.scrape_search_chat import get_docs_from_html, get_nodes_from_docs, rerank_nodes, run_scrape_search_chat
from jet.file.utils import load_file, save_file
from jet.llm.models import OLLAMA_EMBED_MODELS
from jet.llm.prompt_templates.base import generate_json_schema
from jet.logger import logger
from jet.scrapers.browser.formatters import construct_browser_query
from jet.scrapers.utils import safe_path_from_url, scrape_urls, search_data, validate_headers
from jet.token.token_utils import get_model_max_tokens
from jet.utils.class_utils import class_to_string
from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel, Field
from tqdm import tqdm


if __name__ == "__main__":
    class Answer(BaseModel):
        title: str = Field(
            ..., description="The exact title of the anime, as it appears in the document.")
        document_number: int = Field(
            ..., description="The number of the document that includes this anime (e.g., 'Document number: 3').")
        release_year: Optional[int] = Field(
            ...,
            description="The most recent known release year of the anime, if specified in the document.")

    class QueryResponse(BaseModel):
        results: list[Answer] = Field(
            default_factory=list,
            description="List of relevant anime titles extracted from the documents, matching the user's query. Each entry includes the title, source document number, and release year (if known)."
        )

    output_cls = QueryResponse

    # --- Inputs ---
    llm_model = "llama3.1"
    embed_models: list[OLLAMA_EMBED_MODELS] = [
        "paraphrase-multilingual",
        # "mxbai-embed-large",
    ]
    eval_model = llm_model
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    query = "Top otome villainess anime 2025"

    def get_field_descriptions(model_fields: dict):
        # Extract field names and descriptions
        field_descriptions = [f"{idx + 1}. {field_info.description}" for idx,
                              (field_name, field_info) in enumerate(model_fields.items())]
        field_descriptions_str = "\n".join(field_descriptions)
        return field_descriptions_str

    # # Get the field names and descriptions for the QueryResponse and Answer models
    # answer_descriptions_str = get_field_descriptions(Answer.model_fields)
    # query_response_descriptions_str = get_field_descriptions(
    #     QueryResponse.model_fields)
    # field_descriptions_str = f"Answer Fields:\n{answer_descriptions_str}" + \
    #     "\n\n" + f"Query Response Fields:\n{query_response_descriptions_str}"

    # json_schema_context = f"Field Descriptions:\n{field_descriptions_str}\n\nQuery:\n{query}"
    # generated_json_schema = generate_json_schema(context=json_schema_context)
    # json_schema_file = f"{output_dir}/generated_json_schema.json"
    # save_file(generated_json_schema, json_schema_file)

    # query = construct_browser_query(
    #     search_terms="top 10 romantic comedy anime",
    #     include_sites=["myanimelist.net",
    #                    "anilist.co", "animenewsnetwork.com"],
    #     exclude_sites=["wikipedia.org", "imdb.com"],
    #     # after_date="2024-01-01",
    #     # before_date="2025-04-05"
    # )
    min_header_count = 5

    # Search urls
    search_results = search_data(query)
    urls = [item["url"] for item in search_results]

    scraped_urls_results = scrape_urls(urls)
    pbar = tqdm(total=len(urls))
    for url, html in scraped_urls_results:
        pbar.set_description(f"URL: {url}")

        if not validate_headers(html, min_count=min_header_count):
            logger.warning(
                f"Skipping url: {url} due to header count < {min_header_count}")
            continue

        logger.info(f"Scraping url: {url}")
        sub_dir = safe_path_from_url(url, output_dir)

        html_file = f"{sub_dir}/scraped_html.html"
        save_file(html, html_file)

        response_generator = run_scrape_search_chat(
            html=html,
            query=query,
            output_cls=output_cls,
            # output_cls=generated_json_schema,
            llm_model=llm_model,
            embed_models=embed_models,
        )

        class ContextNodes(TypedDict):
            group: int
            tokens: int
            nodes: list[NodeWithScore]

        contexts: list[str] = []

        context_nodes: list[ContextNodes] = []
        context_nodes_dict = {
            "query": query,
            "data": context_nodes,
        }

        class Results(TypedDict):
            group: int
            tokens: int
            results: list[Answer]

        results: list[Results] = []
        results_dict = {
            "query": query,
            "data": results
        }
        for response in response_generator:
            group = response["group"]

            context_tokens = response["context_tokens"]
            context: str = response["context"]
            context_nodes.append(
                {"group": group, "tokens": context_tokens, "nodes": response["context_nodes"]})

            response_obj: QueryResponse = response["response"]
            response_tokens = response["response_tokens"]
            results.append(
                {"group": group, "tokens": response_tokens, "results": response_obj.results})

            contexts.append(f"<!-- Group {group} -->\n\n{context}")
            save_file("\n\n".join(contexts),
                      os.path.join(sub_dir, f"context_nodes.md"))
            save_file(context_nodes_dict, os.path.join(
                sub_dir, f"context_nodes.json"))
            save_file(results_dict, f"{sub_dir}/results.json")

        pbar.update(1)
