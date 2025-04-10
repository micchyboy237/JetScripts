import os
import re
from typing import Optional
from urllib.parse import urlparse

from jet.features.scrape_search_chat import get_docs_from_html, get_nodes_from_docs, rerank_nodes, run_scrape_search_chat, validate_headers
from jet.file.utils import load_file, save_file
from jet.llm.models import OLLAMA_EMBED_MODELS
from jet.logger import logger
from jet.scrapers.browser.formatters import construct_browser_query
from jet.scrapers.utils import safe_path_from_url, scrape_urls, search_data
from jet.token.token_utils import get_model_max_tokens
from pydantic import BaseModel, Field


class Answer(BaseModel):
    title: str = Field(
        ..., description="The exact title of the anime, as it appears in the document.")
    document_number: int = Field(
        ..., description="The number of the document that includes this anime (e.g., 'Document number: 3').")
    release_year: Optional[int] = Field(
        description="The most recent known release year of the anime, if specified in the document.")


class QueryResponse(BaseModel):
    results: list[Answer] = Field(
        default_factory=list,
        description="List of relevant anime titles extracted from the documents, matching the user's query. Each entry includes the title, source document number, and release year (if known)."
    )


output_cls = QueryResponse

if __name__ == "__main__":
    # --- Inputs ---
    # llm_model = "gemma3:4b"
    llm_model = "mistral"
    embed_models: list[OLLAMA_EMBED_MODELS] = [
        "paraphrase-multilingual",
        # "mxbai-embed-large",
    ]
    eval_model = llm_model
    output_dir = f"/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/generated/{os.path.splitext(os.path.basename(__file__))[0]}"
    query = "Top otome villainess anime 2025"
    # query = construct_browser_query(
    #     search_terms="top 10 romantic comedy anime",
    #     include_sites=["myanimelist.net",
    #                    "anilist.co", "animenewsnetwork.com"],
    #     exclude_sites=["wikipedia.org", "imdb.com"],
    #     # after_date="2024-01-01",
    #     # before_date="2025-04-05"
    # )

    # Search urls
    search_results = search_data(query)
    urls = [item["url"] for item in search_results]

    scraped_urls_results = scrape_urls(urls)
    for url, html in scraped_urls_results:
        logger.info(f"Scraping url: {url}")
        sub_dir = safe_path_from_url(url, output_dir)

        html_file = f"{sub_dir}/scraped_html.html"
        save_file(html, html_file)

        response_generator = run_scrape_search_chat(
            html=html,
            query=query,
            output_cls=output_cls,
            llm_model=llm_model,
            embed_models=embed_models,
        )

        for response in response_generator:
            save_file(response["context"], os.path.join(
                output_dir, f"context_nodes_{response["group"]}.md"))
            save_file(
                response["response"], f"{output_dir}/results_{response["group"]}.json")
