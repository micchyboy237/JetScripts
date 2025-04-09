import os
import re
from urllib.parse import urlparse

from jet.features.scrape_search_chat import get_docs_from_html, get_nodes_from_docs, rerank_nodes, run_scrape_search_chat, validate_headers
from jet.file.utils import load_file, save_file
from jet.llm.ollama.base import Ollama
from jet.logger import logger
from jet.scrapers.browser.formatters import construct_browser_query
from jet.scrapers.utils import safe_path_from_url, scrape_urls, search_data
from jet.token.token_utils import get_model_max_tokens, token_counter


if __name__ == "__main__":
    # --- Inputs ---
    # llm_model = "gemma3:4b"
    llm_model = "mistral"
    embed_models = [
        "mxbai-embed-large",
        "paraphrase-multilingual",
        "granite-embedding",
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
    min_headers = 5

    max_model_tokens = get_model_max_tokens(llm_model)
    buffer = max_model_tokens * 0.75
    max_context_tokens = max_model_tokens - buffer

    # Search urls
    search_results = search_data(query)
    urls = [item["url"] for item in search_results]

    scraped_urls_results = scrape_urls(urls)
    for url, html in scraped_urls_results:
        logger.info(f"Scraping url: {url}")
        sub_dir = safe_path_from_url(url, output_dir)

        html_file = f"{sub_dir}/scraped_html.html"
        save_file(html, html_file)

        header_docs = get_docs_from_html(html)
        headers_texts = [header.text for header in header_docs]
        headers_md = "\n\n".join(headers_texts)
        headers_md_file = f"{sub_dir}/headers.md"
        save_file(headers_md, headers_md_file)

        result = run_scrape_search_chat(
            html,
            llm_model,
            embed_models,
            eval_model,
            sub_dir,
            query,
        )

        if not result:
            continue

        save_file(result["docs"], os.path.join(sub_dir, "documents.json"))

        save_file({
            "query": result["query"],
            "results": result["search_nodes"]
        }, os.path.join(sub_dir, "top_nodes.json"))

        # save_file(result["search_eval"], os.path.join(
        #     sub_dir, "eval_context_relevancy.json"))

        history = "\n\n".join([
            f"## Query\n\n{result["query"]}",
            f"## Context\n\n{result["context"]}",
            f"## Response\n\n{result["response"]}",
        ])
        save_file(history, os.path.join(sub_dir, "llm_chat_history.md"))

        # if result["search_eval"].passing:
        #     break
