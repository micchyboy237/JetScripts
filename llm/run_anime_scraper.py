import os
import re
from urllib.parse import urlparse

from jet.features.scrape_search_chat import get_docs_from_html, get_nodes_from_docs, rerank_nodes, run_scrape_search_chat, validate_headers
from jet.file.utils import load_file, save_file
from jet.llm.ollama.base import Ollama
from jet.logger import logger
from jet.scrapers.utils import scrape_urls, search_data
from jet.token.token_utils import get_model_max_tokens, token_counter


def safe_path_from_url(url: str, output_dir: str) -> str:
    parsed = urlparse(url)

    # Sanitize host (remove port, replace . with _, remove other unsafe characters)
    host = parsed.hostname or 'unknown_host'
    safe_host = re.sub(r'\W+', '_', host)

    # Get last non-empty segment of path
    path_parts = [part for part in parsed.path.split('/') if part]
    last_path = path_parts[-1] if path_parts else 'root'

    # Final safe subdir
    sub_dir = os.path.join(safe_host, last_path, output_dir)

    return sub_dir


if __name__ == "__main__":
    # --- Inputs ---
    llm_model = "gemma3:4b"
    embed_models = [
        "mxbai-embed-large",
        "paraphrase-multilingual",
        "granite-embedding",
    ]
    eval_model = llm_model
    output_dir = f"/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/generated/{os.path.splitext(os.path.basename(__file__))[0]}"
    query = "What are the top 10 rom com anime today?"
    min_headers = 5

    max_model_tokens = get_model_max_tokens(llm_model)
    buffer = max_model_tokens * 0.75
    max_context_tokens = max_model_tokens - buffer

    # Search urls
    search_results = search_data(query)
    urls = [item["url"] for item in search_results]

    scraped_urls_results = scrape_urls(urls)
    for url, html in scraped_urls_results:
        sub_dir = safe_path_from_url(url, output_dir)

        result = run_scrape_search_chat(
            html,
            llm_model,
            embed_models,
            eval_model,
            sub_dir,
            query,
        )

        if result["search_eval"].passing:
            save_file({
                "query": result["query"],
                "results": result["search_nodes"]
            }, os.path.join(sub_dir, "top_nodes.json"))

            save_file(result["search_eval"], os.path.join(
                sub_dir, "eval_context_relevancy.json"))

            history = "\n\n".join([
                f"## Query\n\n{result["query"]}",
                f"## Context\n\n{result["context"]}",
                f"## Response\n\n{result["response"]}",
            ])
            save_file(history, os.path.join(sub_dir, "llm_chat_history.md"))
