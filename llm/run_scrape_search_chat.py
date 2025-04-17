import os

from jet.features.search_and_chat import run_scrape_search_chat
from jet.file.utils import load_file, save_file
from jet.llm.ollama.base import Ollama
from jet.logger import logger
from jet.token.token_utils import get_model_max_tokens, token_counter


if __name__ == "__main__":
    llm_model = "gemma3:4b"
    embed_models = [
        "mxbai-embed-large",
        "paraphrase-multilingual",
        "granite-embedding",
    ]
    eval_model = llm_model
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/generated/valid-ids-scraper/philippines_national_id_registration_tips_2025/scraped_html.html"
    output_dir = f"/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/generated/{os.path.splitext(os.path.basename(__file__))[0]}"
    query = "What are the steps in registering a National ID in the Philippines?"

    html = load_file(data_file)

    result = run_scrape_search_chat(
        html,
        llm_model,
        embed_models,
        eval_model,
        output_dir,
        query,
    )

    # if result["search_eval"].passing:
    save_file({
        "query": result["query"],
        "results": result["search_nodes"]
    }, os.path.join(output_dir, "top_nodes.json"))

    save_file(result["search_eval"], os.path.join(
        output_dir, "eval_context_relevancy.json"))

    history = "\n\n".join([
        f"## Query\n\n{result["query"]}",
        f"## Context\n\n{result["context"]}",
        f"## Response\n\n{result["response"]}",
    ])
    save_file(history, os.path.join(output_dir, "llm_chat_history.md"))
