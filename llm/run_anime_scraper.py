import os
from typing import Dict, Optional, TypedDict

from jet.features.scrape_search_chat import DocumentTokensExceedsError, EvalContextError, run_scrape_search_chat
from jet.file.utils import save_file
from jet.llm.evaluators.context_relevancy_evaluator import evaluate_context_relevancy
from jet.llm.models import OLLAMA_EMBED_MODELS
from jet.data.base import create_dynamic_model
from jet.llm.prompt_templates.base import generate_browser_query_json_schema
from jet.logger import logger
from jet.scrapers.utils import safe_path_from_url, scrape_urls, search_data, validate_headers
from jet.transformers.formatters import format_json
from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel, Field
from tqdm import tqdm


if __name__ == "__main__":
    # --- Inputs ---
    llm_model = "llama3.1"
    embed_models: list[OLLAMA_EMBED_MODELS] = [
        "paraphrase-multilingual",
        # "mxbai-embed-large",
    ]
    eval_model = llm_model
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    # query = "Top otome villainess anime 2025"
    query = "Philippines TikTok online seller for live selling registration steps 2025"
    min_header_count = 5

    max_search_depth = 2

    query_dir = "query_" + query.lower().replace(' ', '_')
    output_dir = os.path.join(output_dir, query_dir)

    # Generate output model structure
    json_schema = generate_browser_query_json_schema(query)
    save_file(json_schema, f"{output_dir}/generated_json_schema.json")
    # Create the dynamic model based on the JSON schema
    DynamicModel = create_dynamic_model(json_schema)
    output_cls = DynamicModel

    logger.log("Schema:", format_json(
        output_cls.model_json_schema()), colors=["GRAY", "ORANGE"])

    # Search urls
    search_results = search_data(query)
    urls = [item["url"] for item in search_results]

    scraped_urls_results = scrape_urls(
        urls, max_depth=max_search_depth, query=query)
    pbar = tqdm(total=len(urls))
    for url, html in scraped_urls_results:
        pbar.set_description(f"URL: {url}")
        logger.orange(f"Scraping url: {url}")

        if not validate_headers(html, min_count=min_header_count):
            logger.warning(
                f"Skipping url: {url} due to header count < {min_header_count}")
            continue

        sub_dir = safe_path_from_url(url, output_dir)
        results_file = f"{sub_dir}/results.json"
        eval_context_result_file = f"{sub_dir}/eval_context_result.json"

        if os.path.exists(results_file):
            logger.warning(
                f"Skipping url: {url} as results already exist in {results_file}")
            continue

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

        class RerankedNodes(TypedDict):
            group: int
            tokens: int
            nodes: list[NodeWithScore]

        contexts: list[str] = []

        reranked_nodes: list[RerankedNodes] = []
        reranked_nodes_dict = {
            "query": query,
            "data": reranked_nodes,
        }

        class Results(TypedDict):
            group: int
            tokens: int
            results: Dict

        results: list[Results] = []
        results_dict = {
            "query": query,
            "url": url,
            "data": results
        }

        try:
            for response in response_generator:
                group = response["group"]

                context_tokens = response["context_tokens"]
                context: str = response["context"]
                reranked_nodes.append(
                    {"group": group, "tokens": context_tokens, "nodes": response["reranked_nodes"]})

                response_obj = response["response"]
                response_tokens = response["response_tokens"]
                eval_result = response["eval_result"]
                results.append(
                    {"group": group, "tokens": response_tokens, "results": response_obj})

                contexts.append(f"<!-- Group {group} -->\n\n{context}")
                save_file("\n\n".join(contexts),
                          os.path.join(sub_dir, f"context.md"))
                save_file(reranked_nodes_dict, os.path.join(
                    sub_dir, f"reranked_nodes.json"))
                save_file(results_dict, results_file)
                save_file(eval_result, eval_context_result_file)
        except DocumentTokensExceedsError as e:
            logger.error(
                f"DocumentTokensExceedsError:\n{e}\n\nMetadata:\n{format_json(e.metadata)}")
            continue
        except EvalContextError as e:
            save_file(e.eval_result, os.path.join(
                sub_dir, eval_context_result_file))
            logger.error(
                f"EvalContextError:\n{e}\n\nEvaluation:\n{format_json(e.eval_result)}")
            continue

        pbar.update(1)
