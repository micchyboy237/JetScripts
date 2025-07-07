import argparse
from collections import defaultdict
import os
import re
import shutil
import string
from typing import List, Optional
import asyncio
from jet.code.html_utils import preprocess_html
from jet.code.markdown_utils import analyze_markdown, parse_markdown
from jet.data.header_docs import HeaderDocs
from jet.data.header_types import NodeWithScore
from jet.data.header_utils._prepare_for_rag import prepare_for_rag
from jet.data.header_utils._search_headers import search_headers
from jet.models.embeddings.chunking import chunk_headers_by_hierarchy
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.logger import logger
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.scrapers.hrequests_utils import scrape_urls
from jet.code.markdown_utils import convert_html_to_markdown
from jet.file.utils import save_file
from jet.models.tokenizer.base import count_tokens, get_tokenizer_fn
from jet.scrapers.utils import scrape_links, search_data
from jet.models.tasks.hybrid_search_docs_with_bm25 import search_docs
from jet.llm.evaluators.context_relevancy_evaluator import evaluate_context_relevancy
from jet.llm.evaluators.response_relevancy_evaluator import evaluate_response_relevancy
from jet.llm.mlx.helpers.base import get_system_date_prompt
from jet.llm.mlx.base import MLX
from jet.search.searxng import SearchResult as BrowserSearchResult

PROMPT_TEMPLATE = """\
Context information is below.
---------------------
{context}
---------------------

Given the context information, answer the query.

Query: {query}
"""


def format_sub_dir(text: str) -> str:
    return text.lower().strip('.,!?').replace(' ', '_').replace('.', '_').replace(',', '_').replace('!', '_').replace('?', '_').strip()


def initialize_output_directory(script_path: str, query: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(script_path))
    output_dir = os.path.join(script_dir, "generated", os.path.splitext(
        os.path.basename(script_path))[0])
    query_sub_dir = format_sub_dir(query)
    output_dir = os.path.join(output_dir, query_sub_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def format_sub_url_dir(url: str) -> str:
    clean_url = re.sub(r'^(https?://|www\.)|(\?.*)', '', url)
    trans_table = str.maketrans({p: '_' for p in string.punctuation})
    formatted = clean_url.translate(trans_table).lower()
    formatted = re.sub(r'_+', '_', formatted)
    return formatted.strip('_')


def initialize_search_components(
    llm_model: LLMModelType,
    embed_model: EmbedModelType,
    seed: int
) -> tuple[MLX, callable]:
    mlx = MLXModelRegistry.load_model(llm_model, seed=seed)
    tokenize = get_tokenizer_fn(embed_model)
    return mlx, tokenize


async def fetch_search_results(query: str, output_dir: str, use_cache: bool = False) -> List[BrowserSearchResult]:
    browser_search_results = search_data(query, use_cache=use_cache)
    save_file(
        {"query": query, "count": len(
            browser_search_results), "results": browser_search_results},
        os.path.join(output_dir, "browser_search_results.json")
    )
    return browser_search_results


async def process_search_results(
    browser_search_results: List[BrowserSearchResult],
    query: str,
    output_dir: str,
    top_n: int = 10,
    embed_model: EmbedModelType = "all-MiniLM-L6-v2",
    chunk_size: int = 200,
    chunk_overlap: int = 40,
    max_length: int = 2000,
) -> List[NodeWithScore]:
    selected_urls = [item["url"] for item in browser_search_results[:top_n]]
    SentenceTransformerRegistry.load_model(embed_model)
    tokenizer = SentenceTransformerRegistry.get_tokenizer()
    all_links = []
    all_search_results: List[NodeWithScore] = []

    async for url, status, html in scrape_urls(selected_urls, num_parallel=top_n, limit=top_n, show_progress=True):
        if status == "completed" and html:
            links = set(scrape_links(html, url))
            links = [link for link in links if (
                link != url if isinstance(link, str) else link["url"] != url)]
            all_links.extend(links)

            sub_url_dir = format_sub_url_dir(url)
            sub_output_dir = os.path.join(output_dir, "pages", sub_url_dir)
            os.makedirs(sub_output_dir, exist_ok=True)

            html = preprocess_html(html)
            save_file(html, f"{sub_output_dir}/page.html")

            doc_markdown = convert_html_to_markdown(html)
            save_file(doc_markdown, f"{sub_output_dir}/md_content.md")

            tokens = parse_markdown(doc_markdown, ignore_links=True)
            header_docs = HeaderDocs.from_tokens(tokens)
            header_docs.calculate_num_tokens(embed_model)
            all_nodes = header_docs.as_nodes()

            vector_store = prepare_for_rag(
                all_nodes, model=embed_model, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            search_results = search_headers(
                query, vector_store, top_k=None, threshold=0.7)
            for result in search_results:
                result.metadata.update({"source_url": url})
            all_search_results.extend(search_results)

    all_links = list(set(all_links))
    all_links = [link for link in all_links if (
        link not in selected_urls if isinstance(link, str) else link["url"] not in selected_urls)]
    save_file(all_links, os.path.join(output_dir, "links.json"))

    sorted_search_results = sorted(
        all_search_results, key=lambda x: x.score, reverse=True)

    filtered_search_results = []
    total_tokens = 0
    for result in sorted_search_results:
        num_tokens = getattr(result, "num_tokens",
                             result.metadata.get("num_tokens", 0))
        if total_tokens + num_tokens > max_length:
            break
        filtered_search_results.append(result)
        total_tokens += num_tokens

    return filtered_search_results


def generate_response(
    query: str,
    context: str,
    mlx: MLX,
    output_dir: str
) -> str:
    prompt = PROMPT_TEMPLATE.format(query=query, context=context)
    response = ""
    for chunk in mlx.stream_chat(
        prompt,
        system_prompt=get_system_date_prompt(),
        temperature=0.7,
        max_tokens=10000
    ):
        response += chunk["choices"][0]["message"]["content"]
    save_file(response, os.path.join(output_dir, "response.md"))
    save_file(
        {"query": query, "context": context, "response": response},
        os.path.join(output_dir, "chat_result.json")
    )
    return response


def evaluate_results(
    query: str,
    context: str,
    response: str,
    llm_model: LLMModelType,
    output_dir: str
) -> None:
    os.makedirs(os.path.join(output_dir, "eval"), exist_ok=True)
    eval_context_result = evaluate_context_relevancy(query, context, llm_model)
    save_file(eval_context_result, os.path.join(
        output_dir, "eval", "evaluate_context_relevance_result.json"))
    eval_response_result = evaluate_response_relevancy(
        query, response, llm_model)
    save_file(eval_response_result, os.path.join(
        output_dir, "eval", "evaluate_response_relevance_result.json"))


def group_search_results_by_source_url_for_context(search_results: List[NodeWithScore]) -> str:
    grouped = defaultdict(list)
    for node in search_results:
        url = node.metadata.get("source_url", "Unknown Source")
        grouped[url].append(node)

    context_blocks = []
    for url, nodes in grouped.items():
        block = f"<!-- Source: {url} -->\n\n"
        nodes = sorted(nodes, key=lambda n: (
            getattr(n, "doc_index", 0), getattr(n, "chunk_index", 0)))
        for node in nodes:
            block += node.get_text() + "\n\n"
        context_blocks.append(block.strip())

    return "\n\n".join(context_blocks)


async def main():
    args = parse_args()
    output_dir = initialize_output_directory(__file__, args.query)
    mlx, _ = initialize_search_components(
        args.llm_model, args.embed_model, args.seed)
    save_file(args.query, os.path.join(output_dir, "query.md"))
    browser_results = await fetch_search_results(args.query, output_dir, use_cache=args.use_cache)
    search_results = await process_search_results(browser_results, args.query, output_dir, max_length=1500)
    context_md = group_search_results_by_source_url_for_context(search_results)
    save_file(context_md, os.path.join(output_dir, "context.md"))
    response = generate_response(args.query, context_md, mlx, output_dir)
    evaluate_results(args.query, context_md, response,
                     args.llm_model, output_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run semantic search and processing pipeline.")
    p.add_argument("-q", "--query", type=str,
                   default="Top isekai anime 2025.", help="Search query to process")
    p.add_argument("-k", "--top_k", type=int, default=None,
                   help="Number of top documents to consider")
    p.add_argument("-m", "--llm_model", type=str,
                   default="qwen3-1.7b-4bit", help="LLM model to use")
    p.add_argument("-e", "--embed_model", type=str,
                   default="all-MiniLM-L6-v2", help="Embedding model to use")
    p.add_argument("-min", "--min_tokens", type=int, default=50,
                   help="Maximum number of tokens for final context")
    p.add_argument("-max", "--max_tokens", type=int, default=2000,
                   help="Maximum number of tokens for final context")
    p.add_argument("-s", "--chunk_size", type=int, default=300,
                   help="Maximum number of tokens per context")
    p.add_argument("-c", "--use_cache", action="store_true",
                   default=True, help="Use cached search results if available")
    p.add_argument("--seed", type=int, default=45,
                   help="Random seed for reproducibility")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
