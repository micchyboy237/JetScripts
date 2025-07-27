import asyncio
from collections import defaultdict
import os
import re
import shutil
import string
from typing import DefaultDict, List, Set
from jet.code.markdown_types.markdown_parsed_types import HeaderDoc
from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.code.markdown_utils._markdown_analyzer import analyze_markdown
from jet.code.markdown_utils._markdown_parser import base_parse_markdown, derive_by_header_hierarchy, parse_markdown
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.logger.config import colorize_log
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.models.tokenizer.base import get_tokenizer_fn, count_tokens
from jet.scrapers.hrequests_utils import scrape_urls
from jet.scrapers.utils import scrape_links, search_data
from jet.vectors.semantic_search.header_vector_search import HeaderSearchResult, search_headers


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

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


def format_sub_source_dir(source: str) -> str:
    """Format a source (URL or file path) into a directory name."""
    clean_source = re.sub(r'^(https?://|www\.)|(\?.*)', '', source)
    clean_source = clean_source.replace(os.sep, '_')
    trans_table = str.maketrans({p: '_' for p in string.punctuation})
    formatted = clean_source.translate(trans_table).lower()
    formatted = re.sub(r'_+', '_', formatted)
    return formatted.strip('_')


def group_results_by_source_for_llm_context(
    documents: List[HeaderSearchResult]
) -> str:
    def strip_hashtags(text: str) -> str:
        if text:
            return text.lstrip('#').strip()
        return text

    # Initialize tokenizer (using a default model assumption since llm_model is not provided)
    tokenizer = get_tokenizer_fn(
        "qwen3-1.7b-4bit", add_special_tokens=False, remove_pad_tokens=True)
    separator = "\n\n"
    separator_tokens = len(tokenizer.encode(separator))
    sorted_docs = sorted(
        documents, key=lambda x: x.get("score", 0), reverse=True
    )
    grouped_temp: DefaultDict[str,
                              List[HeaderSearchResult]] = defaultdict(list)
    seen_header_text: DefaultDict[str, Set[str]] = defaultdict(set)

    for doc in sorted_docs:
        text = doc.get("content", "")
        source = doc["metadata"].get("source", "Unknown Source")
        parent_header = doc.get("parent_header", "None")
        header = doc.get("header", None)
        level = doc["metadata"].get("level", 0)
        parent_level = doc["metadata"].get("parent_level", None)

        if not isinstance(text, str):
            logger.debug(
                f"Non-string content found for source: {source}, doc_index: {doc['metadata'].get('doc_index', 0)}, type: {type(text)}. Converting to string.")
            text = str(text) if text else ""

        doc_tokens = doc["metadata"].get(
            "num_tokens", len(tokenizer.encode(text)))
        header_tokens = 0

        if not grouped_temp[source]:
            header_tokens += len(tokenizer.encode(
                f"<!-- Source: {source} -->\n\n"))
            header_tokens += separator_tokens if grouped_temp else 0

        parent_header_key = strip_hashtags(
            parent_header) if parent_header and parent_header != "None" else None
        header_key = strip_hashtags(header) if header else None

        if header_key and header_key not in seen_header_text[source] and level >= 0:
            header_tokens += len(tokenizer.encode(f"{header}\n\n"))
            seen_header_text[source].add(header_key)

        grouped_temp[source].append(doc)

    context_blocks = []
    for source, docs in grouped_temp.items():
        block = f"<!-- Source: {source} -->\n\n"
        seen_header_text_in_block = set()
        docs = sorted(docs, key=lambda x: (
            x["metadata"].get("doc_index", 0),
            x["metadata"].get("chunk_idx", 0)
        ))
        for doc in docs:
            header = doc.get("header", None)
            parent_header = doc.get("parent_header", "None")
            text = doc.get("content", "")

            if not isinstance(text, str):
                logger.debug(
                    f"Non-string content in block for source: {source}, doc_index: {doc['metadata'].get('doc_index', 0)}, type: {type(text)}. Converting to string.")
                text = str(text) if text else ""

            doc_level = doc["metadata"].get(
                "level", 0) if doc["metadata"].get("level") is not None else 0
            parent_level = doc["metadata"].get("parent_level", None)
            parent_header_key = strip_hashtags(
                parent_header) if parent_header and parent_header != "None" else None
            header_key = strip_hashtags(header) if header else None

            has_matching_child = any(
                strip_hashtags(d.get("header", "")) == parent_header_key
                for d in docs
                if d.get("header") and d["metadata"].get("level", 0) >= 0
            )
            if parent_header_key and parent_level is not None and not has_matching_child and parent_header_key not in seen_header_text_in_block:
                parent_header_text = f"{parent_header}\n\n"
                block += parent_header_text
                seen_header_text_in_block.add(parent_header_key)

            if header_key and header_key not in seen_header_text_in_block and doc_level >= 0:
                subheader_text = f"{header}\n\n"
                block += subheader_text
                seen_header_text_in_block.add(header_key)

            block += text + "\n\n"

        block_tokens = len(tokenizer.encode(block))
        if block_tokens > len(tokenizer.encode(f"<!-- Source: {source} -->\n\n")):
            context_blocks.append(block.strip())
        else:
            logger.warning(
                f"Empty block for {source} after processing; skipping.")

    result = "\n\n".join(context_blocks)
    final_token_count = len(tokenizer.encode(result))
    contexts_data = {
        "query": documents[0].get("query", "") if documents else "",
        "count": len(documents),
        "total_tokens": sum(doc["metadata"].get("num_tokens", 0) for doc in documents),
        "results": [
            {
                "rank": doc.get("rank"),
                "score": doc.get("score"),
                "header": doc.get("header"),
                "content": doc.get("content"),
                "source": doc["metadata"].get("source"),
                "parent_header": doc.get("parent_header"),
                "parent_level": doc["metadata"].get("parent_level"),
                "level": doc["metadata"].get("level"),
                "doc_index": doc["metadata"].get("doc_index"),
                "chunk_idx": doc["metadata"].get("chunk_idx"),
                "num_tokens": doc["metadata"].get("num_tokens"),
                "header_content_similarity": doc["metadata"].get("header_content_similarity"),
                "headers_similarity": doc["metadata"].get("headers_similarity"),
                "content_similarity": doc["metadata"].get("content_similarity"),
            } for doc in documents
        ]
    }
    save_file(contexts_data, f"{OUTPUT_DIR}/contexts.json")
    logger.debug(
        f"Grouped context created with {final_token_count} tokens for {len(grouped_temp)} sources")
    return result


async def main(query):
    """Main function to demonstrate file search."""
    embed_model: EmbedModelType = "all-MiniLM-L6-v2"
    llm_model: LLMModelType = "qwen3-1.7b-4bit"
    max_tokens = 2000
    use_cache = True

    top_k = None
    threshold = 0.0  # Using default threshold
    chunk_size = 200
    chunk_overlap = 50
    tokenizer = get_tokenizer_fn(llm_model)
    merge_chunks = False

    def _count_tokens(text):
        return len(tokenizer(text))

    query_output_dir = f"{OUTPUT_DIR}/{format_sub_dir(query)}"
    shutil.rmtree(query_output_dir, ignore_errors=True)

    save_file(query, f"{query_output_dir}/query.md")

    search_engine_results = search_data(query, use_cache=use_cache)
    save_file(search_engine_results,
              f"{query_output_dir}/search_engine_results.json")

    urls = [r["url"] for r in search_engine_results]
    # Limit urls
    urls = urls[:5]

    html_list = []
    header_docs: List[HeaderDoc] = []
    search_results: List[HeaderSearchResult] = []

    async for url, status, html in scrape_urls(urls, show_progress=True):
        if status == "completed" and html:
            html_list.append(html)

            sub_source_dir = format_sub_source_dir(url)
            sub_output_dir = os.path.join(
                query_output_dir, "pages", sub_source_dir)

            save_file(html, f"{sub_output_dir}/page.html")

            links = set(scrape_links(html, url))
            links = [link for link in links if (
                link != url if isinstance(link, str) else link["url"] != url)]
            save_file(links, os.path.join(
                sub_output_dir, "links.json"))

            doc_markdown = convert_html_to_markdown(html, ignore_links=False)
            save_file(doc_markdown, f"{sub_output_dir}/page.md")

            doc_analysis = analyze_markdown(doc_markdown)
            save_file(doc_analysis, f"{sub_output_dir}/analysis.json")
            doc_markdown_tokens = base_parse_markdown(doc_markdown)
            save_file(doc_markdown_tokens,
                      f"{sub_output_dir}/markdown_tokens.json")

            original_docs = derive_by_header_hierarchy(doc_markdown)
            save_file(original_docs, f"{sub_output_dir}/docs.json")

            for doc in original_docs:
                doc["source"] = url

            sub_results = list(
                search_headers(
                    original_docs,
                    query,
                    top_k=top_k,
                    threshold=threshold,
                    embed_model=embed_model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    tokenizer=_count_tokens,
                    merge_chunks=merge_chunks
                )
            )

            save_file(sub_results, f"{sub_output_dir}/search_results.json")

            header_docs.extend(original_docs)
            search_results.extend(sub_results)

    # Sort search_results by score then update rank
    search_results = sorted(
        search_results, key=lambda x: x["score"], reverse=True)
    for i, result in enumerate(search_results, 1):
        result["rank"] = i

    save_file(header_docs, f"{query_output_dir}/docs.json")
    save_file(search_results, f"{query_output_dir}/search_results.json")

    # Filter results so that their combined context does not exceed max_tokens
    filtered_results = []
    current_tokens = 0
    for result in search_results:
        # Use the same context formatting as group_results_by_source_for_llm_context
        # to estimate token count for each result
        # We'll use the 'content' field if available, else str(result)
        content = f"{result["parent_header"] or ""}\n{result["header"]}\n{result['content']}".strip(
        )
        tokens: int = count_tokens(llm_model, content)
        if current_tokens + tokens > max_tokens:
            break
        filtered_results.append(result)
        current_tokens += tokens

    save_file({
        "query": query,
        "count": len(filtered_results),
        "total_tokens": current_tokens,
        "results": filtered_results
    }, f"{query_output_dir}/contexts.json")

    context = group_results_by_source_for_llm_context(filtered_results)
    save_file(context, f"{query_output_dir}/context.md")
    mlx = MLXModelRegistry.load_model(llm_model)
    prompt = PROMPT_TEMPLATE.format(query=query, context=context)
    save_file(prompt, f"{query_output_dir}/prompt.md")
    llm_response = mlx.chat(prompt, llm_model, temperature=0.7, verbose=True)
    save_file(llm_response["content"], f"{query_output_dir}/response.md")

    input_tokens = count_tokens(llm_model, prompt)
    output_tokens = count_tokens(llm_model, llm_response["content"])

    save_file({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }, f"{query_output_dir}/tokens_info.json")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Run semantic search and processing pipeline.")
    p.add_argument("query_pos", type=str, nargs="?",
                   help="Search query as positional argument")
    p.add_argument("-q", "--query", type=str,
                   help="Search query using optional flag")
    args = p.parse_args()

    query = args.query if args.query else args.query_pos or "Top isekai anime 2025."

    asyncio.run(main(query))
