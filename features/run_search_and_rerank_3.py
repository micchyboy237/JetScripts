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
from jet.wordnet.analyzers.text_analysis import analyze_readability


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

    # Initialize tokenizer
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
        if not isinstance(text, str):
            logger.debug(
                f"Non-string content found for source: {source}, doc_index: {doc['metadata'].get('doc_index', 0)}, type: {type(text)}. Converting to string.")
            text = str(text) if text else ""
        grouped_temp[source].append(doc)

    context_blocks = []
    for source, docs in grouped_temp.items():
        block = f"<!-- Source: {source} -->\n\n"
        seen_header_text_in_block = set()

        # Group by doc_index and header to handle overlaps
        grouped_by_header: DefaultDict[tuple[int, str],
                                       List[HeaderSearchResult]] = defaultdict(list)
        for doc in sorted(docs, key=lambda x: (x["metadata"].get("doc_index", 0), x["metadata"].get("start_idx", 0))):
            doc_index = doc["metadata"].get("doc_index", 0)
            header = doc.get("header", "") or ""
            grouped_by_header[(doc_index, header)].append(doc)

        for (doc_index, header), chunks in grouped_by_header.items():
            parent_header = chunks[0].get("parent_header", "None")
            parent_level = chunks[0]["metadata"].get("parent_level", None)
            doc_level = chunks[0]["metadata"].get(
                "level", 0) if chunks[0]["metadata"].get("level") is not None else 0
            parent_header_key = strip_hashtags(
                parent_header) if parent_header and parent_header != "None" else None
            header_key = strip_hashtags(header) if header else None

            # Check for matching child headers to avoid redundant parent headers
            has_matching_child = any(
                strip_hashtags(d.get("header", "")) == parent_header_key
                for d in docs
                if d.get("header") and d["metadata"].get("level", 0) >= 0
            )

            # Add parent header if it hasn't been added and has no matching child
            if parent_header_key and parent_level is not None and not has_matching_child and parent_header_key not in seen_header_text_in_block:
                block += f"{parent_header}\n\n"
                seen_header_text_in_block.add(parent_header_key)

            # Add header if it hasn't been added
            if header_key and header_key not in seen_header_text_in_block and doc_level >= 0:
                block += f"{header}\n\n"
                seen_header_text_in_block.add(header_key)
                seen_header_text[source].add(header_key)

            # Sort chunks by start_idx and merge overlapping or adjacent chunks
            chunks.sort(key=lambda x: x["metadata"]["start_idx"])
            merged_content = ""
            start_idx = chunks[0]["metadata"]["start_idx"]
            end_idx = chunks[0]["metadata"]["end_idx"]
            current_content = chunks[0]["content"]
            merged_content = current_content

            for next_chunk in chunks[1:]:
                next_start = next_chunk["metadata"]["start_idx"]
                next_end = next_chunk["metadata"]["end_idx"]
                next_content = next_chunk["content"]
                if not isinstance(next_content, str):
                    logger.debug(
                        f"Non-string content in chunk for source: {source}, doc_index: {doc_index}, type: {type(next_content)}. Converting to string.")
                    next_content = str(next_content) if next_content else ""

                # Merge if chunks overlap or are adjacent
                if next_start <= end_idx + 1:
                    overlap = end_idx - next_start + 1 if next_start <= end_idx else 0
                    additional_content = next_content[overlap:
                                                      ] if overlap > 0 else next_content
                    merged_content += additional_content
                    end_idx = max(end_idx, next_end)
                else:
                    # Append merged content to block
                    block += merged_content + "\n\n"
                    # Start new merged chunk
                    merged_content = next_content
                    start_idx = next_start
                    end_idx = next_end

            # Append the last merged chunk
            block += merged_content + "\n\n"

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
    merge_chunks = False

    query_output_dir = f"{OUTPUT_DIR}/{format_sub_dir(query)}"
    shutil.rmtree(query_output_dir, ignore_errors=True)

    save_file(query, f"{query_output_dir}/query.md")

    search_engine_results = search_data(query, use_cache=use_cache)
    save_file(search_engine_results,
              f"{query_output_dir}/search_engine_results.json")

    urls = [r["url"] for r in search_engine_results]

    html_list = []
    header_docs: List[HeaderDoc] = []
    search_results: List[HeaderSearchResult] = []

    total_tokens = 0
    total_high_score_tokens = 0
    total_mtld_high_score_average = 0
    HIGH_QUALITY_SCORE = 0.6
    TARGET_HIGH_SCORE_TOKENS = max_tokens * 2
    TARGET_TOKENS = 10000

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
                    tokenizer_model=llm_model,
                    merge_chunks=merge_chunks
                )
            )

            sub_total_tokens = sum(
                result["metadata"]["num_tokens"] for result in sub_results)

            sub_high_score_tokens = sum(
                result["metadata"]["num_tokens"]
                for result in sub_results
                if (
                    result["score"] >= HIGH_QUALITY_SCORE
                    and result.get("mtld_category") != "very_low"
                )
            )
            sub_mtld_high_score_values = [
                analyze_readability(result["content"])["mtld"]
                for result in sub_results
                if (
                    result["score"] >= HIGH_QUALITY_SCORE
                    and analyze_readability(result["content"])["mtld_category"] != "very_low"
                )
            ]
            sub_mtld_high_score_average = (
                sum(sub_mtld_high_score_values) /
                len(sub_mtld_high_score_values)
                if sub_mtld_high_score_values else 0
            )

            save_file({
                "query": query,
                "url": url,
                "count": len(sub_results),
                "total_tokens": sub_total_tokens,
                "high_score_tokens": sub_high_score_tokens,
                "mtld_high_score_average": sub_mtld_high_score_average,
                "results": sub_results,
            }, f"{sub_output_dir}/search_results.json")

            header_docs.extend(original_docs)
            search_results.extend(sub_results)

            total_tokens += sub_total_tokens
            total_high_score_tokens += sub_high_score_tokens
            total_mtld_high_score_average += round(
                sub_mtld_high_score_average, 2)

            if total_high_score_tokens >= TARGET_HIGH_SCORE_TOKENS or total_tokens >= TARGET_TOKENS:
                logger.info(
                    f"Stopping processing: {total_tokens} tokens collected from source: {url}.")
                break

    # Sort search_results by score then update rank
    search_results = sorted(
        search_results, key=lambda x: x["score"], reverse=True)
    for i, result in enumerate(search_results, 1):
        result["rank"] = i

    save_file({
        "query": query,
        "count": len(header_docs),
        "documents": header_docs
    }, f"{query_output_dir}/docs.json")

    searched_urls = list({result.get("metadata", {}).get(
        "source") for result in search_results if result.get("metadata", {}).get("source")})
    save_file({
        "query": query,
        "count": len(search_results),
        "urls": searched_urls,
        "total_tokens": total_tokens,
        "total_high_score_tokens": total_high_score_tokens,
        "total_mtld_high_score_average": total_mtld_high_score_average,
        "results": search_results
    }, f"{query_output_dir}/search_results.json")

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
