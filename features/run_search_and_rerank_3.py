import asyncio
import os
import re
import shutil
import string
from typing import List
from jet.code.markdown_types.markdown_parsed_types import HeaderDoc
from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.code.markdown_utils._markdown_analyzer import analyze_markdown
from jet.code.markdown_utils._markdown_parser import derive_by_header_hierarchy, parse_markdown
from jet.file.utils import load_file, save_file
from jet.logger.config import colorize_log
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.models.tokenizer.base import get_tokenizer_fn
from jet.scrapers.hrequests_utils import scrape_urls
from jet.scrapers.utils import scrape_links, search_data
from jet.vectors.semantic_search.header_vector_search import HeaderSearchResult, search_headers


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


def save_results(query: str, results: List[HeaderSearchResult], split_chunks: bool):
    print(f"Search results for '{query}' in these dirs:")
    for num, result in enumerate(results[:10], start=1):
        header = result["header"]
        parent_header = result["parent_header"]
        start_idx = result["metadata"]["start_idx"]
        end_idx = result["metadata"]["end_idx"]
        chunk_idx = result["metadata"]["chunk_idx"]
        num_tokens = result["metadata"]["num_tokens"]
        score = result["score"]
        print(
            f"{colorize_log(f"{num}.)", "ORANGE")} Score: {colorize_log(f'{score:.3f}', 'SUCCESS')} | Chunk: {chunk_idx} | Tokens: {num_tokens} | Start - End: {start_idx} - {end_idx}\nParent: {parent_header} | Header: {header}")

    save_file({
        "query": query,
        "count": len(results),
        "merged": not split_chunks,
        "results": results
    }, f"{OUTPUT_DIR}/results_{'split' if split_chunks else 'merged'}.json")


def format_sub_source_dir(source: str) -> str:
    """Format a source (URL or file path) into a directory name."""
    clean_source = re.sub(r'^(https?://|www\.)|(\?.*)', '', source)
    clean_source = clean_source.replace(os.sep, '_')
    trans_table = str.maketrans({p: '_' for p in string.punctuation})
    formatted = clean_source.translate(trans_table).lower()
    formatted = re.sub(r'_+', '_', formatted)
    return formatted.strip('_')


async def main():
    """Main function to demonstrate file search."""
    query = "Top isekai anime 2025."
    embed_model: EmbedModelType = "all-MiniLM-L6-v2"
    llm_model: LLMModelType = "qwen3-1.7b-4bit"
    use_cache = True

    # docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_2/top_isekai_anime_2025/pages"

    search_engine_results = search_data(query, use_cache=use_cache)
    save_file(search_engine_results,
              f"{OUTPUT_DIR}/search_engine_results.json")

    urls = [r["url"] for r in search_engine_results]
    # Limit urls
    urls = urls[:2]

    html_list = []
    header_docs: List[HeaderDoc] = []

    async for url, status, html in scrape_urls(urls, show_progress=True):
        if status == "completed" and html:
            html_list.append(html)

            sub_source_dir = format_sub_source_dir(url)
            sub_output_dir = os.path.join(OUTPUT_DIR, "pages", sub_source_dir)
            shutil.rmtree(sub_output_dir, ignore_errors=True)

            save_file(html, f"{sub_output_dir}/page.html")

            links = set(scrape_links(html, url))
            links = [link for link in links if (
                link != url if isinstance(link, str) else link["url"] != url)]
            save_file(links, os.path.join(
                sub_output_dir, "links.json"))

            doc_markdown = convert_html_to_markdown(html)
            save_file(doc_markdown, f"{sub_output_dir}/page.md")

            doc_analysis = analyze_markdown(doc_markdown)
            save_file(doc_analysis, f"{sub_output_dir}/analysis.json")
            doc_markdown_tokens = parse_markdown(
                doc_markdown, merge_headers=False, merge_contents=False, ignore_links=False)
            save_file(doc_markdown_tokens,
                      f"{sub_output_dir}/markdown_tokens.json")

            original_docs = derive_by_header_hierarchy(doc_markdown)
            save_file(original_docs, f"{sub_output_dir}/docs.json")

            header_docs.extend(original_docs)

    save_file(header_docs, f"{OUTPUT_DIR}/header_docs.json")

    top_k = len(header_docs)
    threshold = 0.0  # Using default threshold
    chunk_size = 500
    chunk_overlap = 100
    tokenizer = get_tokenizer_fn(embed_model)

    def count_tokens(text):
        return len(tokenizer(text))

    split_chunks = True
    with_split_chunks_results = list(
        search_headers(
            header_docs,
            query,
            top_k=top_k,
            threshold=threshold,
            embed_model=embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            split_chunks=split_chunks,
            tokenizer=count_tokens
        )
    )
    save_results(query,  with_split_chunks_results, split_chunks)

    split_chunks = False
    without_split_chunks_results = list(
        search_headers(
            header_docs,
            query,
            top_k=top_k,
            threshold=threshold,
            embed_model=embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            split_chunks=split_chunks,
            tokenizer=count_tokens
        )
    )
    save_results(query, without_split_chunks_results, split_chunks)


if __name__ == "__main__":
    asyncio.run(main())
