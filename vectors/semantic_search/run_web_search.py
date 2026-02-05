# ──────────────────────────────────────────────────────────────────────────────
#  run_web_search.py
# ──────────────────────────────────────────────────────────────────────────────


import logging
from pathlib import Path

from jet.code.markdown_types.markdown_parsed_types import HeaderDoc
from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.code.markdown_utils._markdown_parser import (
    derive_by_header_hierarchy,
)
from jet.vectors.semantic_search.web_search import (
    hybrid_search,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem


if __name__ == "__main__":
    import argparse
    import asyncio
    import shutil

    from jet.code.html_utils import preprocess_html
    from jet.code.markdown_utils._converters import convert_html_to_markdown
    from jet.code.markdown_utils._markdown_analyzer import analyze_markdown
    from jet.code.markdown_utils._markdown_parser import base_parse_markdown
    from jet.file.utils import save_file
    from jet.scrapers.utils import scrape_links
    from jet.utils.text import format_sub_dir, format_sub_source_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="Top 10 isekai anime 2026")
    args = parser.parse_args()

    use_cache = True

    query_output_dir = f"{OUTPUT_DIR}/{format_sub_dir(args.query)}"
    shutil.rmtree(query_output_dir, ignore_errors=True)

    llm_log_dir = Path(query_output_dir) / "llm_calls"

    result = asyncio.run(
        hybrid_search(args.query, llm_log_dir=llm_log_dir, use_cache=use_cache)
    )

    print(f"Found {len(result['filtered_results'])} relevant chunks")
    print(f"LLM response length: {len(result['llm_response'])} chars")

    search_engine_results = result.pop("search_engine_results")
    header_docs = result.pop("header_docs")
    all_search_results = result.pop("all_search_results")
    filtered_results = result.pop("filtered_results")
    url_stats = result.pop("url_stats")
    stats = result.pop("stats")
    grouped_context = result.pop("grouped_context")
    llm_messages = result.pop("llm_messages")
    llm_response = result.pop("llm_response")
    token_counts = result.pop("token_counts")
    all_htmls_with_status = result.pop("all_htmls_with_status")

    save_file(result, f"{query_output_dir}/result_meta.json")
    save_file(search_engine_results, f"{query_output_dir}/search_engine_results.json")
    save_file(header_docs, f"{query_output_dir}/header_docs.json")
    save_file(all_search_results, f"{query_output_dir}/all_search_results.json")
    save_file(filtered_results, f"{query_output_dir}/filtered_results.json")
    save_file(url_stats, f"{query_output_dir}/url_stats.json")
    save_file(stats, f"{query_output_dir}/stats.json")
    save_file(grouped_context, f"{query_output_dir}/grouped_context.md")
    save_file(llm_messages, f"{query_output_dir}/llm_messages.json")
    save_file(llm_response, f"{query_output_dir}/llm_response.md")
    save_file(token_counts, f"{query_output_dir}/token_counts.json")

    for url, info in all_htmls_with_status.items():
        status = info["status"]
        html = info["html"]

        sub_source_dir = format_sub_source_dir(url)
        sub_output_dir = Path(query_output_dir) / "pages" / sub_source_dir
        save_file(html, f"{sub_output_dir}/page.html")
        save_file(preprocess_html(html), f"{sub_output_dir}/page_preprocessed.html")

        links = set(scrape_links(html, url))
        links = [
            link
            for link in links
            if (link != url if isinstance(link, str) else link["url"] != url)
        ]
        save_file(links, sub_output_dir / "links.json")

        doc_markdown = convert_html_to_markdown(html, ignore_links=False)
        save_file(doc_markdown, f"{sub_output_dir}/page.md")

        doc_analysis = analyze_markdown(doc_markdown)
        save_file(doc_analysis, sub_output_dir / "analysis.json")
        doc_markdown_tokens = base_parse_markdown(doc_markdown)
        save_file(doc_markdown_tokens, sub_output_dir / "markdown_tokens.json")

        original_docs: list[HeaderDoc] = derive_by_header_hierarchy(
            doc_markdown, ignore_links=True
        )

        save_file(original_docs, sub_output_dir / "docs.json")
