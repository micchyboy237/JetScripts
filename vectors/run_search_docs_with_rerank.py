from urllib.parse import urlparse, unquote
import re
from typing import List, Dict, Optional

from jet.transformers.link_formatters import LinkFormatter
import os
from jet.file.utils import load_file, save_file
from jet.llm.utils.search_docs import search_docs_with_rerank

if __name__ == "__main__":

    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    links_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/links.json"
    links: list[str] = load_file(links_file)

    # Load JSON data
    docs = load_file(docs_file)
    print(f"Loaded JSON data {len(docs)} from: {docs_file}")
    # doc_links = [{"url": doc_link["url"], "text": doc_link["text"]}
    #              for doc in docs for doc_link in doc["metadata"]["links"]]
    doc_links = []

    embed_model = "all-MiniLM-L12-v2"
    query = "Tips and links to 2025 online registration steps for TikTok live selling in the Philippines, or recent guidelines for online sellers on TikTok in the Philippines, or 2025 registration process for TikTok live selling in the Philippines."

    # Initialize formatter
    formatter = LinkFormatter()

    # Format links
    formatted_links = formatter.format_links_for_embedding(links + doc_links)

    # Perform search
    search_links_results = search_docs_with_rerank(
        query=query,
        documents=formatted_links,
        model=embed_model,
        top_k=None
    )

    # Enrich results with formatted link and original URL
    enriched_results = []
    for result in search_links_results:
        formatted = formatted_links[result['doc_index']]
        enriched_results.append({
            **result,
            "formatted_link": formatted,
            "link": formatter.formatted_to_original_map.get(formatted, "")
        })

    # Save results
    save_file({
        "query": query,
        "results": enriched_results
    }, os.path.join(output_dir, "search_links_results.json"))
