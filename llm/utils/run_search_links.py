import os
import shutil
from jet.models.model_types import EmbedModelType
from jet.llm.utils.link_searcher import search_links
from jet.file.utils import load_file, save_file
from jet.llm.utils.search_docs import search_docs
from jet.wordnet.similarity import get_text_groups, group_similar_texts
import re
from urllib.parse import urlparse


def filter_links(links: list[str]) -> list[str]:
    """
    Remove links with no paths or containing hashtag parameters.
    - No path: Links with only domain (e.g., https://example.com)
    - Hashtag params: Links containing '#' in the URL
    """
    filtered = []
    for link in links:
        # Skip if link contains hashtag
        if '#' in link:
            continue

        # Parse URL to check path
        parsed = urlparse(link)
        # Check if path is empty or just '/'
        if not parsed.path or parsed.path == '/':
            continue

        filtered.append(link)

    return filtered


if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)

    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/links.json"

    query = "List all ongoing and upcoming isekai anime 2025."
    embed_model: EmbedModelType = "snowflake-arctic-embed-m"

    # Load JSON data
    links: list[str] = load_file(docs_file)

    # Filter links before sorting
    links = filter_links(links)
    links = sorted(links)
    save_file(links, f"{output_dir}/links.json")

    # grouped_links = group_similar_texts(links)
    grouped_links = get_text_groups(
        links, threshold=0.9, model_name=embed_model)
    save_file(grouped_links, f"{output_dir}/grouped_links.json")
    links = [group[0] for group in grouped_links]

    results = search_docs(query, links, model=embed_model,
                          top_k=5, threshold=0.7)

    output_path = f"{output_dir}/results.json"
    save_file({
        "query": query,
        "results": results
    }, output_path)
