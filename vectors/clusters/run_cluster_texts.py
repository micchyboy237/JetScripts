import os
import shutil
import json
from typing import List
from jet.code.html_utils import convert_dl_blocks_to_md
from jet.file.utils import load_file, save_file
from jet.scrapers.header_hierarchy import HtmlHeaderDoc, extract_header_hierarchy
from jet.vectors.clusters.base import cluster_texts

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Example usage
if __name__ == "__main__":
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_isekai_anime_2025/https_gamerant_com_new_isekai_anime_2025/page.html"

    html_str: str = load_file(html_file)
    html_str = convert_dl_blocks_to_md(html_str)
    save_file(html_str, f"{OUTPUT_DIR}/page.html")

    headings: List[HtmlHeaderDoc] = extract_header_hierarchy(html_str)
    save_file(headings, f"{OUTPUT_DIR}/headings.json")
    sample_texts = [f"{header["header"]}\n{header["content"]}" for header in headings]

    results = cluster_texts(
        texts=sample_texts,
        model_name="all-MiniLM-L12-v2",
        batch_size=32,
        device="cpu",
        reduce_dim=True,
        n_components=5,
        min_cluster_size=2
    )

    # Print results
    for result in results:
        print(f"Text: {json.dumps(result['text'])[:100]}")
        print(f"Cluster: {result['label']}")
        print(f"Probability: {result['cluster_probability']:.4f}")
        print(f"Is Noise: {result['is_noise']}")
        print(f"Cluster Size: {result['cluster_size']}")
        print("-" * 50)

    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    save_file(results, f"{output_dir}/clusters.json")
