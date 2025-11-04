import os
import shutil
import time
from typing import List

from jet.code.html_utils import convert_dl_blocks_to_md
from jet.file.utils import save_file, load_file
from jet.logger import logger
from jet.models.model_types import EmbedModelType
from jet.scrapers.header_hierarchy import HtmlHeaderDoc, extract_header_hierarchy
from jet.vectors.clusters.cluster_types import ClusteringMode
from jet.wordnet.similarity import group_similar_texts

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def main(documents: List[str], ids: List[str], mode: ClusteringMode):
    mode_output_dir = f"{OUTPUT_DIR}/{mode}"

    model_name: EmbedModelType = "all-MiniLM-L6-v2"

    # Start timing
    start_time = time.time()

    grouped_similar_texts = group_similar_texts(
        documents, model_name=model_name, ids=ids, mode=mode)

    # End timing
    end_time = time.time()
    execution_time = end_time - start_time

    # Log performance
    logger.log(f"group_similar_texts ({mode}):",
               f"{execution_time:.2f}s", colors=["WHITE", "ORANGE"])

    # Map grouped_similar_texts (which contains ClusterResult with doc_ids) back to the original doc objects
    doc_id_to_doc = {doc["id"]: doc for doc in docs}
    mapped_results = [
        {
            "label": group["label"],
            "docs": [
                {
                    "rank": doc.get("rank"),
                    "score": doc.get("score"),
                    "header": doc.get("header"),
                    "content": doc.get("content"),
                    "metadata": {
                        "doc_index": doc["metadata"].get("doc_index"),
                        "doc_id": doc["metadata"].get("doc_id"),
                        "source": doc["metadata"].get("source"),
                        "num_tokens": doc["metadata"].get("num_tokens"),
                    }
                }
                for doc_id in group["texts"] if (doc := doc_id_to_doc.get(doc_id))
            ]
        }
        for group in grouped_similar_texts
    ]

    save_file({"execution_time": f"{execution_time:.2f}s", "count": len(grouped_similar_texts), "results": mapped_results},
              f"{mode_output_dir}/results.json")

    clusters = []
    for group in grouped_similar_texts:
        # group["texts"] is a list of doc_ids
        docs_in_group = [doc_id_to_doc[doc_id]
                         for doc_id in group["texts"] if doc_id in doc_id_to_doc]
        total_tokens = sum(doc["metadata"].get("num_tokens", 0)
                           for doc in docs_in_group)
        headers = [doc.get("header") for doc in docs_in_group]
        clusters.append({
            "label": group["label"],
            "count": len(group["texts"]),
            "total_tokens": total_tokens,
            "headers": headers
        })
    save_file({
        "count": len(clusters),
        "total_tokens": sum(cluster["total_tokens"] for cluster in clusters),
        "clusters": clusters,
    }, f"{mode_output_dir}/clusters.json")


if __name__ == '__main__':
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_isekai_anime_2025/https_gamerant_com_new_isekai_anime_2025/page.html"

    html_str: str = load_file(html_file)
    html_str = convert_dl_blocks_to_md(html_str)
    save_file(html_str, f"{OUTPUT_DIR}/page.html")

    headings: List[HtmlHeaderDoc] = extract_header_hierarchy(html_str)
    save_file(headings, f"{OUTPUT_DIR}/headings.json")
    ids = [header["id"] for header in headings]
    docs = [f"{header["header"]}\n{header["content"]}" for header in headings]

    main(docs, ids, "agglomerative")
    main(docs, ids, "kmeans")
    main(docs, ids, "dbscan")
    main(docs, ids, "hdbscan")
