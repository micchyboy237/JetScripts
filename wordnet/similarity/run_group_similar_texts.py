import os
import time

from fastapi.utils import generate_unique_id
from jet.file.utils import save_file, load_file
from jet.logger import logger
from jet.models.model_types import EmbedModelType
from jet.vectors.clusters.cluster_types import ClusteringMode
from jet.vectors.document_types import HeaderDocument
from jet.wordnet.similarity import group_similar_texts


def main(mode: ClusteringMode):
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_5/top_isekai_anime_2025/search_results.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    mode_output_dir = f"{output_dir}/{mode}"

    docs = load_file(docs_file)
    docs = docs["results"]
    documents = [
        f"{doc["header"].lstrip('#').strip()}\n{doc["content"]}" for doc in docs]

    model_name: EmbedModelType = "all-MiniLM-L6-v2"

    # Start timing
    start_time = time.time()

    ids = [doc["metadata"]["doc_id"] for doc in docs]

    grouped_similar_texts = group_similar_texts(
        documents, model_name=model_name, ids=ids, mode=mode)

    # End timing
    end_time = time.time()
    execution_time = end_time - start_time

    # Log performance
    logger.log(f"group_similar_texts:",
               f"{execution_time:.2f}s", colors=["WHITE", "ORANGE"])

    # Map grouped_similar_texts (which contains lists of doc_ids) back to the original doc objects
    doc_id_to_doc = {doc["metadata"]["doc_id"]: doc for doc in docs}
    mapped_results = [
        [
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
            for doc_id in group if (doc := doc_id_to_doc.get(doc_id))
        ]
        for group in grouped_similar_texts
    ]

    save_file({"execution_time": f"{execution_time:.2f}s", "count": len(grouped_similar_texts), "results": mapped_results},
              f"{mode_output_dir}/results.json")

    clusters = []
    for group in grouped_similar_texts:
        # group is a list of doc_ids
        docs_in_group = [doc_id_to_doc[doc_id]
                         for doc_id in group if doc_id in doc_id_to_doc]
        total_tokens = sum(doc["metadata"].get("num_tokens", 0)
                           for doc in docs_in_group)
        headers = [doc.get("header") for doc in docs_in_group]
        clusters.append({
            "count": len(group),
            "total_tokens": total_tokens,
            "headers": headers
        })
    save_file({
        "count": len(clusters),
        "total_tokens": sum(cluster["total_tokens"] for cluster in clusters),
        "clusters": clusters,
    }, f"{mode_output_dir}/clusters.json")


if __name__ == '__main__':
    main("agglomerative")
    main("kmeans")
