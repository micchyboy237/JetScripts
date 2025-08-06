import os
import shutil
import time
from typing import List
import numpy as np
from jet.utils.text import format_sub_dir
from jet.vectors.filters import select_diverse_texts, DiverseResult
from fastapi.utils import generate_unique_id
from jet.file.utils import save_file, load_file
from jet.logger import logger
from jet.models.embeddings.base import generate_embeddings
from jet.models.model_types import EmbedModelType
from jet.models.tokenizer.base import count_tokens
from jet.vectors.clusters.cluster_types import ClusteringMode
from jet.vectors.document_types import HeaderDocument
from jet.wordnet.similarity import group_similar_texts

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


def load_and_process_common_data(embed_model: EmbedModelType) -> tuple[list[dict], list[str]]:
    """Load input documents and generate global diverse texts."""
    query = "Top isekai anime 2025."
    query_sub_dir = format_sub_dir(query)
    docs_file = f"/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_5/{query_sub_dir}/search_results.json"

    query_output_dir = f"{OUTPUT_DIR}/{query_sub_dir}"
    shutil.rmtree(query_output_dir, ignore_errors=True)

    docs = load_file(docs_file)
    docs = docs["results"]
    documents = [
        f"{doc['header'].lstrip('#').strip()}\n{doc['content']}" for doc in docs]

    doc_embeddings = generate_embeddings(
        documents, embed_model, show_progress=True)
    all_doc_embeddings = np.ascontiguousarray(doc_embeddings)

    doc_ids = [doc["id"] for doc in docs]
    diverse_results: List[DiverseResult] = select_diverse_texts(
        cluster_embeddings=all_doc_embeddings,
        cluster_texts=documents,
        max_diverse_texts=len(documents),
        ids=doc_ids
    )
    diverse_texts = [result["text"] for result in diverse_results]
    token_counts: List[int] = count_tokens(
        embed_model, diverse_texts, prevent_total=True)
    save_file({
        "count": len(diverse_results),
        "total_tokens": sum(token_counts),
        "texts": [{
            "id": result["id"],
            "index": result["index"],
            "tokens": tokens,
            "text": result["text"],
            "score": result["score"]
        } for tokens, result in zip(token_counts, diverse_results)],
    }, f"{query_output_dir}/diverse_results.json")

    # Select diverse headers for logging only
    doc_headers = [doc["header"].lstrip('#').strip() for doc in docs]
    doc_header_embeddings = generate_embeddings(
        doc_headers, embed_model, show_progress=True)
    all_doc_header_embeddings = np.ascontiguousarray(doc_header_embeddings)

    diverse_header_results: List[DiverseResult] = select_diverse_texts(
        cluster_embeddings=all_doc_header_embeddings,
        cluster_texts=doc_headers,
        max_diverse_texts=len(doc_headers),
        ids=doc_ids
    )
    diverse_headers = [result["text"] for result in diverse_header_results]
    token_counts: List[int] = count_tokens(
        embed_model, diverse_headers, prevent_total=True)
    save_file({
        "count": len(diverse_header_results),
        "total_tokens": sum(token_counts),
        "texts": [{
            "id": result["id"],
            "index": result["index"],
            "tokens": tokens,
            "text": result["text"],
            "score": result["score"]
        } for tokens, result in zip(token_counts, diverse_header_results)],
    }, f"{query_output_dir}/diverse_headers.json")

    return docs, documents, query_output_dir


def main(mode: ClusteringMode, docs: list[dict], documents: list[str], output_dir: str):
    mode_output_dir = f"{output_dir}/{mode}"
    embed_model: EmbedModelType = "all-MiniLM-L6-v2"

    # Start timing
    start_time = time.time()

    ids = [doc["id"] for doc in docs]

    embeddings = generate_embeddings(
        documents, embed_model, show_progress=True)
    all_embeddings = np.ascontiguousarray(embeddings)

    grouped_similar_texts = group_similar_texts(
        documents, model_name=embed_model, ids=ids, embeddings=embeddings, mode=mode)

    # End timing
    end_time = time.time()
    execution_time = end_time - start_time

    # Log performance
    logger.log(f"group_similar_texts:",
               f"{execution_time:.2f}s", colors=["WHITE", "ORANGE"])

    # Map grouped_similar_texts (List[ClusterResult] with doc_ids) back to the original doc objects
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

    doc_id_to_embeddings = {
        doc["id"]: emb for doc, emb in zip(docs, embeddings)}

    cluster_diverse_texts = []
    cluster_diverse_results = []
    for group in grouped_similar_texts:
        cluster_texts = []
        cluster_embeddings = []
        cluster_ids = []
        for doc_id in group["texts"]:
            header_doc = doc_id_to_doc[doc_id]
            text = f"{header_doc['header']}\n{header_doc['content']}"
            text_embeddings = doc_id_to_embeddings[doc_id]
            cluster_texts.append(text)
            cluster_embeddings.append(text_embeddings)
            cluster_ids.append(doc_id)

        cluster_embeddings = np.ascontiguousarray(cluster_embeddings)

        diverse_results: List[DiverseResult] = select_diverse_texts(
            cluster_embeddings=cluster_embeddings,
            cluster_texts=cluster_texts,
            diversity_threshold=0.8,
            max_diverse_texts=len(cluster_texts),
            ids=cluster_ids
        )
        cluster_diverse_texts.extend([result["text"]
                                     for result in diverse_results])
        cluster_diverse_results.extend([
            {**result, "cluster_label": group["label"]} for result in diverse_results
        ])

    token_counts: List[int] = count_tokens(
        embed_model, cluster_diverse_texts, prevent_total=True)
    save_file({
        "count": len(cluster_diverse_texts),
        "total_tokens": sum(token_counts),
        "texts": [{
            "id": result["id"],
            "index": result["index"],
            "tokens": tokens,
            "text": result["text"],
            "score": result["score"],
            "cluster_label": result["cluster_label"]
        } for tokens, result in zip(token_counts, cluster_diverse_results)],
    }, f"{mode_output_dir}/diverse_cluster_results.json")


if __name__ == '__main__':
    embed_model: EmbedModelType = "all-MiniLM-L6-v2"
    docs, documents, output_dir = load_and_process_common_data(embed_model)
    main("agglomerative", docs, documents, output_dir)
    main("kmeans", docs, documents, output_dir)
    main("dbscan", docs, documents, output_dir)
    main("hdbscan", docs, documents, output_dir)
