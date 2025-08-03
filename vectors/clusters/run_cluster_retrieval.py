import math
import os
import shutil
from typing import List, Tuple
from jet.code.markdown_types.markdown_parsed_types import HeaderDoc
from jet.code.markdown_utils._markdown_parser import derive_by_header_hierarchy
from jet.file.utils import load_file, save_file
from jet.llm.llm_generator import LLMGenerator
from jet.logger import logger
from jet.logger.config import colorize_log
from jet.models.tokenizer.base import count_tokens
from jet.vectors.clusters.retrieval import RetrievalConfigDict, VectorRetriever
from jet.wordnet.text_chunker import chunk_texts

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def main():
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_4/top_isekai_anime_2025/docs.json"
    header_docs: List[HeaderDoc] = load_file(docs_file)["documents"]

    config: RetrievalConfigDict = {
        "model_name": 'all-MiniLM-L6-v2',
        "min_cluster_size": 2,
        "k_clusters": 3,
        "top_k":  None,
        "cluster_threshold": 20,
        "cache_file":  None,
        "threshold": None,
    }
    chunk_size = 500
    chunk_overlap = 100

    query = "Top isekai anime 2025."
    chunked_docs = []
    texts = []
    for doc_index, header_doc in enumerate(header_docs):
        header = header_doc["header"]
        content = header_doc['content']
        buffer: int = count_tokens(config["model_name"], header)
        chunks = chunk_texts(content, chunk_size,
                             chunk_overlap, config["model_name"], buffer)
        chunks_with_headers = [f"{header}\n{chunk}" for chunk in chunks]

        texts.extend(chunks_with_headers)

        chunks_with_info = [{
            "chunk_idx": chunk_idx,
            "tokens": count_tokens(config["model_name"], chunk),
            "chunk": chunk,
        } for chunk_idx, chunk in enumerate(chunks_with_headers)]
        chunked_docs.append({
            "doc_index": doc_index,
            "count": len(chunks_with_info),
            "total_tokens": sum(chunk_info["tokens"] for chunk_info in chunks_with_info),
            "chunks": chunks_with_info,
        })

    token_counts: List[int] = count_tokens(
        config["model_name"], texts, prevent_total=True)
    texts_with_tokens = [{
        "tokens": tokens,
        "text": text,
    } for tokens, text in zip(token_counts, texts)]
    save_file({
        "count": len(texts),
        "max_tokens": max(token_counts),
        "ave_tokens": math.ceil(sum(token_counts) / len(token_counts)) if token_counts else 0,
        "min_tokens": min(token_counts),
        "texts": texts_with_tokens
    }, f"{OUTPUT_DIR}/texts.json")
    save_file({
        "count": len(chunked_docs),
        "chunks": chunked_docs
    }, f"{OUTPUT_DIR}/chunked_docs.json")

    retriever = VectorRetriever(config)
    retriever.load_or_compute_embeddings(texts)
    retriever.cluster_embeddings()
    retriever.build_index()
    search_results = retriever.search_chunks(query)

    # Save cluster assignments to clusters.json
    clusters = retriever.get_clusters()
    save_file({
        "count": len(clusters),
        "clusters": clusters
    }, f"{OUTPUT_DIR}/clusters.json")

    # Save centroid information and similarity scores to centroids.json
    centroids = retriever.get_centroids()
    centroid_scores = retriever.search_centroids(query)
    save_file({
        "query": query,
        "count": len(centroids),
        "centroids": centroid_scores
    }, f"{OUTPUT_DIR}/search_centroids.json")

    save_file({
        "query": query,
        "count": len(search_results),
        "results": search_results
    }, f"{OUTPUT_DIR}/search_results.json")

    logger.info("\nTop relevant chunks for RAG:")
    for i, result in enumerate(search_results, 1):
        print(f"{colorize_log(f"{i}.", "ORANGE")} | {
              colorize_log(f"{result["score"]:.4f}", "SUCCESS")} | {colorize_log(f"{result["num_tokens"]}", "DEBUG")} | {result["text"][:50]}")

    generator = LLMGenerator()
    response = generator.generate_response(
        query, search_results, generation_config={"verbose": True})

    save_file(f"# LLM Generation\n\n## Prompt\n\n{response["prompt"]}\n\n## Response\n\n{response["response"]}",
              f"{OUTPUT_DIR}/llm_generation.md")


if __name__ == "__main__":
    main()
