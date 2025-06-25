import os
import shutil
import time
from jet.file.utils import load_file, save_file
from jet.llm.rag.mlx.classification import MLXRAGClassifier, generate_summary
from jet.logger import logger
from typing import List, Dict
from numpy.typing import NDArray
from jet.models.embeddings.base import generate_embeddings
from jet.models.model_types import ModelType
from jet.vectors.document_utils import get_leaf_documents
from jet.wordnet.text_chunker import truncate_texts
from jet.vectors.document_types import HeaderDocument
from datetime import datetime, timedelta

base_output_dir = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(base_output_dir, ignore_errors=True)


def main(window_size: int, start_index: int):
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_rag_strategies_reddit_2025/docs.json"
    output_dir = os.path.join(
        base_output_dir, f"window_size_{window_size}_start_{start_index}")
    docs: Dict = load_file(docs_file)
    query: str = docs['query']
    model: ModelType = "qwen3-1.7b-4bit"
    docs = HeaderDocument.from_list(docs["documents"])
    docs = [doc for doc in docs if doc["source_url"] ==
            "https://www.louisbouchard.ai/top-rag-techniques" and doc["header"].strip()]
    docs = docs[start_index:start_index + window_size]
    if not docs:
        logger.info(
            f"No valid documents available for window starting at index {start_index}")
        return
    chunks: List[str] = [doc["text"] for doc in docs]
    source_urls: List[str] = [doc["source_url"] for doc in docs]
    top_k: int = len(chunks)
    try:
        start_total = time.time()
        mlx_processor = MLXRAGClassifier(
            model, show_progress=True, batch_size=4)
        # Clear caches before processing to ensure fresh embeddings
        mlx_processor.clear_cache()
        logger.info("Generating embeddings for chunks")
        start_embed = time.time()
        embeddings = mlx_processor.generate_embeddings(
            chunks, group_ids=source_urls)
        end_embed = time.time()
        total_embed = end_embed - start_embed
        logger.info(f"Embedding generation took {total_embed:.2f} seconds")
        if embeddings.shape[0] != len(chunks):
            logger.error("Mismatch between chunks and embeddings, exiting")
            return
        logger.info(f"Query: {query}")
        logger.info(f"Number of chunks processed: {len(chunks)}")
        logger.info(f"Classifying query with classify: {query}")
        logger.info("\nClassifications:")
        results: List[Dict] = []
        start_classify = time.time()
        classification_results = mlx_processor.classify(
            query, chunks, embeddings, verbose=True)
        for res in classification_results:
            logger.debug(
                f"Classification {res['doc_index']}: Label={res['label']}, Score={res['score']:.4f}")
            original_doc = docs[res['doc_index']]
            results.append({
                "doc_index": original_doc["doc_index"],
                "header_level": original_doc["header_level"],
                "parent_header": original_doc["parent_header"],
                "label": res["label"],
                "score": res["score"],
                "source_url": original_doc["source_url"],
                "text": res["text"],
                "original_doc": original_doc["text"],
            })
        results.sort(key=lambda r: r["doc_index"])
        end_classify = time.time()
        total_classify = end_classify - start_classify
        logger.info(
            f"Classification (classify) took {total_classify:.2f} seconds")
        logger.info("Main function completed successfully")
        save_file(query, f"{output_dir}/query.md")
        save_file({
            "window_size": window_size,
            "start_index": start_index,
            "query": query,
            "count": len(results),
            "results": results
        }, f"{output_dir}/results.json")
        save_file(chunks, f"{output_dir}/chunks.json")
        end_total = time.time()
        total_time = end_total - start_total
        logger.info(
            f"Total execution time: {total_time:.2f} seconds")
        summary = generate_summary(
            query, results, chunks, total_embed, total_classify, total_time)
        save_file(summary, f"{output_dir}/summary.md")

        def format_seconds(seconds):
            return str(timedelta(seconds=seconds))

        def format_timestamp(ts):
            return datetime.fromtimestamp(ts).isoformat(sep=' ', timespec='milliseconds')
        timings = {
            "durations": {
                "embedding_generation": format_seconds(total_embed),
                "classification": format_seconds(total_classify),
                "total": format_seconds(total_time),
                "embedding_generation_seconds": total_embed,
                "classification_seconds": total_classify,
                "total_seconds": total_time,
            },
            "timestamps": {
                "start_total": format_timestamp(start_total),
                "end_total": format_timestamp(end_total),
                "start_embed": format_timestamp(start_embed),
                "end_embed": format_timestamp(end_embed),
                "start_classify": format_timestamp(start_classify),
                "end_classify": format_timestamp(end_classify),
                "start_total_unix": start_total,
                "end_total_unix": end_total,
                "start_embed_unix": start_embed,
                "end_embed_unix": end_embed,
                "start_classify_unix": start_classify,
                "end_classify_unix": end_classify,
            }
        }
        save_file(timings, os.path.join(output_dir, "timings.json"))
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
    finally:
        # Ensure cache is cleared after each run
        if 'mlx_processor' in locals():
            mlx_processor.clear_cache()


if __name__ == "__main__":
    window_size = 2
    start_index = 4
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_rag_strategies_reddit_2025/docs.json"
    docs: Dict = load_file(docs_file)
    total_docs = len(HeaderDocument.from_list(docs["documents"]))
    for window_size in range(1, (window_size * 2) + 1):
        main(window_size=window_size, start_index=start_index)
