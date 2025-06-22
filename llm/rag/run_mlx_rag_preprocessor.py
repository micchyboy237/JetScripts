import os
import time
from jet.file.utils import load_file, save_file
from jet.llm.rag.rag_preprocessor import MLXRAGProcessor, WebDataPreprocessor
from jet.logger import logger
from typing import List, Dict
from numpy.typing import NDArray


def main():
    """Main function to demonstrate preprocessing and MLX RAG usage with classification."""
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/chunked_docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    docs: Dict = load_file(docs_file)
    query: str = f"Will this webpage header have an answer to this query?\nQuery: {docs['query']}"
    chunks: List[str] = [doc["metadata"]["header"] for doc in docs["results"]]
    top_k: int = len(chunks)
    try:
        # Timing: total
        start_total = time.time()
        mlx_processor = MLXRAGProcessor(show_progress=True, batch_size=4)
        logger.info("Generating embeddings for chunks")
        # Timing: embedding generation
        start_embed = time.time()
        embeddings: NDArray = mlx_processor.generate_embeddings(chunks)
        end_embed = time.time()
        total_embed = end_embed - start_embed
        logger.info(
            f"Embedding generation took {total_embed:.2f} seconds")
        if embeddings.shape[0] != len(chunks):
            logger.error("Mismatch between chunks and embeddings, exiting")
            return
        logger.info(f"Query: {query}")
        logger.info(f"Number of chunks processed: {len(chunks)}")
        logger.info(f"Classifying query with stream_generate: {query}")
        logger.info("\nStreaming Classifications:")
        response: List[Dict] = []
        # Timing: classification
        start_classify = time.time()
        for label, score, idx in mlx_processor.stream_generate(query, chunks, embeddings, top_k=top_k, relevance_threshold=0.7):
            logger.debug(
                f"Stream Classification {idx}: Label={label}, Score={score:.4f}")
            response.append({
                "num_chunk": idx,
                "label": label,
                "score": score,
                "chunk": chunks[idx],
            })
        end_classify = time.time()
        total_classify = end_classify - start_classify
        logger.info(
            f"Classification (stream_generate) took {total_classify:.2f} seconds")
        logger.info("Main function completed successfully")
        save_file(query, f"{output_dir}/query.md")
        save_file(response, f"{output_dir}/response.json")
        save_file(chunks, f"{output_dir}/chunks.json")
        end_total = time.time()
        total_time = end_total - start_total
        logger.info(
            f"Total execution time: {total_time:.2f} seconds")
        # Save all execution times
        from datetime import datetime, timedelta

        def format_seconds(seconds):
            # Format seconds as H:MM:SS.sss
            return str(timedelta(seconds=seconds))

        def format_timestamp(ts):
            # Format timestamp as ISO 8601 string
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


if __name__ == "__main__":
    main()
