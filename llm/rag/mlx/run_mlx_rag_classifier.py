import os
import time
from jet.file.utils import load_file, save_file
from jet.llm.rag.mlx.classification import MLXRAGClassifier
from jet.logger import logger
from typing import List, Dict
from numpy.typing import NDArray
from jet.vectors.document_types import HeaderDocument
from collections import Counter
from datetime import datetime, timedelta


def generate_summary(query: str, results: List[Dict], chunks: List[str], total_embed: float, total_classify: float, total_time: float) -> str:
    """Generate a Markdown summary of classification results with insights.

    Args:
        query: The query used for classification.
        results: List of result dictionaries containing classification details.
        chunks: List of processed chunks.
        total_embed: Time taken for embedding generation in seconds.
        total_classify: Time taken for classification in seconds.
        total_time: Total execution time in seconds.

    Returns:
        Markdown-formatted summary string.
    """
    total_chunks = len(chunks)
    relevant_count = sum(1 for r in results if r["label"] == "relevant")
    non_relevant_count = total_chunks - relevant_count
    relevant_percentage = (relevant_count / total_chunks *
                           100) if total_chunks > 0 else 0
    non_relevant_percentage = 100 - relevant_percentage

    # Average score for relevant chunks
    relevant_scores = [r["score"] for r in results if r["label"] == "relevant"]
    avg_relevant_score = sum(relevant_scores) / \
        len(relevant_scores) if relevant_scores else 0

    # Distribution by source URL for relevant chunks
    source_url_counts = Counter(r["source_url"]
                                for r in results if r["label"] == "relevant")

    # Top 3 relevant chunks
    top_relevant = sorted(
        [r for r in results if r["label"] == "relevant"],
        key=lambda x: x["score"],
        reverse=True
    )[:3]

    # Format summary
    summary = [
        "# RAG Classification Summary",
        f"**Query**: {query}",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "## Overview",
        f"- **Total Chunks Processed**: {total_chunks}",
        f"- **Relevant Chunks**: {relevant_count} ({relevant_percentage:.2f}%)",
        f"- **Non-Relevant Chunks**: {non_relevant_count} ({non_relevant_percentage:.2f}%)",
        f"- **Average Score for Relevant Chunks**: {avg_relevant_score:.4f}",
        "",
        "## Performance",
        f"- **Embedding Generation Time**: {timedelta(seconds=total_embed)} ({total_embed:.2f} seconds)",
        f"- **Classification Time**: {timedelta(seconds=total_classify)} ({total_classify:.2f} seconds)",
        f"- **Total Execution Time**: {timedelta(seconds=total_time)} ({total_time:.2f} seconds)",
        "",
        "## Source URL Distribution (Relevant Chunks)",
        "\n".join(f"- {url}: {count} chunk(s)" for url,
                  count in source_url_counts.items()) or "- None",
        "",
        "## Top Relevant Chunks",
    ]
    if top_relevant:
        for i, r in enumerate(top_relevant, 1):
            summary.extend([
                f"### {i}. Chunk (Score: {r['score']:.4f})",
                f"- **Source URL**: {r['source_url']}",
                f"- **Chunk**: {r['chunk'][:100]}{'...' if len(r['chunk']) > 100 else ''}",
                ""
            ])
    else:
        summary.append("- No relevant chunks found.")

    return "\n".join(summary)


def main():
    """Main function to demonstrate preprocessing and MLX RAG usage with classification."""
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/chunked_docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    docs: Dict = load_file(docs_file)
    query: str = f"Will this webpage header have a concrete answer to this query?\nQuery: {docs['query']}"
    docs = [HeaderDocument(**doc) for doc in docs["results"]]
    chunks: List[str] = [doc["header"] for doc in docs]
    source_urls: List[str] = [doc["source_url"] for doc in docs]
    top_k: int = len(chunks)
    try:
        start_total = time.time()
        mlx_processor = MLXRAGClassifier(show_progress=True, batch_size=4)
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
        logger.info(f"Classifying query with stream_generate: {query}")
        logger.info("\nStreaming Classifications:")
        results: List[Dict] = []
        start_classify = time.time()
        for label, score, idx in mlx_processor.stream_generate(query, chunks, embeddings, top_k=top_k, relevance_threshold=0.7):
            logger.debug(
                f"Stream Classification {idx}: Label={label}, Score={score:.4f}")
            original_doc = docs[idx]
            results.append({
                "doc_index": original_doc["doc_index"],
                "chunk_index": original_doc["chunk_index"],
                "label": label,
                "score": score,
                "source_url": original_doc["source_url"],
                "chunk": chunks[idx],
                "content": original_doc["content"],
            })
        end_classify = time.time()
        total_classify = end_classify - start_classify
        logger.info(
            f"Classification (stream_generate) took {total_classify:.2f} seconds")
        logger.info("Main function completed successfully")
        save_file(query, f"{output_dir}/query.md")
        save_file(results, f"{output_dir}/results.json")
        save_file(chunks, f"{output_dir}/chunks.json")
        end_total = time.time()
        total_time = end_total - start_total
        logger.info(
            f"Total execution time: {total_time:.2f} seconds")
        # Generate and save summary
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


if __name__ == "__main__":
    main()
