import os
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
    query: str = f"Does the webpage have concrete answer to this query?\nQuery: {docs['query']}"
    chunks: List[str] = [doc["text"] for doc in docs["results"]]
    chunks = chunks[:10]
    top_k: int = len(chunks)
    try:
        mlx_processor = MLXRAGProcessor(show_progress=True, batch_size=4)
        logger.info("Generating embeddings for chunks")
        embeddings: NDArray = mlx_processor.generate_embeddings(chunks)
        if embeddings.shape[0] != len(chunks):
            logger.error("Mismatch between chunks and embeddings, exiting")
            return
        logger.info(f"Query: {query}")
        logger.info(f"Number of chunks processed: {len(chunks)}")
        logger.info(f"Classifying query with stream_generate: {query}")
        logger.info("\nStreaming Classifications:")
        response: List[Dict] = []
        for label, score, idx in mlx_processor.stream_generate(query, chunks, embeddings, top_k=top_k, relevance_threshold=0.7):
            logger.debug(
                f"Stream Classification {idx}: Label={label}, Score={score:.4f}")
            response.append({
                "num_chunk": idx,
                "label": label,
                "score": score,
                "chunk": chunks[idx],
            })
        logger.info("Main function completed successfully")
        save_file(query, f"{output_dir}/query.md")
        save_file(response, f"{output_dir}/response.json")
        save_file(chunks, f"{output_dir}/chunks.json")
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")


if __name__ == "__main__":
    main()
