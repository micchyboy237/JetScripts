import os
from typing import List
from jet.file.utils import load_file, save_file
from jet.llm.rag.rag_preprocessor import MLXRAGProcessor, WebDataPreprocessor
from jet.logger import logger


def generate(query: str, chunks: List[str]) -> str:
    logger.info("Initializing MLXRAGProcessor")
    mlx_processor = MLXRAGProcessor(show_progress=True)
    logger.info("Generating embeddings for chunks")
    embeddings = mlx_processor.generate_embeddings(chunks)
    if embeddings.shape[0] != len(chunks):
        logger.error("Mismatch between chunks and embeddings, exiting")
        return

    logger.info(f"Processing query with generate: {query}")
    response = mlx_processor.generate(query, chunks, embeddings)
    return response


def stream_generate(query: str, chunks: List[str], top_k: int = 5) -> str:
    logger.info("Initializing MLXRAGProcessor")
    mlx_processor = MLXRAGProcessor(show_progress=True)
    logger.info("Generating embeddings for chunks")
    embeddings = mlx_processor.generate_embeddings(chunks)
    if embeddings.shape[0] != len(chunks):
        logger.error("Mismatch between chunks and embeddings, exiting")
        return

    logger.info(f"Processing query with stream_generate: {query}")
    logger.info("\nStreaming Responses:")
    response = ""
    for i, stream_response in enumerate(mlx_processor.stream_generate(query, chunks, embeddings, top_k=top_k), 1):
        logger.success(f"Stream Response {i}: {stream_response}")
        response += stream_response

    return response


def main():
    """Main function to demonstrate preprocessing and MLX RAG usage with streaming."""
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/chunked_docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    try:
        # query = "What is the main topic of the webpage?"
        # logger.info("Initializing WebDataPreprocessor")
        # preprocessor = WebDataPreprocessor(chunk_size=500, chunk_overlap=50)
        # url = "https://example.com"
        # logger.info(f"Preprocessing content from {url}")
        # chunks = preprocessor.preprocess(url)
        # if not chunks:
        #     logger.error("No chunks generated, exiting")
        #     return
        # logger.info(f"Generated {len(chunks)} chunks from {url}")

        docs = load_file(docs_file)
        query = docs["query"]
        chunks = [doc["text"] for doc in docs["results"]]
        top_k = 10

        response = stream_generate(query, chunks, top_k=top_k)
        print(f"Query: {query}")
        print(f"Single Response: {response}")
        print(f"Number of chunks processed: {len(chunks)}")

        logger.info("Main function completed successfully")

        save_file(query, f"{output_dir}/query.md")
        save_file(response, f"{output_dir}/response.md")
        save_file(chunks, f"{output_dir}/chunks.json")
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
