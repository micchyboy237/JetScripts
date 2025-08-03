import os
from typing import List, Tuple
from jet.file.utils import save_file
from jet.llm.llm_generator import LLMGenerator
from jet.logger import logger
from jet.logger.config import colorize_log
from jet.vectors.clusters.retrieval import RetrievalConfigDict, VectorRetriever

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


def main():
    """Runner script for retrieval and response generation."""
    config: RetrievalConfigDict = {
        "min_cluster_size": 2,
        "k_clusters": 2,
        "top_k":  None,
        "cluster_threshold": 20,
        "model_name": 'mxbai-embed-large',
        "cache_file":  None,
        "threshold": None,
    }
    corpus = [
        "Machine learning is a method of data analysis that automates model building.",
        "Supervised learning uses labeled data to train models for prediction.",
        "Unsupervised learning finds patterns in data without predefined labels.",
        "Deep learning is a subset of machine learning using neural networks.",
        "Natural language processing enables machines to understand human language.",
        "Computer vision allows machines to interpret and analyze visual data.",
        "Reinforcement learning trains agents to make decisions by rewards.",
        "Python is a popular language for machine learning and data science.",
        "SQL is used for managing and querying structured databases.",
        "Cloud computing provides scalable resources for machine learning tasks."
    ]
    query = "What is supervised learning in machine learning?"

    retriever = VectorRetriever(config)
    retriever.load_or_compute_embeddings(corpus)
    retriever.cluster_embeddings()
    retriever.build_index()
    top_chunks = retriever.retrieve_chunks(query)

    logger.info("\nTop relevant chunks for RAG:")
    for i, (chunk, score) in enumerate(top_chunks, 1):
        print(f"{colorize_log(f"{i}.", "ORANGE")} | {
              colorize_log(f"{score:.4f}", "SUCCESS")} | {chunk[:50]}")

    generator = LLMGenerator()
    response = generator.generate_response(
        query, top_chunks, generation_config={"verbose": True})

    save_file(top_chunks, f"{OUTPUT_DIR}/top_chunks.json")
    save_file(f"# LLM Generation\n\n## Prompt\n\n{response["prompt"]}\n\n## Response\n\n{response["response"]}",
              f"{OUTPUT_DIR}/llm_generation.md")


if __name__ == "__main__":
    main()
