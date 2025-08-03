import os
from jet.file.utils import save_file
from jet.vectors.clusters.retrieval import LLMGenerator, RetrievalConfig, VectorRetriever

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


def main():
    """Runner script for retrieval and response generation."""
    config = RetrievalConfig()
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

    generator = LLMGenerator()
    response = generator.generate_response(query, top_chunks)

    print("Query:", query)
    print("\nTop relevant chunks for RAG:")
    for i, (chunk, score) in enumerate(top_chunks, 1):
        print(f"{i}. {chunk} (Similarity: {score:.4f})")
    print("\nGenerated Response:")
    print(response)

    save_file(top_chunks, f"{OUTPUT_DIR}/top_chunks.json")
    save_file(f"# LLM Generation\n\n## Query\n\n{query}\n\n## Response\n\n{response}",
              f"{OUTPUT_DIR}/llm_generation.md")


if __name__ == "__main__":
    main()
