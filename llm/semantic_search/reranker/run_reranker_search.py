from langchain_core.callbacks import CallbackManagerForRetrieverRun
from jet.db.chroma import ChromaClient, InitialDataEntry
from jet.llm.ollama import OllamaEmbeddingFunction
from jet.llm.helpers.semantic_search import (
    RerankerRetriever
)
from jet.logger import logger

# Function to initialize the retriever


def initialize_retriever(data: list[str] | list[InitialDataEntry], use_ollama: bool = True) -> RerankerRetriever:
    if use_ollama:
        embed_model = "nomic-embed-text"
        rerank_model = "mxbai-embed-large"
    else:
        embed_model = "sentence-transformers/all-MiniLM-L6-v2"
        rerank_model = "sentence-transformers/all-MiniLM-L6-v2"

    retriever = RerankerRetriever(
        data=data,
        use_ollama=use_ollama,
        collection_name="example_collection",
        embed_model=embed_model,
        rerank_model=rerank_model,
        embed_batch_size=32,
        overwrite=True
    )
    return retriever

# Function to perform a simple search query


def search_query(retriever: RerankerRetriever, query: str, top_k: int):
    results = retriever.search(query, top_k=top_k)
    return results

# Function to perform search with reranking


def search_with_reranking(retriever: RerankerRetriever, query: str, top_k: int, rerank_threshold: float):
    reranked_results = retriever.search_with_reranking(
        query, top_k=top_k, rerank_threshold=rerank_threshold)
    return reranked_results

# Main function to tie everything together


def main():
    # Example data and query
    data = [
        InitialDataEntry(id="1", document="Sample document content.", metadata={
                         "source": "example"}),
        InitialDataEntry(id="2", document="Another document.", metadata={
                         "source": "example"}),
    ]

    query = "Sample document"
    top_k = 10
    rerank_threshold = 0.3
    use_ollama = False

    # Initialize the retriever
    retriever = initialize_retriever(data, use_ollama=use_ollama)

    # Perform a search query
    search_results = search_query(retriever, query, top_k=top_k)
    logger.info("\n--- Search Results ---")
    logger.info(f"\nQuery: {query}")
    for result in search_results:
        logger.log(f"{result['document']}:", f"{
            result['score']:.4f}", colors=["DEBUG", "SUCCESS"])
    # Perform search with reranking
    search_results_with_reranking = search_with_reranking(
        retriever, query, top_k=top_k, rerank_threshold=rerank_threshold)
    logger.info("\n--- Search Results w/ Reranking ---")
    logger.info(f"\nQuery: {query}")
    for result in search_results_with_reranking:
        logger.log(f"{result['document']}:", f"{
            result['score']:.4f}", colors=["DEBUG", "SUCCESS"])


# Run the main function
if __name__ == "__main__":
    main()
