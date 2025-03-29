import faiss
from jet.actions.faiss_search import faiss_search
from jet.logger import logger, time_it


if __name__ == "__main__":
    candidates = [
        "The quick brown fox jumps over the lazy dog",
        "A fast fox jumped over a lazy dog",
        "Hello world, how are you?",
        "Data science and machine learning are fascinating",
        "Artificial intelligence is transforming the world"
    ]
    queries = [
        "What is artificial intelligence?",
        "Tell me about data science and machine learning",
        "What does the quick brown fox do?"
    ]
    top_k = 3
    nlist = 100

    results = faiss_search(queries, candidates, top_k=top_k, nlist=nlist)

    # Display results
    logger.log("\nQuery Results:", f"{
               len(results)}", colors=["DEBUG", "SUCCESS"])
    for query, group in results.items():
        logger.info(f"\nQuery: {query}")
        for result in group:
            logger.log(f"{result['text']}:", f"{
                       result['score']:.4f}", colors=["DEBUG", "SUCCESS"])
