import os
from typing import List
from jet.vectors.semantic_search.text_vector_search import search_texts, TextSearchResult
from jet.models.model_types import EmbedModelType
from jet.logger import logger
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


def preprocess_text(text: str) -> str:
    """Simple preprocessing to lowercase text and remove extra whitespace."""
    return ' '.join(text.lower().split())


def main():
    # Sample texts to search through
    texts: List[str] = [
        "Python is a versatile programming language used for web development and data science.",
        "Machine learning involves training models to make predictions based on data patterns.",
        "Web frameworks like Django and Flask simplify backend development in Python.",
        "Deep learning is a subset of machine learning using neural networks."
    ]

    # Corresponding IDs for the texts
    text_ids: List[str] = ["doc_1", "doc_2", "doc_3", "doc_4"]

    # Search query
    query: str = "Python web development"

    # Configuration
    embed_model: EmbedModelType = 'all-MiniLM-L6-v2'
    chunk_size: int = 100
    chunk_overlap: int = 20
    top_k: int = len(texts)
    threshold: float = 0.5
    split_chunks = True

    try:
        # Perform the search
        results = list(search_texts(
            texts=texts,
            query=query,
            embed_model=embed_model,
            text_ids=text_ids,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            threshold=threshold,
            preprocess=preprocess_text,
            split_chunks=split_chunks
        ))

        # Process and display results
        for result in results[:10]:
            logger.info(f"Rank: {result['rank']}")
            logger.info(f"Score: {result['score']:.4f}")
            logger.info(f"Text ID: {result['metadata']['text_id']}")
            logger.info(f"Text Chunk: {result['text'][:100]}...")
            logger.info(f"Metadata: {result['metadata']}")
            logger.info("-" * 50)

        save_file({
            "query": query,
            "count": len(results),
            "results": results
        }, f"{OUTPUT_DIR}/search_results_split.json")

    except Exception as e:
        logger.error(f"Error during search: {str(e)}")


if __name__ == "__main__":
    main()
