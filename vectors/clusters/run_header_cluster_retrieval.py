import os
import shutil
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.vectors.clusters.retrieval import VectorRetriever, RetrievalConfig
from jet.code.markdown_types.markdown_parsed_types import HeaderDoc, HeaderSearchResult
from typing import List, Optional


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def create_sample_header_docs() -> List[HeaderDoc]:
    """Create sample HeaderDoc objects for demonstration."""
    return [
        {
            "id": "doc1",
            "doc_index": 0,
            "header": "Introduction to AI",
            "content": "Artificial Intelligence (AI) is a field of computer science focused on building systems that mimic human intelligence. This includes machine learning, natural language processing, and robotics.",
            "level": 1,
            "parent_headers": [],
            "parent_header": None,
            "parent_level": None,
            "source": "ai_guide.md",
            "tokens": []
        },
        {
            "id": "doc2",
            "doc_index": 1,
            "header": "Machine Learning Basics",
            "content": "Machine learning is a subset of AI where models learn from data to make predictions or decisions. Common algorithms include neural networks and decision trees.",
            "level": 2,
            "parent_headers": ["Introduction to AI"],
            "parent_header": "Introduction to AI",
            "parent_level": 1,
            "source": "ai_guide.md",
            "tokens": []
        },
        {
            "id": "doc3",
            "doc_index": 2,
            "header": "Natural Language Processing",
            "content": "NLP enables machines to understand and generate human language. Applications include chatbots and sentiment analysis.",
            "level": 2,
            "parent_headers": ["Introduction to AI"],
            "parent_header": "Introduction to AI",
            "parent_level": 1,
            "source": "ai_guide.md",
            "tokens": []
        }
    ]


def search_headers(
    query: str,
    header_docs: List[HeaderDoc],
    top_k: Optional[int] = None,
    threshold: float = 0.0,
    embed_model: EmbedModelType = "all-MiniLM-L6-v2",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    buffer: int = 0,
) -> List[HeaderSearchResult]:
    """Demonstrate usage of VectorRetriever's search_headers function."""
    # Initialize VectorRetriever with configuration
    config = RetrievalConfig(
        embed_model=embed_model,
        min_cluster_size=2,
        k_clusters=3,
        top_k=top_k,
        cluster_threshold=20,
        threshold=threshold
    )
    retriever = VectorRetriever(config)
    logger.info("Initialized VectorRetriever with model: %s",
                config.embed_model)

    # Perform header search
    results = retriever.search_headers(
        header_docs=header_docs,
        query=query,
        top_k=top_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        buffer=buffer,
        threshold=threshold,
    )

    # Display results
    if not results:
        logger.info("No results found for query: %s", query)
        return []

    logger.info("Found %d results for query: %s", len(results), query)
    for result in results:
        logger.info(
            "Rank: %d, Score: %.4f, Header: %s, Content: %s... (Doc Index: %d, Source: %s)",
            result["rank"],
            result["score"],
            result["header"],
            result["content"][:100],  # Truncate content for readability
            result["metadata"]["doc_index"],
            result["metadata"]["source"]
        )

    return results


if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_4/top_isekai_anime_2025/docs.json"
    header_docs: List[HeaderDoc] = load_file(docs_file)["documents"]

    # Define a sample query
    query = "Top isekai anime 2025."
    embed_model: EmbedModelType = "all-MiniLM-L6-v2"
    top_k = None

    results = search_headers(
        query, header_docs, embed_model=embed_model, top_k=top_k)

    save_file({
        "query": query,
        "count": len(results),
        "total_tokens": sum(result["metadata"]["num_tokens"] for result in results),
        "results": results
    }, f"{OUTPUT_DIR}/results.json")
