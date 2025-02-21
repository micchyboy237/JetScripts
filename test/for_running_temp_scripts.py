from typing import List, Tuple
from jet.logger import logger
from sentence_transformers import SentenceTransformer, util
import torch

# Initialize the model
model = SentenceTransformer('all-MiniLM-L12-v2')

# Define queries and documents
queries = [
    "Uses only React.js",
    "Not React.js",
    "For native iOS development"
]
documents = [
    "React Native is a framework for building mobile apps.",
    "Flutter and Swift are alternatives to React Native.",
    "React is a JavaScript library for building UIs.",
    "Node.js is used for backend development.",
]


def filter_documents_by_query(queries: List[str], documents: List[str], threshold: float = 0.5) -> List[Tuple[str, List[Tuple[str, float]]]]:
    """
    Filters documents based on their relevance to each query, including similarity scores.

    Args:
        queries (List[str]): List of query strings.
        documents (List[str]): List of document strings.
        threshold (float): Cosine similarity threshold for relevance.

    Returns:
        List[Tuple[str, List[Tuple[str, float]]]]: A list of tuples where each tuple contains a query and 
        a list of relevant documents along with their similarity scores.
    """
    # Encode queries and documents to get their embeddings
    query_embeddings = model.encode(queries, convert_to_tensor=True)
    document_embeddings = model.encode(documents, convert_to_tensor=True)

    results = []
    for query, query_embedding in zip(queries, query_embeddings):
        # Compute cosine similarities between the query and all documents
        cosine_scores = util.cos_sim(query_embedding, document_embeddings)[0]

        # Find documents that exceed the similarity threshold, including scores
        relevant_docs = [
            (doc, float(score)) for doc, score in zip(documents, cosine_scores) if score >= threshold
        ]
        # Sort documents by similarity score in descending order
        relevant_docs.sort(key=lambda x: x[1], reverse=True)
        results.append((query, relevant_docs))

    return results


if __name__ == "__main__":
    filtered_results = filter_documents_by_query(
        queries, documents, threshold=0.5)
    for query, docs in filtered_results:
        logger.newline()
        logger.info(f"Query: '{query}'")
        if docs:
            logger.debug(f"Relevant Documents ({len(docs)}):")
            for doc, score in docs:
                logger.log("Score:", f"{score:.4f}",
                           colors=["GRAY", "SUCCESS"])
                logger.log("Doc:", doc, colors=["GRAY", "DEBUG"])
        else:
            logger.warning("No relevant documents found.")

    # Define expected results for assertions (including similarity scores will vary, so we check document presence only)
    expected_results = {
        "Uses only React.js": ["React is a JavaScript library for building UIs."],
        "Not React.js": [
            "React Native is a framework for building mobile apps.",
            "Flutter and Swift are alternatives to React Native.",
            "Node.js is used for backend development."
        ],
        "For native iOS development": ["Flutter and Swift are alternatives to React Native."]
    }

    # Assertions to verify the correctness
    for query, expected_docs in expected_results.items():
        for result_query, result_docs in filtered_results:
            if result_query == query:
                # Extract document texts only
                actual_docs = [doc for doc, _ in result_docs]
                assert set(actual_docs) == set(
                    expected_docs), f"Assertion failed for query: {query}"
                logger.success(f"Assertion passed for query: '{query}'")
