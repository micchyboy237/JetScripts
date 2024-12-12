import os
from llama_index.core import download_loader, VectorStoreIndex
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core.postprocessor import SentenceEmbeddingOptimizer
import time


class Document:
    """A class to represent a document."""

    def __init__(self, id: str, text: str):
        """
        Initialize a Document object.

        Args:
            id (str): The ID of the document.
            text (str): The text content of the document.
        """
        self.id = id
        self.text = text


class SentenceEmbeddingOptimizer:
    """A class to represent a sentence embedding optimizer."""

    def __init__(self, percentile_cutoff: float = 0.5, threshold_cutoff: float = None):
        """
        Initialize a SentenceEmbeddingOptimizer object.

        Args:
            percentile_cutoff (float): The percentile cutoff value. Defaults to 0.5.
            threshold_cutoff (float): The threshold cutoff value. If not provided, uses the percentile cutoff.
        """
        self.percentile_cutoff = percentile_cutoff
        self.threshold_cutoff = threshold_cutoff

    def optimize(self, documents: list[Document]) -> list[Document]:
        """
        Optimize a list of documents using sentence embeddings.

        Args:
            documents (list[Document]): A list of Document objects.

        Returns:
            list[Document]: The optimized list of documents.
        """
        # Calculate the similarity scores for each document
        similarity_scores = []
        for document in documents:
            score = self.calculate_similarity_score(document)
            similarity_scores.append((document, score))

        # Sort the documents based on their similarity scores
        sorted_documents = sorted(
            similarity_scores, key=lambda x: x[1], reverse=True)

        # Select the top-scoring documents based on the cutoff value
        if self.threshold_cutoff is not None:
            threshold = self.threshold_cutoff
        else:
            threshold = self.percentile_cutoff

        optimized_documents = [document for document, score in sorted_documents[:int(
            len(sorted_documents) * threshold)]]

        return optimized_documents

    def calculate_similarity_score(self, document: Document) -> float:
        """
        Calculate the similarity score for a given document.

        Args:
            document (Document): A Document object.

        Returns:
            float: The similarity score.
        """
        # This is a placeholder function and should be replaced with actual implementation
        return 0.5


def main():
    # Load Wikipedia data
    loader = WikipediaReader()
    documents = loader.load_data(pages=["Berlin"])

    # Create a vector store index from the documents
    index = VectorStoreIndex.from_documents(documents)

    # Query the index without optimization
    print("Without optimization")
    start_time = time.time()
    query_engine = index.as_query_engine()
    res = query_engine.query("What is the population of Berlin?")
    end_time = time.time()
    print("Total time elapsed: {}".format(end_time - start_time))
    print("Answer: {}".format(res))

    # Query the index with optimization
    print("With optimization")
    start_time = time.time()
    query_engine = index.as_query_engine(
        node_postprocessors=[SentenceEmbeddingOptimizer(percentile_cutoff=0.5)]
    )
    res = query_engine.query("What is the population of Berlin?")
    end_time = time.time()
    print("Total time elapsed: {}".format(end_time - start_time))
    print("Answer: {}".format(res))

    # Query the index with alternative optimization cutoff
    print("Alternate optimization cutoff")
    start_time = time.time()
    query_engine = index.as_query_engine(
        node_postprocessors=[SentenceEmbeddingOptimizer(threshold_cutoff=0.7)]
    )
    res = query_engine.query("What is the population of Berlin?")
    end_time = time.time()
    print("Total time elapsed: {}".format(end_time - start_time))
    print("Answer: {}".format(res))


if __name__ == "__main__":
    main()
