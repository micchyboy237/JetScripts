from typing import Callable, Optional
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from jet.llm.ollama.base import OllamaEmbedding
from jet.logger import logger
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.utils import set_global_tokenizer
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import Document
import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")


class SpacyLemmatizer:
    """A wrapper to use spaCy lemmatization in place of stemming."""

    def __init__(self):
        self.nlp = nlp  # Use the globally loaded spaCy model

    def stemWords(self, words):
        """Lemmatize a list of words using spaCy."""
        return [token.lemma_ for token in self.nlp(" ".join(words))]

    def __call__(self, text: str):
        """Make instance callable to lemmatize a text input."""
        return [token.lemma_ for token in self.nlp(text)]


def get_bm25_retriever(
    index: VectorStoreIndex,
    similarity_k: int = 10,
    language: str = "en",
    verbose: bool = False,
    skip_stemming: bool = False,
    # token_pattern: str = r"(?u)\b[\w.]+\b",
    token_pattern: str = r"(?u)\b\w\w+\b",
    stemmer: Optional[Callable] = None
):
    """
    Initialize a BM25Retriever with specified parameters.

    Args:
        index (VectorStoreIndex): The vector store index to retrieve documents from.
        similarity_k (int): The number of top similar results to return.
        language (str): The language for stopword removal (default: "en").
        verbose (bool): Whether to enable verbose mode (default: False).
        skip_stemming (bool): Whether to skip stemming/lemmatization (default: False).
        token_pattern (str): Tokenization pattern (default: r"(?u)\b\w\w+\b").

    Returns:
        BM25Retriever: Configured BM25 retriever instance.
    """
    stemmer = stemmer or SpacyLemmatizer()
    retriever = BM25Retriever.from_defaults(
        index=index,
        similarity_top_k=similarity_k,
        language=language,
        verbose=verbose,
        skip_stemming=skip_stemming,
        token_pattern=token_pattern,
        stemmer=stemmer  # Pass the custom lemmatizer
    )

    return retriever


# Example usage
if __name__ == "__main__":
    queries = [
        "Native",
        "No React.js",
        "For iOS/Android development"
    ]
    candidates = [
        "React Native is a framework for building mobile apps.",
        "Flutter and Swift are alternatives to React Native.",
        "React.js is a JavaScript library for building UIs.",
        "Node.js is used for backend development."
    ]
    chunk_size = 256
    chunk_overlap = 0
    top_k = len(candidates)

    embed_model = "nomic-embed-text"
    stemmer = SpacyLemmatizer()
    set_global_tokenizer(stemmer)

    # Mock data
    documents = [Document(text=text) for text in candidates]
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    all_nodes = splitter.get_nodes_from_documents(
        documents, show_progress=True)
    index = VectorStoreIndex(
        all_nodes,
        show_progress=True,
        embed_model=OllamaEmbedding(model_name=embed_model),
    )
    retriever = get_bm25_retriever(
        index, similarity_k=top_k, verbose=True, stemmer=stemmer)

    logger.newline()
    for query in queries:
        results = retriever.retrieve(query)
        logger.log("Query:", query, colors=["GRAY", "DEBUG"])
        for result in results:
            logger.success(f" - {result.node.get_content()}")
        logger.newline()

    assert isinstance(
        retriever, BM25Retriever), "Retriever is not an instance of BM25Retriever"
