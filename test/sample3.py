from typing import Any, Optional
from bs4 import BeautifulSoup
from jet.file.utils import load_file
from jet.logger import logger
from jet.logger.timer import time_it
from jet.search.transformers import clean_string
from jet.token.token_utils import split_texts
from jet.utils.commands import copy_to_clipboard
from sentence_transformers import SentenceTransformer
import faiss


class BertSearch:
    @time_it
    def __init__(self, model_name="paraphrase-MiniLM-L12-v2"):
        """Initialize the BERT model and FAISS index."""
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.doc_texts = []

    def _extract_text(self, html_content):
        """Extracts text from raw HTML content using BeautifulSoup."""
        soup = BeautifulSoup(html_content, "html.parser")
        return " ".join(p.text.strip() for p in soup.find_all("p") if p.text.strip())

    @time_it
    def build_index(self, docs, batch_size=32):
        """Processes and indexes a list of HTML documents with batch encoding."""
        # self.doc_texts = [self._extract_text(doc) for doc in html_docs]
        self.doc_texts = split_texts(
            docs, self.model.tokenize, chunk_size=200, chunk_overlap=50)

        # Generate embeddings with batch processing
        embeddings = self.model.encode(
            self.doc_texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)

        # Create FAISS index using Inner Product (for cosine similarity)
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)

    @time_it
    def search(self, query: str, top_k: Optional[int] = None):
        """Searches the indexed documents for the best matches to the query."""
        if self.index is None:
            raise ValueError("Index is empty! Call build_index() first.")

        top_k = min(top_k or len(self.doc_texts), len(self.doc_texts))

        # Encode query using normalized embeddings
        query_embedding = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True)

        # Perform search
        scores, indices = self.index.search(query_embedding, top_k)

        # Return top-k filtered results
        results = [{"text": self.doc_texts[idx], "score": round(float(score), 4)}
                   for idx, score in zip(indices[0], scores[0]) if idx < len(self.doc_texts)]

        return results


if __name__ == "__main__":
    # data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/generated/search_web_data/scraped_texts.json"
    data = load_file(data_file)
    docs = []
    for item in data:
        cleaned_sentence = clean_string(item)
        docs.append(cleaned_sentence)

    # Sample HTML docs
    # docs = [
    #     "<html><body><p>AI is transforming the world with deep learning.</p></body></html>",
    #     "<html><body><p>Quantum computing is the future of high-performance computing.</p></body></html>",
    #     "<html><body><p>Neural networks are a crucial part of artificial intelligence.</p></body></html>"
    # ]

    # Initialize search system
    search_engine = BertSearch()

    # Index the documents
    search_engine.build_index(docs)

    # Perform a search
    query = "Season and episode of \"I'll Become a Villainess Who Goes Down in History\" anime"
    top_k = 10

    results = search_engine.search(query, top_k=top_k)

    copy_to_clipboard({
        "count": len(results),
        "data": results[:50]
    })

    for idx, result in enumerate(results[:10]):
        logger.log(f"{idx + 1}:", result["text"]
                   [:30], colors=["WHITE", "DEBUG"])
        logger.success(f"{result['score']:.2f}")
