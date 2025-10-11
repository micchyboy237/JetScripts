from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from typing import List
from jet.code.markdown_types import HeaderDoc
from jet.code.markdown_utils._preprocessors import clean_markdown_links
from jet.utils.url_utils import clean_links
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.wordnet.text_chunker import chunk_texts
import os
import shutil

OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

def load_sample_data():
    """Load sample dataset from local for topic modeling."""
    embed_model = "embeddinggemma"
    headers_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_isekai_anime_2025/all_headers.json"
    
    logger.info("Loading sample dataset...")
    headers_dict = load_file(headers_file)
    # headers: List[HeaderDoc] = [h for h_list in headers_dict.values() for h in h_list]
    headers: List[HeaderDoc] = headers_dict["https://gamerant.com/new-isekai-anime-2025"]
    documents = [f"{doc["header"]}\n\n{doc['content']}" for doc in headers]

    # Clean all links
    documents = [clean_markdown_links(doc) for doc in documents]
    documents = [clean_links(doc) for doc in documents]

    documents = chunk_texts(
        documents,
        chunk_size=64,
        chunk_overlap=32,
        model=embed_model,
    )
    save_file(documents, f"{OUTPUT_DIR}/documents.json")
    return documents

def search_tfidf(texts: list[str], query: str, top_n: int = 5) -> list[tuple[str, float]]:
    """Search using TF-IDF with cosine similarity."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(
        texts + [query])  # Include query for comparison
    query_vector = tfidf_matrix[-1]  # Last vector is the query
    similarity_scores = cosine_similarity(
        query_vector, tfidf_matrix[:-1]).flatten()

    top_indices = np.argsort(similarity_scores)[::-1][:top_n]
    return [(texts[i], similarity_scores[i]) for i in top_indices]

# Example Usage
if __name__ == "__main__":
    # texts = ["Machine learning is great", "Deep learning advances AI",
    #         "Natural language processing is cool"]
    texts = load_sample_data()
    query = "Top isekai anime 2025"
    result = search_tfidf(texts, query)
    save_file({
        "query": query,
        "count": len(result),
        "result": result,
    }, f"{OUTPUT_DIR}/tfidf_labels.json")
