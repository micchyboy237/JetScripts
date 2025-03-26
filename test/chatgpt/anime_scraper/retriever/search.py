import os
import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import nltk

# nltk.download("punkt")

# Ensure the index directory exists
index_path = "data/faiss_index/anime.index"
os.makedirs(os.path.dirname(index_path), exist_ok=True)

# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Check if the FAISS index file exists
if os.path.exists(index_path):
    index = faiss.read_index(index_path)
else:
    # Create a new FAISS index if it doesn't exist
    d = 384  # Dimension of embeddings (depends on the model)
    index = faiss.IndexFlatL2(d)
    faiss.write_index(index, index_path)  # Save the empty index

# Connect to SQLite database
conn = sqlite3.connect("data/top_upcoming_anime.db")
cursor = conn.cursor()
cursor.execute("SELECT title, synopsis FROM anime")
data = cursor.fetchall()

# Prepare BM25
tokenized_corpus = [nltk.word_tokenize(synopsis) for _, synopsis in data]
bm25 = BM25Okapi(tokenized_corpus)


def hybrid_search(query):
    query_embedding = np.array([model.encode(query)]).astype("float32")

    _, faiss_results = index.search(query_embedding, k=5)
    bm25_results = bm25.get_top_n(nltk.word_tokenize(query), data, n=5)

    return list(set(faiss_results[0]) | set(bm25_results))
