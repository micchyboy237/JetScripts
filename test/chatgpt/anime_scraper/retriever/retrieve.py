import sqlite3
import faiss
from jet.llm.utils.embeddings import get_embedding_function
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

DB_PATH = "data/articles.db"
FAISS_PATH = "data/faiss_index"

embed_model = "mxbai-embed-large"
embed_text = get_embedding_function(embed_model)


def get_bm25_results(query, k=3):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, content FROM articles")
    docs = cursor.fetchall()
    conn.close()

    tokenized_corpus = [word_tokenize(doc[1].lower()) for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(word_tokenize(query.lower()))
    ranked_docs = sorted(
        zip(docs, scores), key=lambda x: x[1], reverse=True)[:k]
    return [doc[0] for doc in ranked_docs]


def get_faiss_results(query_embedding, k=3):
    index = faiss.read_index(FAISS_PATH)
    D, I = index.search(np.array([query_embedding]), k)
    return I[0]


def hybrid_search(query):
    query_embedding = embed_text(query)
    faiss_results = get_faiss_results(query_embedding)
    bm25_results = get_bm25_results(query)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    results = []

    for doc_id in set(faiss_results).union(set(bm25_results)):
        cursor.execute(
            "SELECT title, content, url FROM articles WHERE id=?", (doc_id,))
        results.append(cursor.fetchone())

    conn.close()
    return results
