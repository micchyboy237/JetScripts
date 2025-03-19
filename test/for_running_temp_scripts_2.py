import string
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex
from sklearn.decomposition import TruncatedSVD


# Preprocessing function
def preprocess(text):
    return text.translate(str.maketrans('', '', string.punctuation)).lower().split()


# BM25 Reranking
def bm25_rerank(query, documents):
    corpus = [preprocess(doc) for doc in documents]
    query_tokens = preprocess(query)
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)
    return sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)


# TF-IDF Reranking
def tfidf_rerank(query, documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents + [query])
    query_vector = tfidf_matrix[-1]
    document_vectors = tfidf_matrix[:-1]
    scores = (document_vectors @ query_vector.T).toarray().flatten()
    return sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)


# Cosine Similarity Reranking
def cosine_similarity_rerank(query, documents):
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(documents + [query])
    scores = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    return sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)


# Annoy Reranking
def annoy_rerank(query, documents, num_trees=10):
    vector_length = len(query.split())
    annoy_index = AnnoyIndex(vector_length, 'angular')

    def doc_vector(doc):
        return np.array([doc.count(word) for word in query.split()])

    for idx, doc in enumerate(documents):
        annoy_index.add_item(idx, doc_vector(doc).tolist())

    annoy_index.build(num_trees)
    query_vector = doc_vector(query)
    nearest_neighbors = annoy_index.get_nns_by_vector(
        query_vector.tolist(), len(documents), include_distances=True)

    return sorted(zip(documents, nearest_neighbors[1]), key=lambda x: x[1])


# Word Movers Distance (WMD) Reranking
def wmd_rerank(query, documents):
    vectorizer = CountVectorizer()
    all_text = documents + [query]
    vectors = vectorizer.fit_transform(all_text).toarray()
    query_vector = vectors[-1]
    document_vectors = vectors[:-1]
    distances = np.linalg.norm(document_vectors - query_vector, axis=1)
    return sorted(zip(documents, distances), key=lambda x: x[1])


# Latent Semantic Analysis (LSA) Reranking
def lsa_rerank(query, documents, n_components=2):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents + [query])
    svd = TruncatedSVD(n_components=n_components)
    lsa_matrix = svd.fit_transform(tfidf_matrix)
    query_lsa = lsa_matrix[-1]
    document_lsa = lsa_matrix[:-1]
    scores = (document_lsa @ query_lsa.T).flatten()
    return sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)


# Example Test
documents = [
    "The quick brown fox jumps over the lazy dog",
    "A fast fox leaps over a sleepy dog",
    "Quick foxes are speedy animals",
    "The fox was quick and ran fast",
    "Lazy dogs sleep all day long"
]
query = "quick fox"

print("BM25 Rerank:", bm25_rerank(query, documents))
print("TF-IDF Rerank:", tfidf_rerank(query, documents))
print("Cosine Similarity Rerank:", cosine_similarity_rerank(query, documents))
print("Annoy Rerank:", annoy_rerank(query, documents))
print("WMD Rerank:", wmd_rerank(query, documents))
print("LSA Rerank:", lsa_rerank(query, documents))
