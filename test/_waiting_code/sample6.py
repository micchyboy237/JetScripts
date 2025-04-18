import os
from typing import List, TypedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from jet.file.utils import load_file, save_file
from jet.scrapers.utils import clean_text
from jet.vectors.reranker.heuristics import bm25_plus_search, tfidf_search
from rank_bm25 import BM25Plus
import numpy as np


data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_server/generated/search/top_anime_romantic_comedy_reddit_2024-2025/top_context_nodes.json"
output_dir = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

data = load_file(data_file)


query = data["query"]
corpus = [clean_text(node["text"]) for node in data["results"]]


# --- Execute and Display Results ---
print("=== TF-IDF Results ===")
tfidf_results = tfidf_search(corpus, query)
for res in tfidf_results:
    print(
        f"Document {res['doc_index']}: {res['text']} (Score: {res['score']:.4f})")
save_file(tfidf_results, os.path.join(output_dir, "tfidf_results.json"))

print("\n=== BM25+ Results ===")
bm25_plus_results = bm25_plus_search(corpus, query)
for res in bm25_plus_results:
    print(
        f"Document {res['doc_index']}: {res['text']} (Score: {res['score']:.4f})")
save_file(bm25_plus_results, os.path.join(
    output_dir, "bm25_plus_results.json"))

# --- Optional: BM25+ Re-ranking after TF-IDF ---
print("\n=== BM25+ Re-ranking Top 2 TF-IDF Results ===")
top_tfidf_docs = tfidf_results[:2]
top_tfidf_indices = [res["doc_index"] for res in top_tfidf_docs]
top_tfidf_corpus = [corpus[idx] for idx in top_tfidf_indices]
hybrid_results = bm25_plus_search(top_tfidf_corpus, query)
# Remap to original indices
for res in hybrid_results:
    res["doc_index"] = top_tfidf_indices[res["doc_index"]]
    res["text"] = corpus[res["doc_index"]]
    print(
        f"Document {res['doc_index']}: {res['text']} (Score: {res['score']:.4f})")

save_file(hybrid_results, os.path.join(output_dir, "hybrid_results.json"))
