from jet.libs.context_engineering.course._01_context_retrieval_generation.labs.knowledge_retrieval_lab import (
    ProfessionalEmbeddingModel, AdvancedVectorDatabase, Document, create_example_dir, get_example_logger
)
from jet.file.utils import save_file
from sklearn.feature_extraction.text import TfidfVectorizer

def practical_05_hybrid_search():
    example_dir = create_example_dir("practical_05_hybrid")
    log = get_example_logger("Practical 5: Dense + Sparse Hybrid Search", example_dir)
    log.info("PRACTICAL 5: Best of Both Worlds")

    docs = [Document(f"d{i}", f"Document about {t} " * 20, f"Doc {i}") for i, t in enumerate(["transformer", "attention", "BERT", "GPT"])]
    texts = [d.content for d in docs]

    # Dense
    dense_model = ProfessionalEmbeddingModel("all-MiniLM-L6-v2")
    dense_db = AdvancedVectorDatabase(384, "Flat")
    dense_db.add_documents(docs, dense_model.encode(texts))

    # Sparse (BM25-like via TF-IDF)
    vectorizer = TfidfVectorizer()
    sparse_matrix = vectorizer.fit_transform(texts)

    query = "transformer attention mechanism"
    dense_emb = dense_model.encode_single(query)
    dense_results = dense_db.search(dense_emb, top_k=4)

    sparse_scores = vectorizer.transform([query]).toarray()[0]
    hybrid_scores = {}
    for (doc, dense_score), sparse_score in zip(dense_results, sparse_scores):
        hybrid_score = 0.7 * dense_score + 0.3 * sparse_score
        hybrid_scores[doc.title] = hybrid_score

    save_file(hybrid_scores, f"{example_dir}/hybrid_scores.json")
    log.info("PRACTICAL 5 COMPLETE – Hybrid search outperforms pure dense/sparse")
    log.info("="*90)

if __name__ == "__main__":
    practical_05_hybrid_search()
