from jet.libs.context_engineering.course._01_context_retrieval_generation.labs.knowledge_retrieval_lab import (
    AdvancedVectorDatabase, Document, create_example_dir, get_example_logger
)
from jet.file.utils import save_file
import numpy as np

class CustomSimilarityDB(AdvancedVectorDatabase):
    def _boosted_cosine(self, query_vec, doc_vecs):
        similarities = np.dot(doc_vecs, query_vec)
        # Boost documents with keywords
        boosts = np.array([
            1.5 if "transformer" in d.content.lower() else 1.0
            for d in self.documents
        ])
        return similarities * boosts

    def search(self, query_embedding, top_k=5, **kwargs):
        if self.embeddings is None:
            return []
        similarities = self._boosted_cosine(query_embedding, self.embeddings)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.documents[i], float(similarities[i])) for i in top_indices]

def practical_03_custom_similarity():
    example_dir = create_example_dir("practical_03_custom_similarity")
    log = get_example_logger("Practical 3: Custom Similarity (Keyword Boost)", example_dir)
    log.info("PRACTICAL 3: Hybrid Semantic + Keyword Boost")

    docs = [
        Document("t1", "The Transformer model uses self-attention.", "Transformer Paper"),
        Document("t2", "BERT is a bidirectional transformer.", "BERT"),
        Document("g1", "GPT is autoregressive and generative.", "GPT"),
        Document("o1", "Old neural networks used RNNs.", "Legacy"),
    ]
    # Dummy embeddings
    embeddings = np.random.rand(len(docs), 384)
    db = CustomSimilarityDB(embedding_dim=384)
    db.add_documents(docs, embeddings)

    query_emb = np.random.rand(384)
    results = db.search(query_emb, top_k=3)

    save_file([{"title": d.title, "score": s} for d, s in results], f"{example_dir}/boosted_results.json")
    log.info("PRACTICAL 3 COMPLETE – Transformer docs boosted!")
    log.info("="*90)

if __name__ == "__main__":
    practical_03_custom_similarity()
