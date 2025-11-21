from jet.libs.context_engineering.course._01_context_retrieval_generation.labs.knowledge_retrieval_lab import (
    ProfessionalEmbeddingModel, AdvancedVectorDatabase, Document, create_example_dir, get_example_logger
)
from jet.file.utils import save_file

def practical_01_different_embedding_models():
    example_dir = create_example_dir("practical_01_embedding_comparison")
    log = get_example_logger("Practical 1: Compare Embedding Models", example_dir)
    log.info("PRACTICAL 1: Testing Multiple Embedding Models")

    models_to_test = [
        "all-MiniLM-L6-v2",      # Fast, general-purpose
        "multi-qa-mpnet-base-dot-v1",  # High-quality, QA-optimized
        "paraphrase-multilingual-MiniLM-L12-v2",  # Multilingual
    ]

    docs = [Document(f"doc{i}", f"This document is about AI topic {i}.", f"Doc {i}") for i in range(10)]
    results = {}

    for model_name in models_to_test:
        log.info(f"Loading model: {model_name}")
        model = ProfessionalEmbeddingModel(model_name)
        embeddings = model.encode([d.content for d in docs])
        db = AdvancedVectorDatabase(embedding_dim=model.embedding_dim, index_type="Flat")
        db.add_documents(docs, embeddings)

        query = "Tell me about artificial intelligence"
        q_emb = model.encode_single(query)
        hits = db.search(q_emb, top_k=3)

        results[model_name] = {
            "dim": model.embedding_dim,
            "top_results": [(d.title, float(s)) for d, s in hits]
        }

    save_file(results, f"{example_dir}/embedding_model_comparison.json")
    log.info("PRACTICAL 1 COMPLETE – Multi-model comparison saved")
    log.info("="*90)

if __name__ == "__main__":
    practical_01_different_embedding_models()
