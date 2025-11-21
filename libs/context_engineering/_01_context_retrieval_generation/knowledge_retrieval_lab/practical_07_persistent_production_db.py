from jet.libs.context_engineering.course._01_context_retrieval_generation.labs.knowledge_retrieval_lab import (
    ProfessionalEmbeddingModel, AdvancedVectorDatabase, Document, create_example_dir, get_example_logger
)
from jet.file.utils import save_file

def practical_07_persistent_production_db():
    example_dir = create_example_dir("practical_07_production_db")
    log = get_example_logger("Practical 7: Persistent Production DB", example_dir)
    log.info("PRACTICAL 7: Save/Load Full Knowledge Base")

    # Build once
    model = ProfessionalEmbeddingModel()
    db = AdvancedVectorDatabase(384, "HNSW")
    # ... add 10k docs here in real use
    db.save_index(f"{example_dir}/production_knowledge_base")

    # Load in another process
    loaded = AdvancedVectorDatabase(384, "HNSW")
    loaded.load_index(f"{example_dir}/production_knowledge_base")

    log.info(f"Production DB ready → {len(loaded.documents)} docs loaded")
    log.info("PRACTICAL 7 COMPLETE – Zero-downtime deployment ready")

if __name__ == "__main__":
    practical_07_persistent_production_db()
