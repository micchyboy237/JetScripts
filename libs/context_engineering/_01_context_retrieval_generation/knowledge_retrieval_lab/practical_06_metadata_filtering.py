from jet.libs.context_engineering.course._01_context_retrieval_generation.labs.knowledge_retrieval_lab import (
    ProfessionalEmbeddingModel, AdvancedVectorDatabase, Document, create_example_dir, get_example_logger
)
from jet.file.utils import save_file

def practical_06_metadata_filtering():
    example_dir = create_example_dir("practical_06_filtering")
    log = get_example_logger("Practical 6: Metadata Filtering", example_dir)
    log.info("PRACTICAL 6: Category + Difficulty Filtering")

    model = ProfessionalEmbeddingModel()
    docs = [
        Document("d1", "Beginner ML", "Intro", metadata={"difficulty": "beginner", "category": "ml"}),
        Document("d2", "Advanced Transformers", "Deep", metadata={"difficulty": "advanced", "category": "nlp"}),
    ]
    embeddings = model.encode([d.content for d in docs])
    db = AdvancedVectorDatabase(384)
    db.add_documents(docs, embeddings)

    results = db.search(
        model.encode_single("simple explanation"),
        top_k=5,
        filter_metadata={"difficulty": "beginner"}
    )

    save_file([{
        "id": d.id,
        "score": score,
        "title": d.title,
        "content": d.content,
        "metadata": d.metadata,
    } for d, score in results], f"{example_dir}/searched_results.json")
    log.info("PRACTICAL 6 COMPLETE – Only beginner docs returned")

if __name__ == "__main__":
    practical_06_metadata_filtering()
