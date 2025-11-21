import time
from jet.libs.context_engineering.course._01_context_retrieval_generation.labs.knowledge_retrieval_lab import (
    ProfessionalEmbeddingModel, AdvancedVectorDatabase, Document, create_example_dir, get_example_logger
)
from jet.file.utils import save_file

def practical_08_realtime_update_stream():
    example_dir = create_example_dir("practical_08_streaming")
    log = get_example_logger("Practical 8: Real-time Index Updates", example_dir)
    log.info("PRACTICAL 8: Streaming New Documents")

    db = AdvancedVectorDatabase(384, "HNSW")
    model = ProfessionalEmbeddingModel()

    new_docs = ["New paper on Mixture of Experts", "FlashAttention-3 released"]
    for doc_text in new_docs:
        emb = model.encode_single(doc_text)
        doc = Document(id=f"live_{int(time.time())}", content=doc_text, title="Live Update")
        db.add_documents([doc], emb.reshape(1, -1))
        log.info(f"Indexed live: {doc_text[:50]}...")

    save_file([d.title for d in db.documents], f"{example_dir}/documents.json")
    log.info("PRACTICAL 8 COMPLETE – Index stays fresh in real-time")

if __name__ == "__main__":
    practical_08_realtime_update_stream()
