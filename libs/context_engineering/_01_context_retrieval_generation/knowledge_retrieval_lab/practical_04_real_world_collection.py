from jet.libs.context_engineering.course._01_context_retrieval_generation.labs.knowledge_retrieval_lab import (
    ProfessionalEmbeddingModel, AdvancedVectorDatabase, Document, create_example_dir, get_example_logger
)

def load_arxiv_papers():
    # Simulated real-world arXiv papers (you can replace with real JSON)
    return [
        {"id": "2106.0001", "title": "LoRA: Low-Rank Adaptation", "abstract": "We propose Low-Rank Adaptation..."},
        {"id": "2305.1234", "title": "QLoRA: Efficient Finetuning", "abstract": "QLoRA reduces memory..."},
        {"id": "2005.14165", "title": "GPT-3 Paper", "abstract": "Language Models are Few-Shot Learners..."},
    ]

def practical_04_real_world_collection():
    example_dir = create_example_dir("practical_04_real_world")
    log = get_example_logger("Practical 4: Real-World arXiv Collection", example_dir)
    log.info("PRACTICAL 4: Indexing Real Research Papers")

    papers = load_arxiv_papers()
    docs = [
        Document(
            id=p["id"],
            content=p["abstract"],
            title=p["title"],
            metadata={"source": "arXiv", "year": 2023}
        ) for p in papers
    ]

    model = ProfessionalEmbeddingModel("all-MiniLM-L6-v2")
    embeddings = model.encode([d.content for d in docs])
    db = AdvancedVectorDatabase(embedding_dim=384, index_type="HNSW")
    db.add_documents(docs, embeddings)
    db.save_index(f"{example_dir}/arxiv_knowledge_base")

    log.info(f"Indexed {len(docs)} real papers → saved to disk")
    log.info("PRACTICAL 4 COMPLETE – Production-ready knowledge base")
    log.info("="*90)

if __name__ == "__main__":
    practical_04_real_world_collection()
