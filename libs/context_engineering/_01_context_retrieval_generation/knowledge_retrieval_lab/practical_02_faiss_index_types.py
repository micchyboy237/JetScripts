from jet.libs.context_engineering.course._01_context_retrieval_generation.labs.knowledge_retrieval_lab import (
    ProfessionalEmbeddingModel, AdvancedVectorDatabase, Document, PerformanceBenchmark,
    create_example_dir, get_example_logger
)
from jet.file.utils import save_file
import time

def practical_02_faiss_index_types():
    example_dir = create_example_dir("practical_02_index_types")
    log = get_example_logger("Practical 2: FAISS Index Comparison", example_dir)
    log.info("PRACTICAL 2: Flat vs IVF vs HNSW")

    model = ProfessionalEmbeddingModel("all-MiniLM-L6-v2")
    docs = [Document(f"doc{i}", f"Content about topic {i} " * 50, f"Doc {i}") for i in range(1000)]
    embeddings = model.encode([d.content for d in docs])

    index_types = ["Flat", "IVF", "HNSW"]
    results = {}

    for idx_type in index_types:
        log.info(f"Building {idx_type} index...")
        db = AdvancedVectorDatabase(embedding_dim=384, index_type=idx_type)
        start = time.time()
        db.add_documents(docs, embeddings)
        build_time = time.time() - start

        benchmark = PerformanceBenchmark()
        perf = benchmark.benchmark_search_performance(db, model, num_queries=100)

        results[idx_type] = {
            "build_time_s": build_time,
            "mean_search_ms": perf["mean_search_time"] * 1000,
            "qps": perf["queries_per_second"]
        }

    save_file(results, f"{example_dir}/index_type_benchmark.json")
    log.info("PRACTICAL 2 COMPLETE – HNSW wins for large-scale")
    log.info("="*90)

if __name__ == "__main__":
    practical_02_faiss_index_types()
