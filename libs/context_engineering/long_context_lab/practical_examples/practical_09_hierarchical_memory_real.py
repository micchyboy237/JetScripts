import shutil
from pathlib import Path
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from jet.adapters.llama_cpp.llm import LlamacppLLM
from jet.libs.context_engineering.course._02_context_processing.labs.long_context_lab import (
    get_logger, save_json, save_numpy, HierarchicalMemory
)

def create_example_dir(example_name: str) -> Path:
    base_dir = Path(__file__).parent / "generated" / Path(__file__).stem
    example_dir = base_dir / example_name
    shutil.rmtree(example_dir, ignore_errors=True)
    example_dir.mkdir(parents=True, exist_ok=True)
    return example_dir

def practical_09_hierarchical_memory_real():
    """Test HierarchicalMemory with real embeddinggemma embeddings"""
    example_dir = create_example_dir("practical_09_hierarchical_memory")
    logger = get_logger("hierarchical", example_dir)
    logger.info("PRACTICAL 9: Hierarchical Memory with Real Embeddings")

    embedder = LlamacppEmbedding(model="embeddinggemma", logger=logger)
    llm = LlamacppLLM(verbose=True, logger=logger)

    memory = HierarchicalMemory(d_model=768, short_term_size=1024, medium_term_size=2048)

    # Simulate long conversation
    topics = [
        "context engineering", "RAG systems", "self-refinement", "long context models",
        "sparse attention", "streaming attention", "quality gates", "LLM judges"
    ]

    logger.info("Simulating long conversation across 8 topics...")
    all_stats = []

    for i, topic in enumerate(topics):
        prompt = f"Write 3 detailed paragraphs about {topic} in context engineering."
        text = ""
        for chunk in llm.generate(prompt, temperature=0.7, max_tokens=600, stream=True):
            text += chunk

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        embs = embedder.encode(paragraphs, return_format="numpy")

        for emb in embs:
            stats = memory.add_context(emb[None, :])  # Add one token at a time
            all_stats.append(stats)

        logger.info(f"Topic {i+1}/8: {topic} → "
                    f"Short: {stats['short_term']}, "
                    f"Medium: {stats['medium_term']}, "
                    f"Long: {stats['long_term']}")

    # Query retrieval test
    query = "What is the best way to handle long context?"
    query_emb = embedder.encode([query], return_format="numpy")
    retrieved = memory.retrieve_relevant(query_emb, max_tokens=512)

    logger.info(f"Query: {query}")
    logger.info(f"Retrieved {retrieved.shape[0]} relevant tokens from memory")

    save_json({
        "memory_stats": all_stats[-10:],  # last 10 steps
        "final_short": memory._total_length(memory.short_term),
        "final_medium": memory._total_length(memory.medium_term),
        "final_long": memory._total_length(memory.long_term),
        "retrieved_length": retrieved.shape[0]
    }, example_dir, "results")
    save_numpy(retrieved, example_dir, "retrieved_context")

    logger.info("PRACTICAL 9 COMPLETE: Hierarchical memory works with real data!")
    logger.info("\nNEXT STEP: Run practical_10_full_long_context_pipeline.py")

if __name__ == "__main__":
    practical_09_hierarchical_memory_real()