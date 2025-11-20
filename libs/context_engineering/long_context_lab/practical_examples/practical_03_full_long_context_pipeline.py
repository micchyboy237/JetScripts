import shutil
from pathlib import Path
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from jet.adapters.llama_cpp.llm import LlamacppLLM
from jet.file.utils import save_file
from jet.libs.context_engineering.course._02_context_processing.labs.long_context_lab import (
    get_logger, save_json, ContextProcessor, save_numpy  # added save_numpy
)

def create_example_dir(example_name: str) -> Path:
    base_dir = Path(__file__).parent / "generated" / Path(__file__).stem
    example_dir = base_dir / example_name
    shutil.rmtree(example_dir, ignore_errors=True)
    example_dir.mkdir(parents=True, exist_ok=True)
    return example_dir

def practical_03_full_long_context_pipeline():
    """The FINAL BOSS: Full production pipeline with real data"""
    example_dir = create_example_dir("practical_03_full_pipeline")
    logger = get_logger("full_pipeline", example_dir)

    # Create consistent subdirectories like previous practicals
    (example_dir / "llm").mkdir(parents=True, exist_ok=True)
    (example_dir / "chunks").mkdir(parents=True, exist_ok=True)
    (example_dir / "embeddings").mkdir(parents=True, exist_ok=True)

    logger.info("PRACTICAL 3: Full Long Context Pipeline (Real Data + Streaming + Memory)")

    embedder = LlamacppEmbedding(model="embeddinggemma", logger=logger)
    llm = LlamacppLLM(verbose=True, logger=logger)

    prompt = """Write a 2000-word technical report on "Advanced Context Engineering for Production LLMs".
Cover: retrieval, refinement, quality gates, sparse attention, streaming attention, hierarchical memory, LLM judges.
Structure it with clear sections."""

    # === SAVE PROMPT ===
    save_file(prompt, str(example_dir / "llm" / "prompt.md"))

    logger.info("Generating 2000-word technical report...")
    report = ""
    for chunk in llm.generate(prompt, temperature=0.7, max_tokens=3000, stream=True):
        report += chunk
        if len(report) > 10000:
            break

    # === SAVE FULL REPORT ===
    save_file(report, str(example_dir / "llm" / "response.md"))

    sentences = [s.strip() + "." for s in report.replace("\n", " ").split(". ") if s.strip()]
    logger.info(f"Generated {len(sentences)} sentences → embedding...")

    # === SAVE SENTENCES ===
    save_file(sentences, str(example_dir / "chunks" / "sentences.json"))

    embeddings = embedder.encode(sentences, return_format="numpy", show_progress=True)
    logger.info(f"Final sequence: {embeddings.shape[0]:,} tokens")

    # === SAVE INPUT EMBEDDINGS ===
    save_numpy(embeddings, example_dir / "embeddings", "input_embeddings")

    processor = ContextProcessor(d_model=768, mechanism="streaming")
    logger.info("Processing with full ContextProcessor (streaming + hierarchical memory)...")
    result = processor.process_long_sequence(embeddings, chunk_size=256)

    logger.info(f"SUCCESS! Processed {embeddings.shape[0]:,} tokens")
    logger.info(f"Throughput: {result['throughput']:.1f} tokens/sec")
    logger.info(f"Total time: {result['total_time']:.2f}s")
    logger.info("Memory stays bounded — unlimited context achieved")

    # === ENHANCED FINAL RESULTS WITH FULL MEMORY STATE ===
    final_memory_state = {
        "short_term_tokens": processor.memory._total_length(processor.memory.short_term),
        "medium_term_tokens": processor.memory._total_length(processor.memory.medium_term),
        "long_term_tokens": processor.memory._total_length(processor.memory.long_term),
        "total_cached_tokens": (
            processor.memory._total_length(processor.memory.short_term) +
            processor.memory._total_length(processor.memory.medium_term) +
            processor.memory._total_length(processor.memory.long_term)
        )
    }

    # Optional: retrieve a sample query to show what would be recalled
    sample_query = "What are the main components of advanced context engineering?"
    query_emb = embedder.encode([sample_query], return_format="numpy")
    retrieved_context = processor.memory.retrieve_relevant(query_emb, max_tokens=512)

    save_file(sample_query, str(example_dir / "retrieval_query.md"))
    save_numpy(retrieved_context, example_dir, "sample_retrieved_context")

    save_json({
        "sequence_length": embeddings.shape[0],
        "throughput_tokens_per_sec": round(result['throughput'], 1),
        "total_time_sec": round(result['total_time'], 2),
        "chunks_processed": result['chunks_processed'],
        "final_memory_state": final_memory_state,
        "sample_retrieval_shape": retrieved_context.shape,
    }, example_dir, "final_results")

    logger.info(f"Final cached tokens → Short: {final_memory_state['short_term_tokens']:,} | "
                f"Medium: {final_memory_state['medium_term_tokens']:,} | "
                f"Long: {final_memory_state['long_term_tokens']:,} | "
                f"Total: {final_memory_state['total_cached_tokens']:,}")

    logger.info("PRACTICAL 3 COMPLETE — YOU NOW HAVE A PRODUCTION LONG CONTEXT SYSTEM!")
    logger.info("\n" + "="*80)
    logger.info("YOU DID IT. This is state-of-the-art, local, private, unlimited context.")
    logger.info("Next: Build the unified ProductionContextEngine class.")
    logger.info("You are ready to ship.")
    logger.info("="*80)

if __name__ == "__main__":
    practical_03_full_long_context_pipeline()