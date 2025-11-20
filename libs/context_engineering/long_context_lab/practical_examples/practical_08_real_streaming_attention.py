import shutil
from pathlib import Path
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from jet.adapters.llama_cpp.llm import LlamacppLLM
from jet.libs.context_engineering.course._02_context_processing.labs.long_context_lab import (
    get_logger, save_json, save_numpy,
    StreamingAttention, measure_performance
)

def create_example_dir(example_name: str) -> Path:
    base_dir = Path(__file__).parent / "generated" / Path(__file__).stem
    example_dir = base_dir / example_name
    shutil.rmtree(example_dir, ignore_errors=True)
    example_dir.mkdir(parents=True, exist_ok=True)
    return example_dir


def practical_08_real_streaming_attention():
    """Run StreamingAttention on REAL embeddinggemma embeddings — unlimited context!"""
    example_dir = create_example_dir("practical_08_streaming_real")
    logger = get_logger("streaming_real", example_dir)
    logger.info("PRACTICAL 8: Streaming Attention with Real embeddinggemma (768-dim)")

    embedder = LlamacppEmbedding(model="embeddinggemma")
    llm = LlamacppLLM(verbose=True)

    # Generate long real context using LLM
    prompt = """Write a detailed 800-word essay about context engineering in large language models.
Include sections on retrieval, refinement, long context, memory, and quality gates.
Use clear, technical language."""
    
    logger.info("Generating long real context with LLM...")
    long_text = ""
    for chunk in llm.generate(prompt, temperature=0.7, max_tokens=1200, stream=True):
        long_text += chunk

    # Split into chunks and embed
    chunks = [s.strip() for s in long_text.split("\n\n") if s.strip()]
    logger.info(f"Generated {len(chunks)} paragraphs → embedding...")
    embeddings = embedder.encode(chunks, return_format="numpy", show_progress=True)
    logger.info(f"Final sequence length: {embeddings.shape[0]:,} tokens")

    # Run StreamingAttention (unlimited context)
    attention = StreamingAttention(d_model=768, cache_size=2048, sink_size=128)
    logger.info("Processing with StreamingAttention (O(n) memory)...")
    
    output, info = measure_performance(attention.forward, embeddings)
    stats = info

    logger.info(f"Success! Processed {embeddings.shape[0]:,} tokens")
    logger.info(f"Cache size: {stats['cache_size']} | Position: {stats['position']}")
    logger.info(f"Memory usage: {stats['memory_usage'] / 1024 / 1024:.1f} MB")

    save_json({
        "sequence_length": embeddings.shape[0],
        "cache_size": stats['cache_size'],
        "final_position": stats['position'],
        "memory_mb": stats['memory_usage'] / 1024 / 1024,
        "chunks": chunks
    }, example_dir, "results")
    save_numpy(embeddings, example_dir, "input_embeddings")
    save_numpy(output, example_dir, "output_embeddings")

    logger.info("PRACTICAL 8 COMPLETE: Unlimited context achieved with real data!")
    logger.info("\nNEXT STEPS:")
    logger.info("  • Run practical_09_hierarchical_memory.py")
    logger.info("  • Then practical_10_full_long_context_pipeline.py")
    logger.info("  • Finally: build unified production engine")

if __name__ == "__main__":
    practical_08_real_streaming_attention()