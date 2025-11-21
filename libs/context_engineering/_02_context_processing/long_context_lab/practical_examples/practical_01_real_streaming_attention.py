# practical_01_real_streaming_attention.py
import shutil
from pathlib import Path
from typing import Dict, Any

from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from jet.adapters.llama_cpp.llm import LlamacppLLM
from jet.libs.context_engineering.course._02_context_processing.labs.long_context_lab import (
    get_logger, save_json, save_numpy,
    StreamingAttention, measure_performance
)
from jet.file.utils import save_file


def create_example_dir(example_name: str) -> Path:
    base_dir = Path(__file__).parent / "generated" / Path(__file__).stem
    example_dir = base_dir / example_name
    # shutil.rmtree(example_dir, ignore_errors=True)
    example_dir.mkdir(parents=True, exist_ok=True)
    return example_dir


def practical_01_real_streaming_attention():
    """Run StreamingAttention on REAL embeddinggemma embeddings — unlimited context!"""
    example_dir = create_example_dir("practical_01_streaming_real")
    logger = get_logger("streaming_real", example_dir)
    logger.info("=" * 90)
    logger.info("PRACTICAL 1: Streaming Attention with Real embeddinggemma (768-dim)")
    logger.info("=" * 90)

    # Real embedder + LLM
    embedder = LlamacppEmbedding(model="embeddinggemma", logger=logger)
    llm = LlamacppLLM(verbose=True)

    # Generate long real context
    prompt = """Write a detailed 800-word essay about context engineering in large language models.
Include sections on retrieval, refinement, long context, memory, and quality gates.
Use clear, technical language."""
    save_file(prompt, str(example_dir / "llm" / "prompt.md"))

    logger.info("Generating long real context with LLM (streaming)...")
    long_text = ""
    for chunk in llm.generate(prompt, temperature=0.7, max_tokens=1200, stream=True):
        long_text += chunk
    save_file(long_text, str(example_dir / "llm" / "response.md"))

    # Split into paragraphs
    chunks = [s.strip() for s in long_text.split("\n\n") if s.strip()]
    save_file(chunks, str(example_dir / "chunks.json"))

    if len(chunks) < 5:
        chunks = ["Context engineering combines retrieval, refinement, and memory to improve LLM performance."] * 20

    logger.info(f"Generated {len(chunks)} paragraphs → embedding with embeddinggemma...")
    embeddings = embedder.encode(chunks, return_format="numpy", show_progress=True)
    logger.info(f"Final sequence: {embeddings.shape[0]:,} tokens × {embeddings.shape[1]}-dim")

    # Initialize StreamingAttention
    attention = StreamingAttention(d_model=768, cache_size=2048, sink_size=128)
    logger.info("Processing with StreamingAttention (O(n) memory)...")

    # CORRECT: Run attention twice
    # 1. For timing stats
    _, perf_stats = measure_performance(attention.forward, embeddings)

    # 2. For real attention output + metadata
    attention_output, attention_info = attention.forward(embeddings)

    # Extract real stats
    cache_size = attention_info["cache_size"]
    position = attention_info["position"]
    kv_memory_mb = attention_info["memory_usage"] / (1024 * 1024)

    logger.info("SUCCESS! StreamingAttention processed entire sequence")
    logger.info(f"   • Tokens processed     : {embeddings.shape[0]:,}")
    logger.info(f"   • Final cache size     : {cache_size:,} tokens")
    logger.info(f"   • Final position       : {position:,}")
    logger.info(f"   • KV cache memory      : {kv_memory_mb:.2f} MB")
    logger.info(f"   • Throughput           : {perf_stats.throughput:,.1f} tokens/sec")
    logger.info(f"   • Processing time      : {perf_stats.time_ms:.1f} ms")

    # Save everything correctly
    results: Dict[str, Any] = {
        "sequence_length": embeddings.shape[0],
        "embedding_dim": embeddings.shape[1],
        "cache_size": cache_size,
        "final_position": position,
        "kv_memory_mb": round(kv_memory_mb, 2),
        "throughput_tokens_per_sec": round(perf_stats.throughput, 1),
        "processing_time_ms": round(perf_stats.time_ms, 1),
        "num_input_chunks": len(chunks),
        "sample_input": chunks[:3],
    }

    save_json(results, example_dir, "results")
    save_numpy(embeddings, example_dir, "input_embeddings")
    save_numpy(attention_output, example_dir, "output_embeddings")  # Now safe!

    logger.info("PRACTICAL 8 COMPLETE — All files saved successfully!")
    logger.info("\n" + "NEXT STEPS:".center(90))
    logger.info("  1. Run: practical_09_hierarchical_memory_real.py")
    logger.info("  2. Run: practical_10_full_long_context_pipeline.py")
    logger.info("  3. Final: Build ProductionContextEngine")
    logger.info("=" * 90)


if __name__ == "__main__":
    practical_01_real_streaming_attention()