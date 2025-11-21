import shutil
import numpy as np
from typing import List, Optional
from pathlib import Path
from jet.libs.context_engineering.course._02_context_processing.labs.self_refinement_lab import (
    get_logger,
    ProductionRefinementSystem,
    save_numpy,
    save_json,
)
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding

def create_example_dir(example_name: str) -> Path:
    base_dir = Path(__file__).parent / "generated" / Path(__file__).stem
    example_dir = base_dir / example_name
    # shutil.rmtree(example_dir, ignore_errors=True)
    example_dir.mkdir(parents=True, exist_ok=True)
    return example_dir

# ===================================================================
# PRACTICAL APPLICATION 1: Automated Content Improvement for RAG
# ===================================================================
def practical_01_rag_context_improvement(
    query: str,
    chunks: List[str],
    embedder: Optional[LlamacppEmbedding] = None,
    d_model: int = 768,  # Fixed for embeddinggemma
    cache_dir: Optional[str] = None,  # kept for future use, unused now
) -> dict:
    """
    Production-ready RAG refinement using only LlamacppEmbedding (embeddinggemma, 768-dim).
    Local, fast, consistent embeddings via llama.cpp server.
    """
    example_dir = create_example_dir("practical_01_flexible_rag")
    logger = get_logger("flexible_rag", example_dir)
    logger.info("PRACTICAL 1 (Flexible): Text → Llama.cpp Embedding (embeddinggemma) → Refine → Return")

    # Use provided embedder or default to embeddinggemma
    if embedder is None:
        logger.info("No embedder provided → using LlamacppEmbedding(model='embeddinggemma', dim=768)")
        embedder = LlamacppEmbedding(model="embeddinggemma")
    elif not isinstance(embedder, LlamacppEmbedding):
        raise ValueError("embedder must be an instance of LlamacppEmbedding")

    # Use .encode() for consistency with SentenceTransformer API and to get numpy output
    embed_fn = lambda texts: embedder.encode(
        texts,
        return_format="numpy",
        show_progress=True,
        batch_size=32
    )

    actual_d_model = 768  # embeddinggemma is fixed at 768
    if d_model != actual_d_model:
        logger.warning(f"d_model overridden: {d_model} → {actual_d_model} (embeddinggemma)")
        d_model = actual_d_model

    # --- UPDATED BLOCK preserving per-chunk structure ---
    logger.info(f"Embedding {len(chunks)} chunks + 1 query with embeddinggemma (768-dim)...")
    all_texts = [query] + chunks
    embeddings = embed_fn(all_texts)
    query_emb = embeddings[0:1]           # Shape: (1, 768)
    chunk_embs = embeddings[1:]           # Shape: (n_chunks, 768) ← KEEP 2D!

    # preserve per-chunk structure always, even if one chunk
    raw_context = np.array(chunk_embs)    # ← Now (n_chunks, 768), even if n_chunks == 1
    logger.info(f"Raw context shape: {raw_context.shape}, d_model={d_model}")

    refinement_system = ProductionRefinementSystem(d_model=d_model)
    result = refinement_system.refine_context_production(
        context=raw_context,
        query=query_emb,
        user_requirements=None
    )

    refined_context = result["final_context"]
    logger.info(f"Quality {result['initial_quality'].overall:.4f} → {result['final_quality'].overall:.4f} "
                f"(Δ={result['total_improvement']:+.4f}) in {result['iterations_completed']} iters")

    save_numpy(refined_context, example_dir, "refined_context")
    save_numpy(query_emb, example_dir, "query_embedding")
    save_json({
        "query": query,
        "chunks": chunks,
        "initial_quality": result['initial_quality'].overall,
        "final_quality": result['final_quality'].overall,
        "improvement": result['total_improvement'],
        "iterations": result['iterations_completed'],
    }, example_dir, "report")

    logger.info("Flexible RAG refinement complete (embeddinggemma)!")

    return {
        "refined_context": refined_context,
        "report": result,
        "query_text": query,
        "source_chunks": chunks,
    }

if __name__ == "__main__":
    # Local + fast (embeddinggemma, Llama.cpp only)
    practical_01_rag_context_improvement(
        query="What is context engineering?",
        chunks=[
            "Context engineering is the art of building high-quality prompts.",
            "It involves retrieval, compression, and refinement.",
            "Bad context leads to hallucinations.",
            "Self-refinement improves relevance automatically."
        ]
    )
