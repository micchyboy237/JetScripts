# JetScripts/libs/context_engineering/self_refinement_lab/practical_02_quality_gate.py
import shutil
import numpy as np
from typing import List, Optional
from pathlib import Path

from jet.libs.context_engineering.course._02_context_processing.labs.self_refinement_lab import (
    get_logger,
    ProductionRefinementSystem,
    SemanticCoherenceAssessor,
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
# PRACTICAL APPLICATION 2: Quality Assurance Gate in Generation Pipeline
# ===================================================================
def practical_02_quality_gate(
    query: str = "What is the future of AI agents?",
    chunks: List[str] = None,
    embedder: Optional[LlamacppEmbedding] = None,
    d_model: int = 768,
) -> dict:
    """Reject or refine low-quality context before sending to LLM"""
    example_dir = create_example_dir("practical_02_quality_gate")
    logger = get_logger("qagate", example_dir)

    logger.info("PRACTICAL 2: Quality Gate + Auto-Refine/Reject")

    if embedder is None:
        embedder = LlamacppEmbedding(model="embeddinggemma")
    embed_fn = lambda texts: embedder.encode(texts, return_format="numpy", show_progress=True)

    if chunks is None:
        chunks = [
            "AI agents will revolutionize work. They act autonomously.",  # high
            "Context engineering improves prompts. Use refinement.",       # medium
            "Random noise bad context hallucination bad model fails.",     # poor
        ]

    all_texts = [query] + chunks
    embeddings = embed_fn(all_texts)
    query_emb = embeddings[0:1]
    chunk_embs = embeddings[1:]

    test_cases = [
        ("terrible", chunk_embs[-1:] if len(chunk_embs) >= 1 else chunk_embs),
        ("okay",     chunk_embs[1:2] if len(chunk_embs) >= 2 else chunk_embs),
        ("great",    chunk_embs[0:1] if len(chunk_embs) >= 1 else chunk_embs),
    ]

    assessor = SemanticCoherenceAssessor(d_model=d_model)
    refinement_system = ProductionRefinementSystem(d_model=d_model)
    QUALITY_GATE_THRESHOLD = 0.72

    for name, ctx in test_cases:
        score = assessor.assess_quality(ctx, query_emb)

        logger.info(f"[{name.upper()}] Initial quality: {score.overall:.4f}")

        if score.overall >= QUALITY_GATE_THRESHOLD:
            logger.info("PASSED quality gate → send directly to LLM")
            final_ctx = ctx
        else:
            logger.info("FAILED gate → triggering refinement")
            result = refinement_system.refine_context_production(ctx, query_emb)
            final_ctx = result["final_context"]
            logger.info(f"After refinement: {result['final_quality'].overall:.4f} "
                        f"(Δ={result['total_improvement']:+.4f})")

        save_numpy(final_ctx, example_dir, f"final_context_{name}")
        logger.info("-" * 60)

    save_json({"query": query, "chunks": chunks}, example_dir, "input_texts")
    return {"example_dir": example_dir, "passed_cases": [name for name, _ in test_cases]}


# ---------------------------------------------------------------------- #
# Demo
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    practical_02_quality_gate()