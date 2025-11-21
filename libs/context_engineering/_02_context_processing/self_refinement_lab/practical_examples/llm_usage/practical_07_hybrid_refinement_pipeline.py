"""
PRACTICAL 7: Best of both worlds
Fast embedding-space refinement → fallback to LLM when stuck
"""
import os
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
import shutil
from jet.adapters.llama_cpp.llm import LlamacppLLM
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from jet.libs.context_engineering.course._02_context_processing.labs.self_refinement_lab import (
    create_example_dir, get_logger, ProductionRefinementSystem, save_numpy, save_json
)
from jet.file.utils import save_file

def practical_07_hybrid_refinement(
    query_emb: np.ndarray,
    context_embs: np.ndarray,
    texts: List[str],
    llm: LlamacppLLM,
    embedder: LlamacppEmbedding,
    embedding_first_iters: int = 3,
) -> np.ndarray:
    example_dir = create_example_dir("practical_05_llm_critique_refine")
    logger = get_logger("llm_critic", example_dir)

    save_file(texts, os.path.join(example_dir, "texts.json"))

    system = ProductionRefinementSystem(d_model=768)
    result = system.refine_context_production(context_embs, query_emb)

    save_file(result, os.path.join(example_dir, "result.json"))

    if result["final_quality"].overall < 0.82 and result["iterations_completed"] >= embedding_first_iters:
        logger.info("Embedding refinement plateaued → triggering LLM rescue")
        combined_text = "\n".join(texts)
        rescued = llm.chat([{
            "role": "user",
            "content": f"Query: {query_emb} (ignore embedding)\nContext: {combined_text}\n\nMake this context excellent. Fix all problems."
        }], temperature=0.2)
        rescued_embs = embedder.encode([rescued], return_format="numpy")
        return rescued_embs[0]

    return result["final_context"]

if __name__ == "__main__":
    from jet.adapters.llama_cpp.llm import LlamacppLLM
    from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding

    llm = LlamacppLLM(model="qwen3-instruct-2507:4b")
    embedder = LlamacppEmbedding(model="embeddinggemma")

    query = "Explain the difference between self-refinement and chain-of-thought"
    chunks = [
        "CoT makes model think step by step.",
        "Self-refinement is when model critiques its own output.",
        "Both improve reasoning.",
        "Self-refinement doesn't need external verifier.",
        "Random unrelated fact about quantum physics.",
    ]

    emb = embedder.encode([query] + chunks, return_format="numpy")
    final_context = practical_07_hybrid_refinement(
        query_emb=emb[0:1],
        context_embs=emb[1:],
        texts=chunks,
        llm=llm,
        embedder=embedder,
        embedding_first_iters=3,
    )

    print("Hybrid refinement completed")
    print(f"Final context shape: {final_context.shape}")
