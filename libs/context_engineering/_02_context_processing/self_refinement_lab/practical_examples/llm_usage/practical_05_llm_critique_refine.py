"""
PRACTICAL 5: LLM-as-a-Critic → Self-Refine (Madaan et al. style)
Uses LlamacppLLM to generate critique → revise → re-embed → repeat
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

def practical_05_llm_critique_refine(
    query: str,
    chunks: List[str],
    llm: LlamacppLLM,
    embedder: LlamacppEmbedding,
    max_rounds: int = 3,
) -> Dict[str, Any]:
    example_dir = create_example_dir("practical_05_llm_critique_refine")
    logger = get_logger("llm_critic", example_dir)

    save_file(query, os.path.join(example_dir, "query.md"))
    save_file(chunks, os.path.join(example_dir, "chunks.json"))

    embeddings = embedder.encode([query] + chunks, return_format="numpy")
    query_emb, chunk_embs = embeddings[0:1], embeddings[1:]

    context_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(chunks))
    current_context = context_text
    save_file(context_text, os.path.join(example_dir, "context.md"))

    history = []

    for round in range(max_rounds):
        critique_prompt = f"""Query: {query}

Context:
{current_context}

Instructions: Analyze the above context for the query. Rate relevance (0–10), coherence, completeness, and list specific problems (e.g. off-topic, redundant, missing info). Be harsh but constructive.

Response format:
RELEVANCE: <score>/10
COHERENCE: <score>/10
COMPLETENESS: <score>/10
ISSUES:
- <issue 1>
- <issue 2>
...
SUGGESTIONS:
- <suggestion 1>
..."""

        critique = llm.chat([{"role": "user", "content": critique_prompt}], temperature=0.3)
        logger.info(f"[Round {round+1}] Critique:\n{critique}")

        revise_prompt = f"""Query: {query}

Previous Context:
{current_context}

Critique:
{critique}

Task: Rewrite the context to fix all issues. Keep it concise, relevant, and well-structured. Only output the improved context."""

        improved = llm.chat([{"role": "user", "content": revise_prompt}], temperature=0.1)
        logger.info(f"[Round {round+1}] Improved context:\n{improved}")

        # Re-embed the improved version
        new_embs = embedder.encode([query, improved], return_format="numpy")
        new_query_emb, new_chunk_embs = new_embs[0:1], new_embs[1:]

        # Assess quality using existing embedding-based assessor
        system = ProductionRefinementSystem(d_model=768)
        result = system.refine_context_production(new_chunk_embs, new_query_emb)

        history.append({
            "round": round + 1,
            "critique": critique,
            "improved_text": improved,
            "final_quality": result["final_quality"].overall,
            "improvement": result["total_improvement"],
        })

        current_context = improved
        logger.info(f"→ Quality: {result['final_quality'].overall:.4f}")

        save_json({"round": round + 1, "critique": critique, "improved": improved}, example_dir, f"round_{round+1}")
        save_numpy(result["final_context"], example_dir, f"context_round_{round+1}")

        save_file(history, os.path.join(example_dir, "history.json"), verbose=not history)

    return {"final_text": current_context, "history": history, "example_dir": example_dir}

if __name__ == "__main__":
    from jet.adapters.llama_cpp.llm import LlamacppLLM
    from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding

    llm = LlamacppLLM(
        model="qwen3-instruct-2507:4b",  # or "llama3.2:8b", "phi4", etc.
        verbose=True,
    )
    embedder = LlamacppEmbedding(model="embeddinggemma", verbose=True)

    result = practical_05_llm_critique_refine(
        query="How can self-refinement improve RAG systems?",
        chunks=[
            "RAG is great but sometimes returns garbage chunks.",
            "Context engineering is about making prompts better.",
            "Self-refinement uses the model to critique its own context.",
            "There are many ways to do retrieval: BM25, dense, hybrid.",
            "Hallucinations happen when context is bad.",
            "Embedding models like embeddinggemma are fast and local.",
        ],
        llm=llm,
        embedder=embedder,
        max_rounds=3,
    )

    print("\nFINAL REFINED CONTEXT:")
    print(result["final_text"])
    print(f"\nHistory saved to: {result['example_dir']}")
