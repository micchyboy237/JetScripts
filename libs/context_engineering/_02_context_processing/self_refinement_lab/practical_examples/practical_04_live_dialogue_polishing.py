# JetScripts/libs/context_engineering/self_refinement_lab/practical_04_live_dialogue_polishing.py
import shutil
import numpy as np
from typing import List, Optional
from pathlib import Path

from jet.libs.context_engineering.course._02_context_processing.labs.self_refinement_lab import (
    get_logger,
    SelfRefinementPipeline,
    save_numpy,
)
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding

def create_example_dir(example_name: str) -> Path:
    base_dir = Path(__file__).parent / "generated" / Path(__file__).stem
    example_dir = base_dir / example_name
    # shutil.rmtree(example_dir, ignore_errors=True)
    example_dir.mkdir(parents=True, exist_ok=True)
    return example_dir

# ===================================================================
# PRACTICAL APPLICATION 4: Self-Improving Dialogue System (Live Polishing)
# ===================================================================
def practical_04_live_dialogue_polishing(
    dialogue_turns: List[str] = None,
    embedder: Optional[LlamacppEmbedding] = None,
    d_model: int = 768,
):
    """One refinement step per turn → progressively better context"""
    example_dir = create_example_dir("practical_04_dialogue_polishing")
    logger = get_logger("dialogue", example_dir)

    logger.info("PRACTICAL 4: Live Dialogue Context Polishing")

    if embedder is None:
        embedder = LlamacppEmbedding(model="embeddinggemma")
    embed_fn = lambda texts: embedder.encode(texts, return_format="numpy", show_progress=False)

    if dialogue_turns is None:
        dialogue_turns = [
            "Hi, what is context engineering?",
            "It's about building better prompts with retrieval and refinement.",
            "Can you give an example?",
            "Yes, self-refinement improves context quality automatically.",
            "That's cool. Does it work in real time?",
            "Absolutely, one refinement per turn keeps quality high.",
            "What about hallucinations?",
            "Quality gating prevents bad context from reaching the model.",
        ]

    pipeline = SelfRefinementPipeline(
        max_iterations=1,
        convergence_threshold=0.0,
        quality_threshold=0.99,
        d_model=d_model
    )

    current_context = np.zeros((0, d_model))
    all_turn_texts = []

    for turn in range(1, len(dialogue_turns) + 1):
        new_text = dialogue_turns[turn - 1]
        all_turn_texts.append(new_text)
        turn_emb = embed_fn([new_text])
        current_context = np.concatenate([current_context, turn_emb], axis=0)

        # Use latest user message as query
        query_text = new_text if turn % 2 == 1 else dialogue_turns[max(0, turn-2)]
        query_emb = embed_fn([query_text])

        logger.info(f"Turn {turn:2} | Context length: {current_context.shape[0]}")
        result = pipeline.refine_context(current_context, query_emb, target_quality=0.99)
        current_context = result["final_context"]
        logger.info(f" → Quality after polish: {result['final_quality'].overall:.4f}")
        save_numpy(current_context, example_dir, f"dialogue_context_turn_{turn:02d}")

    from jet.libs.context_engineering.course._02_context_processing.labs.self_refinement_lab import save_json
    save_json({"dialogue": dialogue_turns}, example_dir, "dialogue_log")
    logger.info("Dialogue polishing complete – context quality improves every turn!")
    return {"final_context": current_context, "example_dir": example_dir}

if __name__ == "__main__":
    practical_04_live_dialogue_polishing()