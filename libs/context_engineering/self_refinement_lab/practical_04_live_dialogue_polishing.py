# JetScripts/libs/context_engineering/self_refinement_lab/practical_04_live_dialogue_polishing.py
import shutil
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

from jet.libs.context_engineering.course._02_context_processing.labs.self_refinement_lab import (
    get_logger,
    SelfRefinementPipeline,
    save_numpy,
    create_sample_context,
)
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding


def create_example_dir(example_name: str) -> Path:
    base_dir = Path(__file__).parent / "generated" / Path(__file__).stem
    example_dir = base_dir / example_name
    shutil.rmtree(example_dir, ignore_errors=True)
    example_dir.mkdir(parents=True, exist_ok=True)
    return example_dir


# ===================================================================
# PRACTICAL APPLICATION 4: Self-Improving Dialogue System (Live Polishing)
# ===================================================================
def practical_04_live_dialogue_polishing():
    """One refinement step per turn → progressively better context"""
    example_dir = create_example_dir("practical_04_dialogue_polishing")
    logger = get_logger("dialogue", example_dir)

    logger.info("PRACTICAL 4: Live Dialogue Context Polishing")

    # Light pipeline: only 1 iteration per turn
    pipeline = SelfRefinementPipeline(
        max_iterations=1,           # single refinement per turn
        convergence_threshold=0.0,  # always do one pass
        quality_threshold=0.99      # never early-stop
    )

    # Simulate 8-turn conversation
    dialogue_history = []  # list of (speaker, embedding_chunk)
    current_context = np.zeros((0, 256))

    for turn in range(1, 9):
        # Simulate new turn (user or assistant message)
        new_chunk = create_sample_context(
            seq_len=np.random.randint(16, 64),
            quality_level='poor' if turn % 3 == 0 else 'medium'
        )
        dialogue_history.append(("user" if turn % 2 else "assistant", new_chunk))
        current_context = np.concatenate([current_context, new_chunk], axis=0)

        query = create_sample_context(32, quality_level='high')  # current user query

        logger.info(f"Turn {turn:2} | Context length: {current_context.shape[0]}")

        # Single-step polish
        result = pipeline.refine_context(current_context, query, target_quality=0.99)

        current_context = result["final_context"]
        logger.info(f"   → Quality after polish: {result['final_quality'].overall:.4f}")

        # In real system: generate next response using current_context
        # Here we just save for inspection
        save_numpy(current_context, example_dir, f"dialogue_context_turn_{turn:02d}")

    logger.info("Dialogue polishing complete – context quality improves every turn!")


if __name__ == "__main__":
    practical_04_live_dialogue_polishing()