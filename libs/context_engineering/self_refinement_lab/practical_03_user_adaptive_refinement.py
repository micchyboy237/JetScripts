# JetScripts/libs/context_engineering/self_refinement_lab/practical_03_user_adaptive_refinement.py
import shutil
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
import pickle

from jet.libs.context_engineering.course._02_context_processing.labs.self_refinement_lab import (
    get_logger,
    ProductionRefinementSystem,
    MetaRefinementController,
    save_numpy,
    save_json,
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
# PRACTICAL APPLICATION 3: Adaptive Optimization per User/Task
# ===================================================================
def practical_03_user_adaptive_refinement():
    """Different users get different refinement budgets"""
    example_dir = create_example_dir("practical_03_user_adaptive")
    logger = get_logger("useradaptive", example_dir)

    logger.info("PRACTICAL 3: User/Task-Adaptive Refinement")

    # Persistent meta-controller (survives restarts)
    meta_path = example_dir / "meta_performance.pkl"
    class PersistentMeta(MetaRefinementController):
        def __init__(self): super().__init__()
        def save(self):
            import pickle
            with open(meta_path, "wb") as f: pickle.dump(self.strategy_performance, f)
        def load(self):
            if meta_path.exists():
                import pickle
                with open(meta_path, "rb") as f:
                    self.strategy_performance = pickle.load(f)

    persistent_meta = PersistentMeta()
    persistent_meta.load()

    system = ProductionRefinementSystem(d_model=256)
    system.meta_controller = persistent_meta  # monkey-patch

    users = [
        ("free_user", create_sample_context(256, d_model=256, quality_level='poor'), {"tier": "free"}),
        ("pro_user", create_sample_context(256, d_model=256, quality_level='poor'), {"tier": "pro"}),
        ("enterprise", create_sample_context(256, d_model=256, quality_level='poor'), {"tier": "enterprise"}),
    ]

    for user_id, ctx, profile in users:
        query = create_sample_context(32, d_model=256, quality_level='high')

        # Simple profile → strategy override
        if profile["tier"] == "enterprise":
            system.pipeline.max_iterations = 10
            system.pipeline.convergence_threshold = 0.003
        elif profile["tier"] == "pro":
            system.pipeline.max_iterations = 6
        else:
            system.pipeline.max_iterations = 3

        result = system.refine_context_production(ctx, query)
        logger.info(f"{user_id:12} → strategy auto-selected, "
                    f"final quality: {result['final_quality'].overall:.4f} "
                    f"in {result['iterations_completed']} iters")
    persistent_meta.save()


if __name__ == "__main__":
    practical_03_user_adaptive_refinement()