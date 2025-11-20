# JetScripts/libs/context_engineering/self_refinement_lab/practical_02_quality_gate.py
import shutil
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path

from jet.libs.context_engineering.course._02_context_processing.labs.self_refinement_lab import (
    get_logger,
    ProductionRefinementSystem,
    SemanticCoherenceAssessor,
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
# PRACTICAL APPLICATION 2: Quality Assurance Gate in Generation Pipeline
# ===================================================================
def practical_02_quality_gate():
    """Reject or refine low-quality context before sending to LLM"""
    example_dir = create_example_dir("practical_02_quality_gate")
    logger = get_logger("qagate", example_dir)

    logger.info("PRACTICAL 2: Quality Gate + Auto-Refine/Reject")

    assessor = SemanticCoherenceAssessor(d_model=256)
    refinement_system = ProductionRefinementSystem(d_model=256)

    QUALITY_GATE_THRESHOLD = 0.72

    test_cases = [
        ("terrible", create_sample_context(512, quality_level='poor')),
        ("okay",     create_sample_context(512, quality_level='medium')),
        ("great",    create_sample_context(512, quality_level='high')),
    ]

    for name, ctx in test_cases:
        query = create_sample_context(32, quality_level='high')
        score = assessor.assess_quality(ctx, query)

        logger.info(f"[{name.upper()}] Initial quality: {score.overall:.4f}")

        if score.overall >= QUALITY_GATE_THRESHOLD:
            logger.info("PASSED quality gate → send directly to LLM")
            final_ctx = ctx
        else:
            logger.info("FAILED gate → triggering refinement")
            result = refinement_system.refine_context_production(ctx, query)
            final_ctx = result["final_context"]
            logger.info(f"After refinement: {result['final_quality'].overall:.4f} "
                        f"(Δ={result['total_improvement']:+.4f})")

        save_numpy(final_ctx, example_dir, f"final_context_{name}")
        logger.info("-" * 60)


# ---------------------------------------------------------------------- #
# Demo
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    practical_02_quality_gate()