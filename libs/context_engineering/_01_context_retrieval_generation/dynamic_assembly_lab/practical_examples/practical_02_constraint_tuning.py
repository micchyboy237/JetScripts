from jet.libs.context_engineering.course._01_context_retrieval_generation.labs.dynamic_assembly_lab import (
    ContextOrchestrator, AssemblyConstraints, AssemblyStrategy, create_example_dir, get_example_logger
)
from jet.file.utils import save_file


def practical_02_constraint_tuning():
    example_dir = create_example_dir("practical_02_constraint_tuning")
    log = get_example_logger("Practical 2: Constraint Exploration", example_dir)
    log.info("="*90)
    log.info("PRACTICAL 2: Constraint Tuning Experiments")
    log.info("="*90)

    orchestrator = ContextOrchestrator()
    scenarios = [
        ("tight", AssemblyConstraints(max_tokens=800, min_relevance=0.7)),
        ("balanced", AssemblyConstraints(max_tokens=2000, min_relevance=0.4)),
        ("permissive", AssemblyConstraints(max_tokens=4000, min_relevance=0.1)),
    ]

    results = {}
    for name, constraints in scenarios:
        result = orchestrator.assemble_with_pattern(
            "rag_pipeline",
            strategy=AssemblyStrategy.GREEDY,
            constraints=constraints,
            query="Latest advances in sparse attention mechanisms",
            knowledge_docs=[f"Document about attention {i}" for i in range(10)]
        )
        results[name] = {
            "tokens_used": result["total_tokens"],
            "components": len(result["components"]),
            "utilization": result["token_utilization"]
        }
        save_file(result, f"{example_dir}/result_{name}.json")

    save_file(results, f"{example_dir}/constraint_comparison.json")
    log.info(f"Tight: {results['tight']['components']} comp, {results['tight']['tokens_used']} tokens")
    log.info(f"Balanced: {results['balanced']['components']} comp, {results['balanced']['tokens_used']} tokens")
    log.info("PRACTICAL 2 COMPLETE")
    log.info("="*90)


if __name__ == "__main__":
    practical_02_constraint_tuning()
