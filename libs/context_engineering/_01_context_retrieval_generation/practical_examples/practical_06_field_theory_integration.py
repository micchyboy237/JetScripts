from jet.libs.context_engineering.course._01_context_retrieval_generation.labs.dynamic_assembly_lab import (
    ContextAssembler, AssemblyConstraints, ContextComponent, ComponentType,
    create_example_dir, get_example_logger
)
from jet.file.utils import save_file


def practical_06_field_theory_integration():
    example_dir = create_example_dir("practical_06_field_theory")
    log = get_example_logger("Practical 6: Field Theory Integration", example_dir)
    log.info("="*90)
    log.info("PRACTICAL 6: Mythic + Mathematical + Metaphorical Resonance")
    log.info("="*90)

    assembler = ContextAssembler(AssemblyConstraints(max_tokens=1200))
    query = ContextComponent(
        ComponentType.QUERY,
        "How can we achieve harmony in AI cognition?",
        metadata={"resonance_target": ["mythic", "mathematical", "metaphorical"]}
    )
    components = [
        ContextComponent(ComponentType.KNOWLEDGE, "The hero's journey is a universal pattern", metadata={"field": "mythic", "strength": 0.9}),
        ContextComponent(ComponentType.KNOWLEDGE, "Dynamic programming solves overlapping subproblems", metadata={"field": "mathematical", "strength": 0.95}),
        ContextComponent(ComponentType.KNOWLEDGE, "Context assembly is like conducting an orchestra", metadata={"field": "metaphorical", "strength": 0.8}),
    ]
    assembler.add_components(components + [query])
    result = assembler.greedy_assembly()

    resonance = len({c.metadata.get("field") for c in result["components"] if "field" in c.metadata}) == 3
    save_file({"result": result, "full_resonance_achieved": resonance}, f"{example_dir}/field_resonance.json")
    log.info(f"FULL FIELD RESONANCE ACHIEVED: {resonance}")
    log.info("PRACTICAL 6 COMPLETE – Emergent cognition unlocked")
    log.info("="*90)


if __name__ == "__main__":
    practical_06_field_theory_integration()
