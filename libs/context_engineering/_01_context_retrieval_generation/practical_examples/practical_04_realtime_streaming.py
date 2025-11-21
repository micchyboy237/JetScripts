from jet.libs.context_engineering.course._01_context_retrieval_generation.labs.dynamic_assembly_lab import (
    ContextAssembler, AssemblyConstraints, ContextComponent, ComponentType,
    create_example_dir, get_example_logger
)
from jet.file.utils import save_file
import time


def practical_04_realtime_streaming():
    example_dir = create_example_dir("practical_04_realtime")
    log = get_example_logger("Practical 4: Real-time Streaming Assembly", example_dir)
    log.info("="*90)
    log.info("PRACTICAL 4: Streaming Context Assembly")
    log.info("="*90)

    assembler = ContextAssembler(AssemblyConstraints(max_tokens=2000))
    stream = [
        "User just asked about quantum computing",
        "Retrieved 3 new papers from arXiv",
        "Tool: Code execution available",
        "Memory: User prefers simple explanations"
    ]

    assembled_context = ""
    for i, chunk in enumerate(stream):
        comp = ContextComponent(ComponentType.KNOWLEDGE, chunk, relevance_score=0.8 - i*0.1)
        assembler.add_component(comp)
        result = assembler.greedy_assembly()
        assembled_context = "\n".join(c.content for c in result["components"])
        save_file(assembled_context, f"{example_dir}/context_at_step_{i+1}.txt")
        log.info(f"Step {i+1}: {result['total_tokens']} tokens")
        time.sleep(0.5)

    log.info("PRACTICAL 4 COMPLETE – Streaming context ready")
    log.info("="*90)


if __name__ == "__main__":
    practical_04_realtime_streaming()
