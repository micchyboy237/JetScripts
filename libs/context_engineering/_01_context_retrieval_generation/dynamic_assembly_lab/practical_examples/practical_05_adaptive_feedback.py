from jet.libs.context_engineering.course._01_context_retrieval_generation.labs.dynamic_assembly_lab import (
    ContextAssembler, AssemblyConstraints, ContextComponent, ComponentType,
    create_example_dir, get_example_logger
)
from jet.file.utils import save_file


class AdaptiveAssembler(ContextAssembler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feedback_scores = {}

    def record_feedback(self, component_id: str, score: float):
        self.feedback_scores[component_id] = score
        # Boost/banish based on feedback
        for c in self.components:
            if str(id(c)) == component_id:
                c.priority = max(0.1, min(1.0, c.priority + score * 0.3))


def practical_05_adaptive_feedback():
    example_dir = create_example_dir("practical_05_adaptive")
    log = get_example_logger("Practical 5: Learning from Feedback", example_dir)
    log.info("="*90)
    log.info("PRACTICAL 5: Adaptive Assembly with User Feedback")
    log.info("="*90)

    assembler = AdaptiveAssembler(AssemblyConstraints(max_tokens=1500))
    comps = [ContextComponent(ComponentType.KNOWLEDGE, f"Tip {i}", priority=0.7) for i in range(5)]
    for c in comps: assembler.add_component(c)

    # Simulate 3 rounds of feedback
    for round_num in range(1, 4):
        result = assembler.greedy_assembly()
        save_file(result, f"{example_dir}/round_{round_num}_before.json")
        # User liked component 2, disliked 4
        assembler.record_feedback(str(id(comps[1])), +1.0)
        assembler.record_feedback(str(id(comps[3])), -0.8)
        log.info(f"Round {round_num}: Feedback applied")

    final = assembler.greedy_assembly()
    save_file(final, f"{example_dir}/final_after_feedback.json")
    log.info("PRACTICAL 5 COMPLETE – System learned from user!")
    log.info("="*90)


if __name__ == "__main__":
    practical_05_adaptive_feedback()
