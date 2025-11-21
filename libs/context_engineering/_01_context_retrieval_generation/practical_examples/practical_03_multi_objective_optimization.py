from jet.libs.context_engineering.course._01_context_retrieval_generation.labs.dynamic_assembly_lab import (
    ContextAssembler, AssemblyConstraints, ContextComponent, ComponentType,
    create_example_dir, get_example_logger
)
from jet.file.utils import save_file


class MultiObjectiveAssembler(ContextAssembler):
    def pareto_assembly(self):
        eligible = [c for c in self.components if c.relevance_score >= self.constraints.min_relevance]
        solutions = []
        for comp in eligible:
            solo = {"components": [comp], "tokens": comp.token_count}
            solutions.append(solo)
        # Simplified Pareto: maximize relevance + diversity
        ranked = sorted(
            solutions,
            key=lambda x: sum(c.relevance_score for c in x["components"]) / (1 + x["tokens"]/1000),
            reverse=True
        )
        return ranked[:5]


def practical_03_multi_objective_optimization():
    example_dir = create_example_dir("practical_03_multi_objective")
    log = get_example_logger("Practical 3: Multi-Objective Optimization", example_dir)
    log.info("="*90)
    log.info("PRACTICAL 3: Relevance + Diversity Trade-off")
    log.info("="*90)

    assembler = MultiObjectiveAssembler(AssemblyConstraints(max_tokens=3000))
    # Add diverse high/low relevance components
    for i in range(20):
        relevance = 0.9 if i < 10 else 0.4
        content = f"Highly specialized knowledge {i}" if i < 10 else f"General background info {i}"
        assembler.add_component(ContextComponent(ComponentType.KNOWLEDGE, content, relevance_score=relevance))

    pareto_front = assembler.pareto_assembly()
    save_file(pareto_front, f"{example_dir}/pareto_front.json")
    log.info(f"Top solution: {len(pareto_front[0]['components'])} components")
    log.info("PRACTICAL 3 COMPLETE")
    log.info("="*90)


if __name__ == "__main__":
    practical_03_multi_objective_optimization()
