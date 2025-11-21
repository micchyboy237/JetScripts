from jet.libs.context_engineering.course._01_context_retrieval_generation.labs.dynamic_assembly_lab import (
    AssemblyEvaluator, create_example_dir, get_example_logger
)
from jet.file.utils import save_file


class LegalEvaluator(AssemblyEvaluator):
    def evaluate_compliance_risk(self, components):
        risky_phrases = ["shall", "must", "hereby", "notwithstanding"]
        risk_count = 0
        for c in components:
            risk_count += sum(1 for p in risky_phrases if p in c.content.lower())
        return 1.0 / (1 + risk_count)


def practical_07_domain_metrics():
    example_dir = create_example_dir("practical_07_legal_metrics")
    log = get_example_logger("Practical 7: Domain-Specific Metrics", example_dir)
    log.info("="*90)
    log.info("PRACTICAL 7: Legal Compliance Evaluation")
    log.info("="*90)

    evaluator = LegalEvaluator()
    # Simulate legal context
    result = {"components": [
        type('obj', (), {'content': "The party shall hereby comply..."})(),
        type('obj', (), {'content': "User agrees to terms..."})(),
    ]}
    compliance_score = evaluator.evaluate_compliance_risk(result["components"])
    save_file({"compliance_risk_score": compliance_score}, f"{example_dir}/legal_risk.json")
    log.info(f"Legal compliance score: {compliance_score:.3f}")
    log.info("PRACTICAL 7 COMPLETE")
    log.info("="*90)


if __name__ == "__main__":
    practical_07_domain_metrics()
