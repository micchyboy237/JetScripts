from jet.libs.context_engineering.course._01_context_retrieval_generation.labs.dynamic_assembly_lab import (
    ContextOrchestrator, AssemblyConstraints, AssemblyStrategy, ComponentType, ContextComponent,
    create_example_dir, get_example_logger
)
from jet.file.utils import save_file


def _medical_diagnosis_pattern(query: str, patient_history: str, lab_results: list, guidelines: str):
    """Custom pattern: Medical Diagnosis Assistant"""
    components = []
    components.append(ContextComponent(ComponentType.QUERY, f"Diagnosis Query: {query}", priority=1.0, relevance_score=1.0))
    components.append(ContextComponent(
        ComponentType.INSTRUCTIONS,
        "You are a senior physician. Provide differential diagnosis with evidence, risk assessment, and next steps.",
        priority=0.95
    ))
    components.append(ContextComponent(ComponentType.KNOWLEDGE, f"Patient History:\n{patient_history}", priority=0.9))
    for i, result in enumerate(lab_results):
        components.append(ContextComponent(
            ComponentType.KNOWLEDGE,
            f"Lab Result {i+1}: {result}",
            priority=0.85,
            metadata={"type": "lab"}
        ))
    components.append(ContextComponent(ComponentType.KNOWLEDGE, f"Clinical Guidelines:\n{guidelines}", priority=0.8))
    return components


def practical_01_custom_domain_pattern():
    example_dir = create_example_dir("practical_01_medical_diagnosis")
    log = get_example_logger("Practical 1: Custom Medical Pattern", example_dir)
    log.info("="*90)
    log.info("PRACTICAL 1: Custom Domain Pattern – Medical Diagnosis")
    log.info("="*90)

    orchestrator = ContextOrchestrator()
    orchestrator.patterns["medical_diagnosis"] = _medical_diagnosis_pattern

    result = orchestrator.assemble_with_pattern(
        pattern_name="medical_diagnosis",
        strategy=AssemblyStrategy.GREEDY,
        constraints=AssemblyConstraints(max_tokens=3000),
        query="65-year-old male with chest pain and shortness of breath",
        patient_history="Hypertension (10y), smoker (40 pack-years), father died of MI at 58",
        lab_results=["Troponin: 0.8 ng/mL (elevated)", "ECG: ST elevation in V2-V4", "BP: 180/100"],
        guidelines="AHA/ACC Guidelines for Acute Coronary Syndrome (2023)"
    )

    save_file(result, f"{example_dir}/medical_context_assembly.json")
    log.info(f"Medical context assembled → {len(result['components'])} components")
    log.info("PRACTICAL 1 COMPLETE")
    log.info("="*90)


if __name__ == "__main__":
    practical_01_custom_domain_pattern()
