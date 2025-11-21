from jet.libs.context_engineering.course._01_context_retrieval_generation.labs.dynamic_assembly_lab import (
    ContextOrchestrator, create_example_dir, get_example_logger
)
from jet.file.utils import save_file


def practical_08_cross_modal_assembly():
    example_dir = create_example_dir("practical_08_cross_modal")
    log = get_example_logger("Practical 8: Cross-Modal Context Assembly", example_dir)
    log.info("="*90)
    log.info("PRACTICAL 8: Full Multimodal Dynamic Assembly")
    log.info("="*90)

    orchestrator = ContextOrchestrator()
    result = orchestrator.assemble_with_pattern(
        "multi_modal",
        query="Describe this scene and generate alt text",
        text_content="A golden retriever playing in a sunny park",
        image_descriptions=["Bright green grass", "Red ball in motion", "Happy dog expression"],
        audio_transcripts=["Dog barking excitedly", "Children laughing in background"]
    )

    save_file(result, f"{example_dir}/multimodal_context.json")
    log.info(f"Cross-modal context: {len(result['components'])} components from 3 modalities")
    log.info("PRACTICAL 8 COMPLETE – True multimodal orchestration achieved")
    log.info("="*90)


if __name__ == "__main__":
    practical_08_cross_modal_assembly()
