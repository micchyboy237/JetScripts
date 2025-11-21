# practical_03_schema_reasoning.py
import json

from jet.adapters.llama_cpp.llm import LlamacppLLM
from jet.libs.context_engineering.course._02_context_processing.labs.structured_data_lab import (
    create_example_dir, get_logger,
    StructuredDataProcessor, Schema, EntityType, RelationType,
    GraphReasoner, create_sample_knowledge_graph
)
from jet.file.utils import save_file  # <-- NEW


def practical_03_schema_reasoning():
    example_dir = create_example_dir("practical_03_schema_reasoning")
    logger = get_logger("schema_reasoning", example_dir)
    logger.info("=" * 90)
    logger.info("PRACTICAL 3: Schema Validation & Graph Reasoning")
    logger.info("=" * 90)

    (example_dir / "llm").mkdir(exist_ok=True)
    (example_dir / "data").mkdir(exist_ok=True)

    llm = LlamacppLLM(verbose=True)

    # Schema
    schema = Schema(
        name="tech_data",
        entity_types={EntityType.PERSON, EntityType.ORGANIZATION, EntityType.PRODUCT},
        relation_types={RelationType.WORKS_FOR, RelationType.CREATED_BY},
        constraints={
            "required_fields": ["company_name", "founder"],
            "field_types": {"founded_year": int}
        }
    )
    processor = StructuredDataProcessor()
    processor.register_schema(schema)

    # Generate structured data
    prompt = """Generate 3 JSON objects about tech companies. Include some missing/invalid fields for testing."""
    save_file(prompt, str(example_dir / "llm" / "data_prompt.md"))

    logger.info("Generating structured data with LLM...")
    raw_data = ""
    for chunk in llm.generate(prompt, temperature=0.7, max_tokens=800, stream=True):
        raw_data += chunk
    save_file(raw_data, str(example_dir / "llm" / "data_response.md"))

    try:
        entries = json.loads(raw_data)
    except json.JSONDecodeError:
        entries = [
            {"company_name": "AIWorks", "founder": "Sam", "founded_year": 2022},
            {"company_name": "BadCo", "founder": "Lisa"},
            {"founded_year": "2021"}
        ]

    save_file(json.dumps(entries, indent=2), str(example_dir / "data" / "raw_entries.json"))

    # Validate
    validation_results = []
    for i, entry in enumerate(entries):
        res = processor.validate_data("tech_data", entry)
        validation_results.append({"input": entry, "result": res})
        save_file(json.dumps(res, indent=2), str(example_dir / f"validation_result_{i+1}.json"))

    save_file(json.dumps(validation_results, indent=2), str(example_dir / "validation_summary.json"))

    # Reasoning
    kg = create_sample_knowledge_graph()
    reasoner = GraphReasoner(kg)
    reasoner.add_inference_rule(
        "location_transitivity",
        [("?x", RelationType.WORKS_FOR, "?y"), ("?y", RelationType.LOCATED_IN, "?z")],
        ("?x", RelationType.LOCATED_IN, "?z")
    )
    new_relations = reasoner.infer_new_relations()
    save_file(
        json.dumps([{"source": r.source.name, "target": r.target.name, "type": r.relation_type.value} for r in new_relations], indent=2),
        str(example_dir / "inferred_relations.json")
    )

    logger.info(f"Validated {len(entries)} entries | Inferred {len(new_relations)} new relations")
    logger.info("PRACTICAL 3 COMPLETE")
    logger.info("\nYou now have a full structured data pipeline!")
    logger.info("=" * 90)


if __name__ == "__main__":
    practical_03_schema_reasoning()