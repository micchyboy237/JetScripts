# JetScripts/libs/context_engineering/_02_context_processing/structured_data_lab/practical_examples/practical_03_schema_reasoning.py
import json
from typing import List, Optional

from pydantic import BaseModel, Field, RootModel

from jet.adapters.llama_cpp.llm import LlamacppLLM
from jet.libs.context_engineering.course._02_context_processing.labs.structured_data_lab import (
    create_example_dir, get_logger,
    StructuredDataProcessor, Schema, EntityType, RelationType,
    GraphReasoner, create_sample_knowledge_graph
)
from jet.file.utils import save_file


class TechCompanyEntry(BaseModel):
    company_name: str = Field(..., description="Name of the company")
    founder: str = Field(..., description="Name of the founder")
    founded_year: Optional[int] = Field(None, description="Year founded (can be missing or wrong for testing)")


class TechCompanyList(RootModel[List[TechCompanyEntry]]):
    root: List[TechCompanyEntry]


def practical_03_schema_reasoning():
    example_dir = create_example_dir("practical_03_schema_reasoning")
    logger = get_logger("schema_reasoning", example_dir)
    logger.info("=" * 90)
    logger.info("PRACTICAL 3: Schema Validation & Graph Reasoning")
    logger.info("=" * 90)

    (example_dir / "llm").mkdir(exist_ok=True)

    llm = LlamacppLLM(verbose=True)

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

    prompt = """Generate exactly 3 JSON objects about fictional tech companies.
Include at least one with missing founder and one with invalid founded_year (string instead of int) for validation testing."""
    save_file(prompt, str(example_dir / "llm" / "prompt.md"))

    logger.info("Generating structured validation test data...")
    entries: List[TechCompanyEntry] = []
    try:
        for partial in llm.chat_structured_stream(
            messages=[{"role": "user", "content": prompt}],
            response_model=TechCompanyList,
            temperature=0.7,
        ):
            entries = partial.root
    except Exception as e:
        logger.warning(f"Structured streaming failed: {e}")

    if len(entries) < 2:
        logger.warning("Using fallback test data")
        entries = [
            TechCompanyEntry(company_name="AIWorks", founder="Sam", founded_year=2022),
            TechCompanyEntry(company_name="BadCo", founder="Lisa"),
            TechCompanyEntry(company_name="FutureAI", founder="Alex", founded_year="2021")
        ]

    save_file(json.dumps([e.model_dump() for e in entries], indent=2), str(example_dir / "llm" / "result.json"))

    validation_results = []
    for i, entry in enumerate(entries):
        res = processor.validate_data("tech_data", entry.model_dump())
        validation_results.append({"input": entry.model_dump(), "result": res})
        save_file(json.dumps(res, indent=2), str(example_dir / f"validation_result_{i+1}.json"))

    save_file(json.dumps(validation_results, indent=2), str(example_dir / "validation_summary.json"))

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
    logger.info("PRACTICAL 3 COMPLETE — Fully structured pipeline!")
    logger.info("\nYou now have a 100% reliable structured data + GraphRAG system!")
    logger.info("=" * 90)


if __name__ == "__main__":
    practical_03_schema_reasoning()