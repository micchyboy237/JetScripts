# JetScripts/libs/context_engineering/_02_context_processing/structured_data_lab/practical_examples/practical_01_knowledge_graph_construction.py
import json
import numpy as np
from typing import List
from pydantic import BaseModel, Field

from jet.adapters.llama_cpp.llm import LlamacppLLM
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from jet.libs.context_engineering.course._02_context_processing.labs.structured_data_lab import (
    create_example_dir, get_logger, save_numpy,
    KnowledgeGraph, Entity, EntityType, Relation, RelationType, Schema
)
from jet.file.utils import save_file


class Location(BaseModel):
    city: str = Field(..., description="City name")
    country: str = Field(..., description="Country name")


class Person(BaseModel):
    name: str = Field(..., description="Full name of the person")
    role: str = Field(..., description="Role in the company (e.g. CEO, CTO)")


class Product(BaseModel):
    name: str = Field(..., description="Product name")
    description: str = Field(..., description="Short product description")


class CompanyInfo(BaseModel):
    company_name: str = Field(..., description="Name of the AI startup")
    location: Location
    people: List[Person] = Field(..., min_items=2, max_items=5)
    products: List[Product] = Field(..., min_items=1, max_items=5)


def practical_01_knowledge_graph_construction():
    example_dir = create_example_dir("practical_01_knowledge_graph")
    logger = get_logger("kg_construction", example_dir)
    logger.info("=" * 90)
    logger.info("PRACTICAL 1: Building Knowledge Graph with Real Tech Company Data")
    logger.info("=" * 90)

    (example_dir / "llm").mkdir(exist_ok=True)

    llm = LlamacppLLM(verbose=True)
    embedder = LlamacppEmbedding(model="embeddinggemma", use_cache=True, cache_backend="sqlite")

    schema = Schema(
        name="tech_kg",
        entity_types={EntityType.PERSON, EntityType.ORGANIZATION, EntityType.PRODUCT, EntityType.LOCATION},
        relation_types={RelationType.WORKS_FOR, RelationType.CREATED_BY, RelationType.LOCATED_IN}
    )
    kg = KnowledgeGraph(d_model=768, schema=schema)

    prompt = """Generate a detailed fictional AI startup with:
- A creative company name
- A real-world city and country as headquarters
- Exactly 2 key people (CEO and CTO preferred)
- 2 innovative AI products with short descriptions

Return only valid JSON matching the schema."""
    save_file(prompt, str(example_dir / "llm" / "prompt.md"))

    logger.info("Generating structured company data with LLM (chat_structured_stream)...")

    company_info: CompanyInfo | None = None
    try:
        for structured_obj in llm.chat_structured_stream(
            messages=[{"role": "user", "content": prompt}],
            response_model=CompanyInfo,
            temperature=0.7,
        ):
            company_info = structured_obj
            break  # Expect only one object
    except Exception as e:
        logger.warning(f"Structured streaming failed: {e}")

    # Fallback only if completely failed
    if company_info is None:
        logger.warning("Using fallback company data")
        company_info = CompanyInfo(
            company_name="NeuraNest",
            location=Location(city="Boston", country="USA"),
            people=[
                Person(name="Emma Lee", role="CEO"),
                Person(name="Liam Patel", role="CTO")
            ],
            products=[
                Product(name="SmartQuery", description="AI-powered search engine"),
                Product(name="InsightBot", description="Real-time analytics agent")
            ]
        )

    save_file(company_info.model_dump_json(indent=2), str(example_dir / "llm" / "response.json"))

    # Build entities
    entities: List[Entity] = []

    company_entity = Entity(
        id=f"org_{company_info.company_name.lower().replace(' ', '_')}",
        name=company_info.company_name,
        entity_type=EntityType.ORGANIZATION,
        properties={"founded": 2023}
    )
    entities.append(company_entity)

    location_entity = Entity(
        id=f"loc_{company_info.location.city.lower()}",
        name=f"{company_info.location.city}, {company_info.location.country}",
        entity_type=EntityType.LOCATION,
        properties={"country": company_info.location.country}
    )
    entities.append(location_entity)

    for person in company_info.people:
        entities.append(Entity(
            id=f"person_{person.name.lower().replace(' ', '_')}",
            name=person.name,
            entity_type=EntityType.PERSON,
            properties={"role": person.role}
        ))

    for product in company_info.products:
        entities.append(Entity(
            id=f"prod_{product.name.lower().replace(' ', '_')}",
            name=product.name,
            entity_type=EntityType.PRODUCT,
            properties={"description": product.description}
        ))

    logger.info("Generating embeddings for entities...")
    names = [e.name for e in entities]
    embeddings = embedder.encode(names, return_format="numpy", show_progress=True)
    for e, emb in zip(entities, embeddings):
        e.embedding = emb

    # Add to graph
    for e in entities:
        kg.add_entity(e)

    for person in company_info.people:
        person_id = f"person_{person.name.lower().replace(' ', '_')}"
        kg.add_relation(Relation(kg.get_entity(person_id), company_entity, RelationType.WORKS_FOR))

    kg.add_relation(Relation(company_entity, location_entity, RelationType.LOCATED_IN))

    for product in company_info.products:
        prod_id = f"prod_{product.name.lower().replace(' ', '_')}"
        kg.add_relation(Relation(kg.get_entity(prod_id), company_entity, RelationType.CREATED_BY))

    stats = kg.get_statistics()
    save_file(json.dumps(stats, indent=2), str(example_dir / "graph_stats.json"))
    save_numpy(np.stack([e.embedding for e in entities if e.embedding is not None]), example_dir, "entity_embeddings")

    logger.info(f"Graph built: {stats['num_entities']} entities, {stats['num_relations']} relations")
    logger.info("PRACTICAL 1 COMPLETE — 100% structured & reliable")
    logger.info("\nNEXT STEPS → Run practical_02_graph_rag_retrieval.py")
    logger.info("=" * 90)


if __name__ == "__main__":
    practical_01_knowledge_graph_construction()