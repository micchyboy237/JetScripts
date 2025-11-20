# practical_01_knowledge_graph_construction.py
import json
import numpy as np
from typing import List

from jet.adapters.llama_cpp.llm import LlamacppLLM
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from jet.libs.context_engineering.course._02_context_processing.labs.structured_data_lab import (
    create_example_dir, get_logger, save_numpy,
    KnowledgeGraph, Entity, EntityType, Relation, RelationType, Schema
)
from jet.file.utils import save_file  # <-- NEW


def practical_01_knowledge_graph_construction():
    example_dir = create_example_dir("practical_01_knowledge_graph")
    logger = get_logger("kg_construction", example_dir)
    logger.info("=" * 90)
    logger.info("PRACTICAL 1: Building Knowledge Graph with Real Tech Company Data")
    logger.info("=" * 90)

    # Create subdirectories
    (example_dir / "llm").mkdir(exist_ok=True)
    (example_dir / "chunks").mkdir(exist_ok=True)

    llm = LlamacppLLM(verbose=True)
    embedder = LlamacppEmbedding(model="embeddinggemma", use_cache=True, cache_backend="sqlite")

    # Schema + KG
    schema = Schema(
        name="tech_kg",
        entity_types={EntityType.PERSON, EntityType.ORGANIZATION, EntityType.PRODUCT, EntityType.LOCATION},
        relation_types={RelationType.WORKS_FOR, RelationType.CREATED_BY, RelationType.LOCATED_IN}
    )
    kg = KnowledgeGraph(d_model=768, schema=schema)

    # Prompt
    prompt = """Generate a detailed description of a fictional AI startup, including:
- Company name and location (city, country)
- 2 key people (names, roles)
- 2 products (names, descriptions)
Use JSON format."""
    save_file(prompt, str(example_dir / "llm" / "prompt.md"))

    logger.info("Generating tech company data with LLM...")
    company_data = ""
    for chunk in llm.generate(prompt, temperature=0.7, max_tokens=600, stream=True):
        company_data += chunk
    save_file(company_data, str(example_dir / "llm" / "response.md"))

    # Parse
    try:
        company_info = json.loads(company_data)
    except json.JSONDecodeError:
        company_info = {
            "company_name": "NeuraNest",
            "location": {"city": "Boston", "country": "USA"},
            "people": [{"name": "Emma Lee", "role": "CEO"}, {"name": "Liam Patel", "role": "CTO"}],
            "products": [{"name": "SmartQuery", "description": "AI search"}, {"name": "InsightBot", "description": "Analytics bot"}]
        }
        logger.warning("LLM output not valid JSON → using fallback")

    save_file(json.dumps(company_info, indent=2), str(example_dir / "chunks" / "company_info.json"))

    # Build entities
    entities: List[Entity] = []
    # Company
    company_entity = Entity(
        id=f"org_{company_info['company_name'].lower().replace(' ', '_')}",
        name=company_info["company_name"],
        entity_type=EntityType.ORGANIZATION,
        properties={"founded": 2023}
    )
    entities.append(company_entity)

    # Location
    location_entity = Entity(
        id=f"loc_{company_info['location']['city'].lower()}",
        name=company_info["location"]["city"],
        entity_type=EntityType.LOCATION,
        properties={"country": company_info["location"]["country"]}
    )
    entities.append(location_entity)

    # People & Products
    for p in company_info["people"]:
        entities.append(Entity(
            id=f"person_{p['name'].lower().replace(' ', '_')}",
            name=p["name"],
            entity_type=EntityType.PERSON,
            properties={"role": p["role"]}
        ))
    for p in company_info["products"]:
        entities.append(Entity(
            id=f"prod_{p['name'].lower().replace(' ', '_')}",
            name=p["name"],
            entity_type=EntityType.PRODUCT,
            properties={"description": p["description"]}
        ))

    # Embeddings
    logger.info("Generating embeddings for entities...")
    names = [e.name for e in entities]
    embeddings = embedder.encode(names, return_format="numpy", show_progress=True)
    for e, emb in zip(entities, embeddings):
        e.embedding = emb

    # Add to graph
    for e in entities:
        kg.add_entity(e)

    # Relations
    for p in company_info["people"]:
        person_id = f"person_{p['name'].lower().replace(' ', '_')}"
        kg.add_relation(Relation(kg.get_entity(person_id), company_entity, RelationType.WORKS_FOR))
    kg.add_relation(Relation(company_entity, location_entity, RelationType.LOCATED_IN))
    for p in company_info["products"]:
        prod_id = f"prod_{p['name'].lower().replace(' ', '_')}"
        kg.add_relation(Relation(kg.get_entity(prod_id), company_entity, RelationType.CREATED_BY))

    # Save results
    stats = kg.get_statistics()
    save_file(json.dumps(stats, indent=2), str(example_dir / "graph_stats.json"))
    save_numpy(np.stack([e.embedding for e in entities]), example_dir, "entity_embeddings")

    logger.info(f"Graph built: {stats['num_entities']} entities, {stats['num_relations']} relations")
    logger.info("PRACTICAL 1 COMPLETE")
    logger.info("\nNEXT STEPS → Run practical_02_graph_rag_retrieval.py")
    logger.info("=" * 90)


if __name__ == "__main__":
    practical_01_knowledge_graph_construction()