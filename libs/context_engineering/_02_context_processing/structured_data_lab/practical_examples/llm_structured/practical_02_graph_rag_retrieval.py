# JetScripts/libs/context_engineering/_02_context_processing/structured_data_lab/practical_examples/practical_02_graph_rag_retrieval.py
import json
from typing import List

from pydantic import RootModel   # ← RootModel is required in Pydantic v2

from jet.adapters.llama_cpp.llm import LlamacppLLM
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from jet.libs.context_engineering.course._02_context_processing.labs.structured_data_lab import (
    create_example_dir, get_logger, GraphRAG, Entity, EntityType, Relation, RelationType, Schema
)
from jet.file.utils import save_file


# Correct way in Pydantic v2
class Queries(RootModel[List[str]]):
    root: List[str]


def _build_tech_knowledge_graph(d_model: int = 768):
    schema = Schema(
        name="tech_company_schema",
        entity_types={EntityType.PERSON, EntityType.ORGANIZATION, EntityType.PRODUCT, EntityType.LOCATION},
        relation_types={RelationType.WORKS_FOR, RelationType.CREATED_BY, RelationType.LOCATED_IN}
    )
    kg = GraphRAG(d_model=d_model).knowledge_graph
    kg.schema = schema

    entities = [
        Entity("person_alice", "Alice Johnson", EntityType.PERSON, {"role": "CEO"}),
        Entity("person_bob", "Bob Chen", EntityType.PERSON, {"role": "CTO"}),
        Entity("org_techcorp", "TechCorp", EntityType.ORGANIZATION, {"industry": "AI", "founded": 2020}),
        Entity("prod_aiassistant", "AIAssistant", EntityType.PRODUCT, {"version": "2.0"}),
        Entity("loc_sanfrancisco", "San Francisco", EntityType.LOCATION, {"country": "USA"}),
    ]
    relations = [
        Relation(entities[0], entities[2], RelationType.WORKS_FOR),
        Relation(entities[1], entities[2], RelationType.WORKS_FOR),
        Relation(entities[3], entities[1], RelationType.CREATED_BY),
        Relation(entities[2], entities[4], RelationType.LOCATED_IN),
        Relation(entities[3], entities[2], RelationType.CREATED_BY),
    ]
    return kg, entities, relations


def practical_02_graph_rag_retrieval():
    example_dir = create_example_dir("practical_02_graph_rag")
    logger = get_logger("graph_rag", example_dir)
    logger.info("=" * 90)
    logger.info("PRACTICAL 2: Graph-Enhanced Retrieval (GraphRAG)")
    logger.info("=" * 90)

    (example_dir / "llm").mkdir(exist_ok=True)
    (example_dir / "documents").mkdir(exist_ok=True)

    llm = LlamacppLLM(verbose=True)
    embedder = LlamacppEmbedding(model="embeddinggemma", use_cache=True)

    kg, entities, relations = _build_tech_knowledge_graph(d_model=768)
    graph_rag = GraphRAG(d_model=768)
    graph_rag.knowledge_graph = kg

    logger.info("Generating embeddings for entities...")
    names = [e.name for e in entities]
    embeddings = embedder.encode(names, return_format="numpy", show_progress=True)
    for entity, emb in zip(entities, embeddings):
        entity.embedding = emb
        kg.add_entity(entity)
    for rel in relations:
        kg.add_relation(rel)

    docs = {
        "doc_1": "Alice Johnson is the CEO and founder of TechCorp, based in San Francisco. She launched AIAssistant in 2023.",
        "doc_2": "Bob Chen serves as CTO at TechCorp and leads development of the AIAssistant platform with advanced NLP features."
    }
    for doc_id, text in docs.items():
        emb = embedder.encode([text], return_format="numpy")[0]
        graph_rag.add_document(doc_id, text, [], [], emb)
        save_file(text, str(example_dir / "documents" / f"{doc_id}.txt"))

    prompt = "Generate exactly 4 diverse, natural questions about TechCorp, its team, products, and location."
    save_file(prompt, str(example_dir / "llm" / "prompt.md"))

    logger.info("Generating questions with structured streaming...")
    queries: List[str] = []
    try:
        for partial in llm.chat_structured_stream(
            messages=[{"role": "user", "content": prompt}],
            response_model=Queries,
            temperature=0.7,
        ):
            queries = partial.root  # ← .root instead of .__root__
    except Exception as e:
        logger.warning(f"Structured streaming failed: {e}")

    if len(queries) < 3:
        logger.warning("Using fallback queries")
        queries = [
            "Who founded TechCorp?",
            "What is TechCorp's main product?",
            "Where is TechCorp located?",
            "Who is the CTO of TechCorp?"
        ]

    save_file(json.dumps(queries, indent=2), str(example_dir / "llm" / "result.json"))

    all_results = []
    for i, q in enumerate(queries):
        logger.info(f"Query {i+1}: {q}")
        q_emb = embedder.encode([q], return_format="numpy")[0]
        result = graph_rag.retrieve_context(q, q_emb, top_k_entities=4, top_k_documents=2)
        result_summary = {
            "query": q,
            "entities_retrieved": [e[0].name for e in result["relevant_entities"]],
            "documents_retrieved": [doc_id for doc_id, _ in result["relevant_documents"]],
            "unified_context_shape": result["unified_context"].shape
        }
        all_results.append(result_summary)
        save_file(json.dumps(result, indent=2, default=str), str(example_dir / f"retrieval_result_{i+1}.json"))

    save_file(json.dumps(all_results, indent=2), str(example_dir / "retrieval_summary.json"))
    logger.info(f"GraphRAG retrieval completed — processed {len(queries)} queries")
    logger.info("PRACTICAL 2 COMPLETE — Fully structured & robust")
    logger.info("\nNEXT STEP → Run practical_03_schema_reasoning.py")
    logger.info("=" * 90)


if __name__ == "__main__":
    practical_02_graph_rag_retrieval()