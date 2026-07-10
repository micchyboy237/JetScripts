"""
09_demo_graph_rag.py
Strategy: GraphRAG (entity graph construction + multi-hop traversal)

Problem it solves: vector/BM25 retrieval returns individual chunks,
but some questions need facts *connected across* documents — e.g.
"does ACME Corp's revenue trend matter to its contract termination
rights?" needs both the SEC filing and the vendor contract, which
share no vocabulary a flat search would match on. GraphRAG builds a
graph linking entities to the documents that mention them, then
traverses it to pull multi-hop evidence a flat retriever would miss.

Use case: legal discovery, investigative/compliance research, due
diligence — anywhere the answer requires connecting scattered facts
rather than a single passage lookup.

Note: a real system extracts entities/relations with an LLM. Here
entity extraction is a STUB that reads structured `metadata` fields
already present on each doc (company, parties, system, sku) — no
external NLP/API call.
"""
from collections import defaultdict

from common import load_docs


def extract_entities_stub(doc):
    """STUB for LLM/NER entity extraction: pull entities straight out
    of the doc's own metadata fields instead of running NLP over the
    raw text."""
    meta = doc.get("metadata", {})
    entities = set()
    for key in ("company", "parties", "system", "sku", "policy_id"):
        val = meta.get(key)
        if isinstance(val, list):
            entities.update(val)
        elif isinstance(val, str):
            entities.add(val)
    return entities


def build_graph(docs):
    """Returns (graph, doc_entities):
    graph: entity -> set of doc ids that mention it
    doc_entities: doc id -> set of entities mentioned in that doc
    """
    graph = defaultdict(set)
    doc_entities = {}
    for doc in docs:
        entities = extract_entities_stub(doc)
        doc_entities[doc["id"]] = entities
        for e in entities:
            graph[e].add(doc["id"])
    return graph, doc_entities


def traverse(graph, doc_entities, start_entity, hops=2):
    """Breadth-first traversal: entity -> its docs -> their other
    entities -> more docs, up to `hops` levels out."""
    visited_docs = set()
    visited_entities = {start_entity}
    frontier_entities = {start_entity}

    for _ in range(hops):
        next_docs = set()
        for e in frontier_entities:
            next_docs |= graph.get(e, set())
        visited_docs |= next_docs

        next_entities = set()
        for doc_id in next_docs:
            next_entities |= doc_entities.get(doc_id, set())
        frontier_entities = next_entities - visited_entities
        visited_entities |= next_entities

        if not frontier_entities:
            break

    return visited_docs, visited_entities


def main():
    docs = load_docs()
    graph, doc_entities = build_graph(docs)
    docs_by_id = {d["id"]: d for d in docs}

    print("Entity graph (entity -> connected doc ids):")
    for entity, doc_ids in graph.items():
        print(f"  {entity!r}: {sorted(doc_ids)}")

    start_entity = "ACME Corp"
    print(f"\nTraversing 2 hops from entity: {start_entity!r}")
    found_docs, found_entities = traverse(graph, doc_entities, start_entity, hops=2)

    print(f"\nEntities reached: {sorted(found_entities)}")
    print("Documents reached via graph traversal (includes docs a single-hop flat search would miss):")
    for doc_id in sorted(found_docs):
        d = docs_by_id[doc_id]
        print(f"  {doc_id}: {d['title']}")


if __name__ == "__main__":
    main()
