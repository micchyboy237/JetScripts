"""
01_demo_chunking.py
Strategy: Chunking (structural, semantic, hierarchical)

Problem it solves: retrieval only works if the *unit* you retrieve
matches how users ask questions. A single giant document embeds/matches
poorly as a whole, and naive fixed-size splitting cuts sentences in half
and loses meaning. This demo builds three chunking strategies over the
same document and shows the resulting boundaries.

Use case: long structured documents (insurance booklets, contracts,
multi-step runbooks) where a user's question maps to one section, not
the whole document.
"""
import re

from common import TFIDFIndex, load_docs


def structural_chunks(doc):
    """Split on explicit structural markers ('Section N:', 'Step N:').
    Falls back to sentence splitting if no markers are found.
    Best for documents with predictable formatting."""
    text = doc["text"]
    parts = re.split(r"(?=Section\s+\d+:|Step\s+\d+:)", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in text.split(". ") if p.strip()]
    return [
        {"chunk_id": f"{doc['id']}::s{i}", "parent_id": doc["id"], "text": p}
        for i, p in enumerate(parts)
    ]


def semantic_chunks(doc, similarity_drop=0.25):
    """Group consecutive sentences into a chunk; start a new chunk when
    topical similarity to the running chunk drops sharply. Uses TF-IDF
    cosine similarity between sentences as a cheap stand-in for a real
    sentence-embedding similarity model. Best for long-form prose
    without clean structural markers."""
    sentences = [s.strip() for s in doc["text"].replace("\n", " ").split(". ") if s.strip()]
    if len(sentences) <= 1:
        return [{"chunk_id": f"{doc['id']}::sem0", "parent_id": doc["id"], "text": doc["text"]}]

    pseudo_docs = [{"id": f"sent{i}", "title": "", "text": s} for i, s in enumerate(sentences)]
    index = TFIDFIndex(pseudo_docs)

    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = TFIDFIndex._cosine(index.doc_vecs[i - 1], index.doc_vecs[i])
        if sim < similarity_drop:
            chunks.append(". ".join(current) + ".")
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    chunks.append(". ".join(current) + ".")

    return [
        {"chunk_id": f"{doc['id']}::sem{i}", "parent_id": doc["id"], "text": c}
        for i, c in enumerate(chunks)
    ]


def hierarchical_chunks(doc, child_chunks):
    """Wrap small child chunks with a pointer back to the full parent
    document text: retrieval matches on the precise child (high
    precision), but the parent text can be handed to the LLM for
    fuller context (high recall of surrounding detail)."""
    return [{**c, "parent_text": doc["text"]} for c in child_chunks]


def main():
    docs = load_docs()
    target = next((d for d in docs if d["type"] in ("contract", "log")), docs[0])
    print(f"Chunking document: {target['id']} - {target['title']}\n")

    s_chunks = structural_chunks(target)
    print(f"Structural chunks ({len(s_chunks)}):")
    for c in s_chunks:
        print(f"  {c['chunk_id']}: {c['text'][:80]}...")

    sem_chunks = semantic_chunks(target)
    print(f"\nSemantic chunks ({len(sem_chunks)}):")
    for c in sem_chunks:
        print(f"  {c['chunk_id']}: {c['text'][:80]}...")

    h_chunks = hierarchical_chunks(target, s_chunks)
    print(f"\nHierarchical chunks ({len(h_chunks)}) — each child links back to full parent text:")
    for c in h_chunks[:2]:
        print(
            f"  {c['chunk_id']} -> parent_id={c['parent_id']}, "
            f"parent_text_len={len(c['parent_text'])} chars"
        )


if __name__ == "__main__":
    main()
