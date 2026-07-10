"""
05_demo_contextual_retrieval.py
Strategy: Contextual Retrieval (contextual embeddings + contextual BM25)

Problem it solves: an isolated chunk like "Revenue grew 3% over the
previous quarter" is nearly useless for retrieval — it's missing which
company and which quarter. Anthropic's approach: before indexing,
prepend each chunk with a short piece of context pulled from the
parent document, then index *that* augmented text with both the dense
and sparse retrievers.

Use case: long structured documents (SEC filings, contracts) split
into many small chunks, where a chunk in isolation loses the
identifying details a query would search for.

Note: a real system asks an LLM to write the context sentence. Here we
use a STUB that builds the context deterministically from the parent
document's title/metadata — same pipeline shape, no external API call.
"""
from common import BM25Index, TFIDFIndex, load_docs, print_results

NAIVE_CHUNK = "Revenue grew 3% over the previous quarter."


def generate_context_stub(parent_doc):
    """STUB for the LLM contextualization step: build a short context
    string from the parent doc's title + metadata instead of asking a
    model to summarize where the chunk sits in the document."""
    meta = parent_doc.get("metadata", {})
    company = meta.get("company", parent_doc["title"])
    date = meta.get("date", "an unspecified period")
    return f"This chunk is from {parent_doc['title']} ({company}, {date})."


def main():
    docs = load_docs()
    parent = next((d for d in docs if "Revenue" in d["text"]), docs[0])

    context = generate_context_stub(parent)
    contextual_chunk = f"{context} {NAIVE_CHUNK}"

    print(f"Naive chunk:      {NAIVE_CHUNK!r}")
    print(f"Contextual chunk: {contextual_chunk!r}\n")

    naive_docs = list(docs) + [{"id": "chunk_naive", "title": "chunk", "text": NAIVE_CHUNK}]
    contextual_docs = list(docs) + [{"id": "chunk_contextual", "title": "chunk", "text": contextual_chunk}]

    query = "What was ACME Corp's quarterly revenue growth?"

    print("--- Without contextual retrieval ---")
    dense = TFIDFIndex(naive_docs).search(query, top_k=3)
    sparse = BM25Index(naive_docs).search(query, top_k=3)
    print_results("Dense", dense)
    print_results("BM25", sparse)

    print("\n--- With contextual retrieval ---")
    dense_c = TFIDFIndex(contextual_docs).search(query, top_k=3)
    sparse_c = BM25Index(contextual_docs).search(query, top_k=3)
    print_results("Dense", dense_c)
    print_results("BM25", sparse_c)

    naive_found = any(d["id"] == "chunk_naive" for d, _ in dense) or any(
        d["id"] == "chunk_naive" for d, _ in sparse
    )
    contextual_found = any(d["id"] == "chunk_contextual" for d, _ in dense_c) or any(
        d["id"] == "chunk_contextual" for d, _ in sparse_c
    )
    print(f"\nNaive chunk retrieved in top-3?      {naive_found}")
    print(f"Contextual chunk retrieved in top-3? {contextual_found}")


if __name__ == "__main__":
    main()
