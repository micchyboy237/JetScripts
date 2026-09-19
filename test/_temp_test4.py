"""
Demo 3: OpenSearch Neural Sparse Encoding (Asymmetric)
Model: opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte
Strategy: Separate query/document encoders with semantic term expansion
Requires: pip install sentence-transformers>=5.0.0
"""

from sentence_transformers.sparse_encoder import SparseEncoder


def main():
    # 1. Load asymmetric sparse model
    model = SparseEncoder(
        "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte",
        trust_remote_code=True,
    )

    # 2. Encode DOCUMENT (heavier expansion, done once at ingest)
    doc_text = "Currently New York is rainy."
    doc_tensor = model.encode_document(doc_text)
    doc_embedding = model.decode(doc_tensor, top_k=10)

    print("=== DOCUMENT ENCODING ===")
    print(f"Input:  '{doc_text}'")
    print(f"Output: {dict(doc_embedding)}\n")
    # Note: 'weather', 'rain', 'wet', 'nyc' appear despite not being in input

    # 3. Encode QUERY (lightweight, done per-search)
    query_text = "What's the weather in NY now?"
    query_tensor = model.encode_query(query_text)
    query_embedding = model.decode(query_tensor)

    print("=== QUERY ENCODING ===")
    print(f"Input:  '{query_text}'")
    print(f"Output: {dict(query_embedding)}\n")

    # 4. Verify semantic match: both share 'weather', 'ny' tokens
    doc_tokens = set(doc_embedding.keys())
    query_tokens = set(query_embedding.keys())
    shared = doc_tokens & query_tokens
    print(f"Shared expanded tokens: {shared}")
    print("→ This is why learned sparse beats BM25 for synonym/paraphrase queries")


if __name__ == "__main__":
    main()
