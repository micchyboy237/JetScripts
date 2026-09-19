"""
Demo 1: SPLADE via Sentence Transformers v5+
Model: naver/splade-cocondenser-ensembledistil
Strategy: Learned sparse embedding with built-in decode/interpretability
"""

from sentence_transformers import SparseEncoder


def main():
    # 1. Load pretrained SPLADE model
    model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

    sentences = [
        "Apple releases new iPhone with titanium frame",
        "The weather is lovely and sunny today",
        "Chandrayaan-3 landed on the Moon in August 2023",
    ]

    # 2. Generate sparse embeddings (returns sparse tensor [N, 30522])
    embeddings = model.encode(sentences, max_active_dims=64)

    # 3. Inspect sparsity statistics
    stats = SparseEncoder.sparsity(embeddings)
    print(f"Sparsity: {stats['sparsity_ratio']:.2%}")
    print(f"Avg non-zero dims: {stats['active_dims']:.1f}\n")

    # 4. Decode to human-readable token-weight pairs
    top_k = 10
    token_weights = model.decode(embeddings, top_k=top_k)

    for i, sentence in enumerate(sentences):
        pairs = ", ".join(
            f'("{tok.strip()}", {val:.2f})' for tok, val in token_weights[i]
        )
        print(f"[{i}] {sentence}")
        print(f"    Top tokens: {pairs}\n")


if __name__ == "__main__":
    main()
