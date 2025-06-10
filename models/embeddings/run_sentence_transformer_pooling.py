from jet.logger import logger
from jet.models.embeddings.sentence_transformer_pooling import encode_sentences, load_sentence_transformer


if __name__ == "__main__":
    model_name = "all-MiniLM-L6-v2"
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A robot learns to dance in the moonlight."
    ]
    pooling_modes = ["cls_token", "mean_tokens",
                     "max_tokens", "mean_sqrt_len_tokens"]

    for mode in pooling_modes:
        try:
            logger.info(f"\nTesting pooling mode: {mode}")
            model = load_sentence_transformer(
                model_name, pooling_mode=mode, use_mps=True)
            embeddings = encode_sentences(model, sentences)
            print(f"Pooling mode: {mode}")
            print(f"Embeddings shape: {embeddings.shape}")
            print(f"First embedding (first 5 dims): {embeddings[0][:5]}...")
        except Exception as e:
            print(f"Error with {mode}: {str(e)}")
