# Example usage
from jet.models.embeddings.base import get_embedding_function


if __name__ == "__main__":
    # BERT example
    bert_tokenize = get_embedding_function("bert-base-cased")
    result = bert_tokenize("I can feel the magic, can you?")
    print(f"BERT Tokens: {result}")

    # Sentence Transformer example
    st_tokenize = get_embedding_function(
        "sentence-transformers/all-MiniLM-L6-v2")
    result = st_tokenize(["This is a test.", "Another sentence."])
    print(f"Sentence Transformer Tokens: {result}")

    # Causal model example
    gpt_tokenize = get_embedding_function("gpt2")
    result = gpt_tokenize("The quick brown fox jumps.")
    print(f"GPT-2 Tokens: {result}")
