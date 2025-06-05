# Example usage
from jet.models.embeddings.base import get_embedding_function


if __name__ == "__main__":
    # BERT example
    bert_tokenize = get_embedding_function(
        "bert-base-cased", show_progress=True)
    result = bert_tokenize("I can feel the magic, can you?")
    print(f"BERT Tokens: {result}")

    # Span marker example
    span_marker_tokenize = get_embedding_function(
        "tomaarsen/span-marker-bert-base-fewnerd-fine-super", show_progress=True)
    result = span_marker_tokenize(
        "Cleopatra VII, also known as Cleopatra the Great, was the last active ruler of the Ptolemaic Kingdom of Egypt. She was born in 69 BCE and ruled Egypt from 51 BCE until her death in 30 BCE.")
    print(f"Span Marker Tokens: {result}")

    # Sentence Transformer example
    st_tokenize = get_embedding_function(
        "sentence-transformers/all-MiniLM-L6-v2", show_progress=True)
    result = st_tokenize(["This is a test.", "Another sentence."])
    print(f"Sentence Transformer Tokens: {result}")

    # Causal model example
    gpt_tokenize = get_embedding_function("gpt2", show_progress=True)
    result = gpt_tokenize("The quick brown fox jumps.")
    print(f"GPT-2 Tokens: {result}")
