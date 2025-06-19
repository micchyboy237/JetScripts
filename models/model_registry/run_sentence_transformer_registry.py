from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
import torch


def load_pytorch_model(model_id: str = "static-retrieval-mrl-en-v1"):
    """Load a SentenceTransformer model using PyTorch backend."""
    model = SentenceTransformerRegistry.load_model(
        model_id=model_id,
        device="cpu",
        precision="fp32",
        backend="pytorch"
    )
    print(f"Loaded PyTorch model: {model_id}")
    return model


def encode_single_sentence(text: str, model_id: str = "static-retrieval-mrl-en-v1"):
    """Generate embedding for a single input sentence."""
    registry = SentenceTransformerRegistry()
    embedding = registry.generate_embeddings(
        input_data=text,
        model_id=model_id,
        return_format="list",
        backend="pytorch"
    )
    print(
        f"Single sentence embedding (length={len(embedding)}):\n{embedding[:5]} ...")
    return embedding


def encode_batch(sentences, model_id: str = "static-retrieval-mrl-en-v1"):
    """Generate embeddings for a list of sentences using ONNX."""
    registry = SentenceTransformerRegistry()
    embeddings = registry.generate_embeddings(
        input_data=sentences,
        model_id=model_id,
        batch_size=2,
        show_progress=False,
        return_format="torch",
        backend="onnx"
    )
    print(f"Batch embeddings tensor shape: {embeddings.shape}")
    return embeddings


def show_tokenizer_and_config(model_id: str):
    """Load and show tokenizer and config information."""
    registry = SentenceTransformerRegistry()
    tokenizer = registry.get_tokenizer(model_id)
    config = registry.get_config(model_id)
    print(f"Tokenizer type: {type(tokenizer)}")
    print(f"Model hidden size from config: {config.hidden_size}")
    return tokenizer, config


def clear_registry():
    """Clear cached models, sessions, and configs."""
    registry = SentenceTransformerRegistry()
    registry.clear()
    print("Registry cleared.")


def main():
    # Load PyTorch model and encode a single sentence
    load_pytorch_model()
    encode_single_sentence("What is the capital of France?")

    # Encode a batch of sentences with ONNX
    sentences = [
        "What is the capital of France?",
        "How far is the moon from Earth?",
        "Explain quantum entanglement."
    ]
    encode_batch(sentences)

    # Access tokenizer and config
    show_tokenizer_and_config("static-retrieval-mrl-en-v1")

    # Clear registry
    clear_registry()


if __name__ == "__main__":
    main()
