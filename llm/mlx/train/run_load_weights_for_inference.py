# JetScripts/llm/mlx/inference/load_qwen3_for_inference.py
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, TypedDict
import numpy as np
import mlx.core as mx
from mlx_lm.models.qwen3 import Model, ModelArgs
from transformers import AutoTokenizer
from jet.logger import logger

# Type for weights dictionary


class WeightsDict(TypedDict):
    weights: Dict[str, np.ndarray]


@dataclass
class InferenceConfig:
    model_path: str
    vocab_size: int
    hidden_size: int
    num_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    head_dim: int
    tie_word_embeddings: bool


def load_weights(npz_path: Path) -> WeightsDict:
    """
    Load weights from a .npz file into a dictionary.

    Args:
        npz_path: Path to the weights.npz file.

    Returns:
        WeightsDict: Dictionary containing weight names and their NumPy arrays.

    Raises:
        FileNotFoundError: If the .npz file does not exist.
    """
    logger.info(f"Loading weights from {npz_path}")
    if not npz_path.exists():
        logger.error(f"Weights file not found: {npz_path}")
        raise FileNotFoundError(f"Weights file not found: {npz_path}")

    weights = np.load(npz_path, allow_pickle=False)
    logger.info(f"Loaded {len(weights)} weight tensors")
    return {"weights": {k: v for k, v in weights.items()}}


def initialize_model(config: InferenceConfig, weights: WeightsDict) -> Model:
    """
    Initialize Qwen3 model with weights.

    Args:
        config: Model configuration.
        weights: Dictionary of weights from .npz file.

    Returns:
        Model: Initialized Qwen3 model.
    """
    logger.info("Initializing Qwen3 model")
    model_args = ModelArgs(
        model_type="qwen3",
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_layers,
        intermediate_size=config.intermediate_size,
        num_attention_heads=config.num_attention_heads,
        rms_norm_eps=config.rms_norm_eps,
        vocab_size=config.vocab_size,
        num_key_value_heads=config.num_key_value_heads,
        max_position_embeddings=config.max_position_embeddings,
        rope_theta=config.rope_theta,
        head_dim=config.head_dim,
        tie_word_embeddings=config.tie_word_embeddings,
        rope_scaling=None
    )

    model = Model(model_args)
    sanitized_weights = model.sanitize(weights["weights"])
    model.load_weights(list(sanitized_weights.items()))
    mx.eval(model.parameters())
    logger.info("Model initialized and weights loaded")
    return model


def generate_text(
    model: Model,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 50
) -> str:
    """
    Generate text using the Qwen3 model.

    Args:
        model: Qwen3 model instance.
        tokenizer: Hugging Face tokenizer.
        prompt: Input prompt text.
        max_length: Maximum length of generated text.

    Returns:
        str: Generated text.
    """
    logger.info(f"Generating text with prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="np",
                       padding=True, truncation=True)
    input_ids = mx.array(inputs["input_ids"])
    attention_mask = mx.array(inputs["attention_mask"])

    generated_ids = input_ids
    for _ in range(max_length - input_ids.shape[1]):
        logits = model(generated_ids, mask=attention_mask)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        generated_ids = mx.concatenate(
            [generated_ids, next_token[:, None]], axis=1)
        attention_mask = mx.concatenate(
            [attention_mask, mx.ones_like(next_token[:, None])], axis=1
        )

    generated_text = tokenizer.decode(
        generated_ids[0].tolist(), skip_special_tokens=True)
    logger.info(f"Generated text: {generated_text}")
    return generated_text


if __name__ == "__main__":
    # Configuration for Qwen3-1.7B-4bit (adjust based on your model's config.json)
    config = InferenceConfig(
        model_path="mlx-community/Qwen3-1.7B-4bit-DWQ-053125",
        vocab_size=151936,  # From Qwen3 config
        hidden_size=2048,
        num_layers=24,
        intermediate_size=5632,
        num_attention_heads=16,
        num_key_value_heads=16,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        head_dim=128,
        tie_word_embeddings=False
    )

    # Path to weights.npz
    weights_path = Path(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/mlx/train/models/qwen3-1.7b-4bit-dwq-053125/weights.npz")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    # Load weights and initialize model
    weights = load_weights(weights_path)
    model = initialize_model(config, weights)

    # Generate text
    prompt = "Hello, how can I assist you today?"
    generated_text = generate_text(model, tokenizer, prompt)
    print(f"Generated text: {generated_text}")
