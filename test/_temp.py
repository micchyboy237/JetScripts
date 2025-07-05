from typing import Union, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from jet.models.model_types import ModelValue

# Load the model and tokenizer
# Use Hugging Face equivalent for compatibility
model_path: ModelValue = "mlx-community/Llama-3.2-1B-Instruct-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Set padding token if not defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def tokenize_and_encode(
    text: Union[str, List[str]], max_length: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize text and return padded token IDs and attention mask.

    Args:
        text: Single string or list of strings to tokenize.
        max_length: Maximum sequence length for padding/truncation.

    Returns:
        Tuple of (padded token IDs, attention mask).
    """
    if isinstance(text, str):
        text = [text]

    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    return encoded["input_ids"], encoded["attention_mask"]


def get_embeddings(
    token_ids: torch.Tensor, attention_mask: torch.Tensor
) -> np.ndarray:
    """
    Generate sequence embeddings using mean pooling, excluding padding tokens.

    Args:
        token_ids: Tensor of shape [batch_size, seq_length] with token IDs.
        attention_mask: Tensor of shape [batch_size, seq_length] indicating non-padding tokens.

    Returns:
        Numpy array of shape [batch_size, hidden_size] with sequence embeddings.
    """
    with torch.no_grad():
        outputs = model(token_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
        # Last layer: [batch_size, seq_length, hidden_size]
        hidden_states = outputs.hidden_states[-1]

    # Mask padding tokens and compute mean pooling
    attention_mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
    masked_hidden = hidden_states * attention_mask
    sum_hidden = masked_hidden.sum(dim=1)  # Sum along sequence length
    valid_lengths = attention_mask.sum(dim=1)  # Number of non-padding tokens
    embeddings = sum_hidden / valid_lengths.clamp(min=1)  # Mean pooling

    # Convert to numpy and L2-normalize
    embeddings = embeddings.cpu().numpy()
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


def compute_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity between embeddings.

    Args:
        embeddings: Numpy array of shape [batch_size, hidden_size].

    Returns:
        Numpy array of shape [batch_size, batch_size] with cosine similarities.
    """
    return cosine_similarity(embeddings)


# Example usage
if __name__ == "__main__":
    texts = ["I like Llama models!", "AI is amazing", "Hello world"]
    token_ids, attention_mask = tokenize_and_encode(texts)
    embeddings = get_embeddings(token_ids, attention_mask)
    similarities = compute_cosine_similarity(embeddings)

    print("Cosine Similarities:\n", similarities)
