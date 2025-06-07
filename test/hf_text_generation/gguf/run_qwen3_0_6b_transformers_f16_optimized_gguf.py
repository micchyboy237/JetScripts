from tqdm import tqdm
import numpy as np
from llama_cpp import Llama
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_with_padding(model: Llama, texts: List[str], max_length: int = 512, batch_size: int = 8) -> np.ndarray:
    """
    Encode texts with padding and return fixed-size embeddings in batches.

    Args:
        model: Llama model instance.
        texts: List of texts to encode.
        max_length: Maximum token length for padding/truncation.
        batch_size: Number of texts to process per batch.

    Returns:
        np.ndarray: Array of embeddings with shape (len(texts), embedding_dim).
    """
    def tokenize_text(text: str) -> List[int]:
        """Tokenize a single text and pad/truncate to max_length."""
        tokens = model.tokenize(text.encode('utf-8'), add_bos=True)
        if len(tokens) > max_length:
            return tokens[:max_length]
        return tokens + [0] * (max_length - len(tokens))

    embeddings = []
    # Process texts in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
        batch_texts = texts[i:i + batch_size]

        # Parallelize tokenization
        with ThreadPoolExecutor(max_workers=model.n_threads) as executor:
            batch_tokens = list(executor.map(tokenize_text, batch_texts))

        # Dynamic padding: pad to max length in this batch
        batch_max_len = min(max_length, max(len(tokens)
                            for tokens in batch_tokens))
        batch_tokens = [tokens[:batch_max_len] + [0] * (batch_max_len - len(tokens))
                        if len(tokens) < batch_max_len else tokens[:batch_max_len]
                        for tokens in batch_tokens]

        # Generate embeddings for the batch
        batch_embeddings = []
        for text in batch_texts:
            try:
                embedding = model.embed(text)
                embedding = np.array(embedding)
                if len(embedding.shape) > 1:
                    embedding = embedding[-1]  # Last token pooling
                batch_embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error embedding text: {str(e)}")
                # Fallback zero vector
                batch_embeddings.append(np.zeros(embedding.shape))

        # Validate shapes
        shapes = [e.shape for e in batch_embeddings]
        if len(set(shapes)) > 1:
            logger.warning(
                f"Inconsistent embedding shapes in batch {i//batch_size}: {shapes}")
            # Pad/truncate to max shape for consistency
            max_dim = max(s[0] for s in shapes)
            batch_embeddings = [np.pad(e, (0, max_dim - e.shape[0]), mode='constant')
                                if e.shape[0] < max_dim else e[:max_dim]
                                for e in batch_embeddings]

        embeddings.extend(batch_embeddings)

    return np.array(embeddings)
