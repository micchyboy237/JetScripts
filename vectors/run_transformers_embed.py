from typing import List, Union, Literal
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Supported model mapping
EMBED_MODELS = {
    "nomic-embed-text": "nomic-ai/nomic-embed-text-v1.5",
    "mxbai-embed-large": "mixedbread-ai/mxbai-embed-large-v1",
    "granite-embedding": "ibm-granite/granite-embedding-30m-english",
    "granite-embedding:278m": "ibm-granite/granite-embedding-278m-multilingual",
    "all-minilm:22m": "sentence-transformers/all-MiniLM-L6-v2",
    "all-minilm:33m": "sentence-transformers/all-MiniLM-L12-v2",
    "snowflake-arctic-embed:33m": "Snowflake/snowflake-arctic-embed-s",
    "snowflake-arctic-embed:137m": "Snowflake/snowflake-arctic-embed-m-long",
    "snowflake-arctic-embed": "Snowflake/snowflake-arctic-embed-l",
    "paraphrase-multilingual": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "bge-large": "BAAI/bge-large-en-v1.5",
}


def generate_embeddings(
    model_key: Literal[*EMBED_MODELS.keys()],
    texts: Union[str, List[str]],
    batch_size: int = 8,
    normalize: bool = True
) -> Union[List[float], List[List[float]]]:
    """
    Generate embeddings using a selected Hugging Face model.

    Args:
        model_key: A key from the EMBED_MODELS dict.
        texts: Single string or list of strings to embed.
        batch_size: Batch size for encoding multiple texts.
        normalize: Whether to L2 normalize the embeddings.

    Returns:
        A single embedding (List[float]) or list of embeddings (List[List[float]]).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = EMBED_MODELS[model_key]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    if isinstance(texts, str):
        texts = [texts]

    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, padding=True,
                               truncation=True, return_tensors="pt").to(device)
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # mean pooling
            if normalize:
                embeddings = torch.nn.functional.normalize(
                    embeddings, p=2, dim=1)
            all_embeddings.extend(embeddings.cpu().tolist())

    return all_embeddings[0] if len(all_embeddings) == 1 else all_embeddings
