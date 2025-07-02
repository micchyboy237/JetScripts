from sentence_transformers.util import cos_sim, dot_score, pairwise_dot_score
from sentence_transformers import SentenceTransformer
from typing import List, TypedDict, Literal
from typing import List, Dict, Any, Literal, Optional, TypedDict, Union
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, SimilarityFunction, models
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig
from jet.data.header_types import TextNode
from jet.file.utils import load_file, save_file
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType, ModelType
from jet.models.config import HF_TOKEN, MODELS_CACHE_DIR
from jet.models.utils import get_embedding_size

from jet.logger import logger
from jet.models.tokenizer.base import count_tokens, get_max_token_count
from jet.models.utils import resolve_model_value


EMBED_MODEL: EmbedModelType = "all-MiniLM-L6-v2"


# Define types
SimilarityFunction = Literal["cosine", "dot", "euclidean"]


class SimilarityResult(TypedDict):
    id: str
    rank: int
    doc_index: int
    score: float
    text: str
    tokens: int
    matched: dict[str, int]


def compute_similarity_results(
    query: str,
    passages: List[str],
    similarity_fn: SimilarityFunction = "cosine"
) -> List[SimilarityResult]:
    """
    Compute similarity results between a query and passages with optimized memory usage and progress tracking.

    Args:
        model: SentenceTransformer model.
        query: Query text.
        passages: List of passage texts.
        similarity_fn: Similarity function name (e.g., "cosine").

    Returns:
        List of SimilarityResult dictionaries.
    """
    # Encode query (single item, minimal memory)
    # query_embedding = model.encode(
    #     [query], convert_to_tensor=True, show_progress_bar=False)
    query_embedding = SentenceTransformerRegistry.generate_embeddings(
        [query], return_format="numpy")[0]

    # Batch encode passages to reduce memory footprint
    batch_size = 32  # Adjustable based on available memory
    passage_embeddings = SentenceTransformerRegistry.generate_embeddings(
        passages, batch_size=batch_size, show_progress=True, return_format="numpy")
    # passage_embeddings = []
    # for i in tqdm(range(0, len(passages), batch_size), desc="Encoding passages"):
    #     batch = passages[i:i + batch_size]
    #     embeddings = model.encode(
    #         batch, convert_to_tensor=True, show_progress_bar=False)
    #     passage_embeddings.append(embeddings)
    # passage_embeddings = torch.cat(passage_embeddings, dim=0)

    # Compute similarities based on similarity_fn
    if similarity_fn == "cosine":
        similarities = cos_sim(query_embedding, passage_embeddings)[0]
    elif similarity_fn == "dot":
        similarities = dot_score(query_embedding, passage_embeddings)[0]
    elif similarity_fn == "euclidean":
        # Euclidean similarity: Convert distance to similarity (1 / (1 + distance))
        distances = torch.cdist(query_embedding, passage_embeddings, p=2)[0]
        similarities = 1 / (1 + distances)
    else:
        raise ValueError(f"Unsupported similarity function: {similarity_fn}")

    similarities_np = similarities.cpu().numpy().astype(np.float32)

    # Compute ranks (1 for highest, no skips)
    sorted_indices = np.argsort(-similarities_np)  # Descending order
    ranks = np.zeros(len(passages), dtype=np.int32)
    for rank, idx in enumerate(sorted_indices, 1):
        ranks[idx] = rank

    # Get token counts efficiently
    tokenizer = SentenceTransformerRegistry.get_tokenizer()

    token_counts: List[int] = count_tokens(
        tokenizer, passages, prevent_total=True)

    # Compute matched terms and build results
    query_terms = query.lower().split()
    results: List[SimilarityResult] = []
    for idx in tqdm(range(len(passages)), desc="Building results"):
        token_count = token_counts[idx]
        passage = passages[idx]
        matched = {
            term: passage.lower().count(term)
            for term in query_terms
            if term in passage.lower()
        }
        result: SimilarityResult = {
            "id": f"doc_{idx}",
            "rank": int(ranks[idx]),
            "doc_index": idx,
            "score": float(similarities_np[idx]),
            "text": passage,
            "tokens": token_count,
            "matched": matched,
        }
        results.append(result)

    return results


def load_pretrained_model_with_default_settings() -> List[SimilarityResult]:
    """
    Demonstrates loading a pre-trained SentenceTransformer model with default settings.
    Uses: model_name_or_path, device
    Scenario: Basic text embedding for general-purpose sentence similarity.
    """
    # Initialize model with a pre-trained model name and device
    model = SentenceTransformer(
        model_name_or_path=EMBED_MODEL,
        backend="onnx",
        device="cpu",
    )

    # Define query and sentences
    query = "I enjoy coding."
    sentences = ["I enjoy coding.",
                 "Programming is fun!", "I like to read books."]

    # Compute similarity results
    results = compute_similarity_results(model, query, sentences)
    logger.gray("\nPre-trained model similarities:")
    logger.success(results)

    return results


def create_custom_model_with_modules() -> List[SimilarityResult]:
    """
    Build a custom SentenceTransformer using standard modules.
    TokenizerModule removed — handled separately as SentenceTransformer expects.
    """
    # Define query and sentences
    query = "This is a test."
    sentences = ["This is a test.", "Another test sentence."]

    # Compute dynamic max_seq_length
    max_seq_length = get_max_token_count("bert-base-uncased", sentences)

    # Step 1: Tokenizer and transformer
    word_embedding_model = models.Transformer(
        "bert-base-uncased", max_seq_length=max_seq_length)

    # Step 2: Pooling layer
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )

    # Step 3: Combine into model
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Compute similarity results
    results = compute_similarity_results(model, query, sentences)
    logger.gray("\nCustom model similarities:")
    logger.success(results)

    return results


def use_prompts_for_contextual_encoding(query: str, passages: List[str], model: EmbedModelType = "multi-qa-MiniLM-L6-cos-v1") -> List[SimilarityResult]:
    """
    Demonstrates using prompts and default_prompt_name for contextual encoding.
    Uses: model_name_or_path, prompts, default_prompt_name, device
    Scenario: Encoding queries and passages for a search engine with role-specific prompts.
    """
    prompts = {
        "query": "query: ",
        "passage": "passage: "
    }
    # model = SentenceTransformer(
    #     model_name_or_path=model,
    #     prompts=prompts,
    #     default_prompt_name="query",
    #     # backend="onnx",
    #     # device="cpu"
    #     device="mps"
    # )
    SentenceTransformerRegistry.load_model(
        model, truncate_dim=None, prompts=prompts)
    # query = "What is Python?"
    # passages = [
    #     "Python is a programming language.",
    #     "Java is used for enterprise applications.",
    #     "Python is great for data science."
    # ]
    results = compute_similarity_results(query, passages)
    logger.gray("\nPre-trained model similarities:")
    logger.success(results)

    return results


def load_private_model_with_auth() -> List[SimilarityResult]:
    """
    Demonstrates loading a private model from Hugging Face with authentication.
    Uses: model_name_or_path, token, device, cache_folder
    Scenario: Accessing a private model for a company-specific application.
    """
    # Initialize model with authentication
    model = SentenceTransformer(
        model_name_or_path=EMBED_MODEL,
        token=HF_TOKEN,
        cache_folder=MODELS_CACHE_DIR,
        backend="onnx",
        device="cpu",
    )

    # Define query and sentences
    query = "Confidential data analysis."
    sentences = ["Confidential data analysis.", "Secure text processing."]

    # Compute similarity results
    results = compute_similarity_results(model, query, sentences)
    logger.gray("\nPrivate model similarities:")
    logger.success(results)

    return results


def use_optimized_backend_with_truncation(model: EmbedModelType = EMBED_MODEL, truncate_dim: Optional[int] = None) -> List[SimilarityResult]:
    """
    Demonstrates using an optimized backend (ONNX) with truncated embeddings.
    Uses: model_name_or_path, backend, truncate_dim, model_kwargs, device
    Scenario: Deploying an efficient model for low-latency inference.
    """
    # Define query and sentences
    query = "What is Python?"
    sentences = [
        "Python is a programming language.",
        "Java is used for enterprise applications.",
        "Python is great for data science."
    ]

    # Compute dynamic truncate_dim
    truncate_dim = truncate_dim or get_max_token_count(
        EMBED_MODEL, sentences) + 32  # Add buffer

    # Define model kwargs for ONNX
    model_kwargs = {
        "provider": "CPUExecutionProvider",
        "export": True
    }

    # Initialize model with ONNX backend
    model = SentenceTransformer(
        model_name_or_path=EMBED_MODEL,
        backend="onnx",
        truncate_dim=truncate_dim,
        device="cpu",
        model_kwargs=model_kwargs,
    )

    # Encode sentences to verify truncation
    embeddings = model.encode(sentences)
    print(
        f"Truncated embedding shape: {embeddings.shape} | Dim: {len(embeddings[0])}")

    # Compute similarity results
    results = compute_similarity_results(model, query, sentences)
    logger.gray("\nONNX backend similarities:")
    logger.success(results)

    return {
        "shape": embeddings.shape,
        "dimensions": len(embeddings[0]),
        "results": results
    }


def load_specific_revision_with_custom_config() -> List[SimilarityResult]:
    """
    Demonstrates loading a specific model revision with custom configuration.
    Uses: model_name_or_path, revision, config_kwargs, tokenizer_kwargs, trust_remote_code
    Scenario: Using a specific model version for reproducibility in research.
    """
    # Define custom config and tokenizer kwargs
    config_kwargs = {"hidden_dropout_prob": 0.2}
    tokenizer_kwargs = {"use_fast": True}

    # Initialize model with specific revision
    model = SentenceTransformer(
        model_name_or_path=EMBED_MODEL,
        revision="main",
        config_kwargs=config_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
        trust_remote_code=True,
        backend="onnx",
        device="cpu"
    )

    # Define query and sentences
    query = "Reproducible research."
    sentences = ["Reproducible research.", "Consistent model version."]

    # Compute similarity results
    results = compute_similarity_results(model, query, sentences)
    logger.gray("\nSpecific revision similarities:")
    logger.success(results)

    return results


def main() -> None:
    """Run all example functions."""
    import os
    import shutil

    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)

    print("Running SentenceTransformer Examples...")

    # results = use_optimized_backend_with_truncation()
    # save_file(
    #     results, f"{output_dir}/use_optimized_backend_with_truncation_auto_dim.json")
    # results = use_optimized_backend_with_truncation(truncate_dim=64)
    # save_file(
    #     results, f"{output_dir}/use_optimized_backend_with_truncation_64_dim.json")
    # results = use_optimized_backend_with_truncation(truncate_dim=128)
    # save_file(
    #     results, f"{output_dir}/use_optimized_backend_with_truncation_128_dim.json")
    # results = use_optimized_backend_with_truncation(truncate_dim=256)
    # save_file(
    #     results, f"{output_dir}/use_optimized_backend_with_truncation_256_dim.json")
    # max_model_size = get_embedding_size(EMBED_MODEL)
    # results = use_optimized_backend_with_truncation(
    #     truncate_dim=max_model_size)
    # save_file(
    #     results, f"{output_dir}/use_optimized_backend_with_truncation_{max_model_size}_dim.json")

    query = load_file(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/query.md")
    nodes = load_file(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/generated/run_header_docs/rag/all_nodes.json")
    nodes = [TextNode(**node) for node in nodes]

    documents = [node.get_text() for node in nodes]
    results = use_prompts_for_contextual_encoding(query, documents)
    save_file(
        results, f"{output_dir}/use_prompts_for_contextual_encoding.json")


if __name__ == "__main__":
    main()
