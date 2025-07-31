from typing import List, Union, Literal
from sentence_transformers import CrossEncoder
from jet.models.model_registry.transformers.cross_encoder_model_registry import CrossEncoderRegistry
from collections import defaultdict
import re
from typing import DefaultDict, Dict, TypedDict, List, Optional, Set
import os
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from jet.code.markdown_types.markdown_parsed_types import MarkdownToken
from jet.code.markdown_utils._markdown_analyzer import analyze_markdown
from jet.code.markdown_utils._markdown_parser import derive_by_header_hierarchy
from jet.file.utils import save_file
from jet.logger import logger
from jet.models.embeddings.base import generate_embeddings
from jet.models.embeddings.chunking import chunk_headers_by_hierarchy, ChunkResult
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.models.tokenizer.base import get_tokenizer_fn, count_tokens
from jet.utils.text_constants import TEXT_CONTRACTIONS_EN


PROMPT_TEMPLATE = """\
Resume information is below.
---------------------
{context}
---------------------

Given the resume information, answer the query.

Query: {query}
"""


class SearchResult(TypedDict):
    score: float
    header_score: float
    content_score: float
    header: str
    content: str
    rank: int
    level: int
    parent_header: Optional[str]
    parent_level: Optional[int]
    metadata: dict
    num_tokens: int


class HeaderDoc(TypedDict):
    doc_id: str
    header: str
    content: str
    level: Optional[int]
    parent_header: Optional[str]
    parent_level: Optional[int]
    tokens: List[MarkdownToken]
    source: str


def load_markdown_files(data_path: str) -> List[tuple[str, str]]:
    """
    Load all markdown files from the given directory with their file names,
    or a single markdown file if data_path is a file.
    """
    markdown_texts = []
    path = Path(data_path)
    if path.is_file() and path.suffix.lower() == ".md":
        with open(path, "r", encoding="utf-8") as file:
            markdown_texts.append((file.read(), path.name))
    elif path.is_dir():
        for file_path in path.glob("*.md"):
            with open(file_path, "r", encoding="utf-8") as file:
                markdown_texts.append((file.read(), file_path.name))
    else:
        raise FileNotFoundError(f"No markdown file(s) found at {data_path}")
    return markdown_texts


def preprocess_markdown_texts(text_file_tuples: List[tuple[str, str]], ignore_links: bool = False) -> List[HeaderDoc]:
    preprocessed_md_texts: List[HeaderDoc] = []
    for text, source in text_file_tuples:
        docs = derive_by_header_hierarchy(text, ignore_links=ignore_links)
        for doc in docs:
            preprocessed_md_texts.append({
                **doc,
                "source": source
            })
    return preprocessed_md_texts


def get_chunks(docs: List[HeaderDoc], chunk_size: int = 512) -> List[ChunkResult]:
    """Chunk all markdown texts using existing chunking function, including file names."""
    all_chunks = []
    for doc_index, doc in enumerate(docs):
        chunks = chunk_headers_by_hierarchy(
            f"{doc["header"]}\n{doc["content"]}", chunk_size=chunk_size)
        for chunk in chunks:
            chunk["shared_doc_id"] = doc["doc_id"]
            chunk["doc_index"] = doc_index
            chunk["parent_header"] = doc["parent_header"]
            chunk["parent_level"] = doc["parent_level"]
            chunk["metadata"] = {
                **chunk["metadata"],
                "source": doc["source"],
            }
        all_chunks.extend(chunks)
    return all_chunks


def preprocess_text(
    text: str,
    preserve_chars: Optional[Set[str]] = None,
) -> str:
    if not text or not text.strip():
        logger.debug(f"Empty or whitespace-only input text: '{text}'")
        return ""

    text = re.sub(r'\s+', ' ', text.strip())
    for contraction, expanded in TEXT_CONTRACTIONS_EN.items():
        text = re.sub(r'\b' + contraction + r'\b',
                      expanded, text, flags=re.IGNORECASE)
    text = text.lower()
    preserve_chars = preserve_chars or {'-', '_'}
    pattern = r'[^a-z0-9\s' + ''.join(map(re.escape, preserve_chars)) + r']'
    text = re.sub(pattern, '', text)
    text = re.sub(r'\s+', ' ', text.strip())

    return text


def compute_embeddings(chunks: List[ChunkResult], embed_model: EmbedModelType) -> tuple[np.ndarray, np.ndarray, List[dict]]:
    """Compute embeddings for header and content separately with file name as metadata."""
    header_texts = []
    content_texts = []
    metadata = []
    for chunk in chunks:
        source = chunk["metadata"]["source"].lower()
        # Only include parent_header if it's not a root header (parent_level != 0)
        header_text = f"{chunk['parent_header'].lstrip('#')}\n{chunk['header'].lstrip('#')}".strip(
        ) if chunk['parent_header'] and chunk['parent_level'] != 1 else chunk['header'].lstrip('#').strip()
        header_texts.append(header_text)
        content_texts.append(chunk["content"])
        metadata.append({
            "source": chunk["metadata"]["source"],
            "context": source.replace('.md', '').replace('_', ' ').title()
        })
    header_texts = [preprocess_text(text) for text in header_texts]
    content_texts = [preprocess_text(text) for text in content_texts]
    all_embeddings = generate_embeddings(
        header_texts + content_texts, embed_model, return_format="numpy", show_progress=True)
    header_embeddings = all_embeddings[:len(header_texts)]
    content_embeddings = all_embeddings[len(header_texts):]
    return header_embeddings, content_embeddings, metadata


def cosine_similarity(query_embedding: np.ndarray, chunk_embeddings: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between query and chunk embeddings."""
    dot_product = np.dot(chunk_embeddings, query_embedding.T)
    norms = np.linalg.norm(chunk_embeddings, axis=1) * \
        np.linalg.norm(query_embedding)
    # Avoid division by zero
    norms = np.where(norms == 0, 1e-10, norms)
    return dot_product / norms


def search_resume(chunks: List[ChunkResult], query: str, embed_model: EmbedModelType = "mxbai-embed-large", top_k: int = 5) -> List[SearchResult]:
    """Perform vector search on resume markdown files."""
    # Initialize model
    SentenceTransformerRegistry.load_model(embed_model)

    # Compute embeddings with metadata
    header_embeddings, content_embeddings, metadata = compute_embeddings(
        chunks, embed_model)
    query_embedding = generate_embeddings(
        [preprocess_text(query)], embed_model, return_format="numpy")[0]

    # Calculate similarities
    header_similarities = cosine_similarity(query_embedding, header_embeddings)
    content_similarities = cosine_similarity(
        query_embedding, content_embeddings)
    # Compute average similarity
    similarities = (header_similarities + content_similarities) / 2

    # Get top k results
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    results = [
        SearchResult(
            rank=idx + 1,
            score=float(similarities[i]),
            header_score=float(header_similarities[i]),
            content_score=float(content_similarities[i]),
            num_tokens=chunks[i]["num_tokens"],
            parent_header=chunks[i]["parent_header"],
            header=chunks[i]["header"],
            content=chunks[i]["content"],
            level=chunks[i]["level"],
            parent_level=chunks[i]["parent_level"],
            metadata={
                **chunks[i]["metadata"],
                **metadata[i],
                "chunk_index": chunks[i]["chunk_index"],
                "doc_index": chunks[i]["doc_index"],
                "doc_id": chunks[i]["shared_doc_id"],
            },


        )
        for idx, i in enumerate(top_k_indices)
    ]

    return results


def normalize_scores(scores: Union[List[float], np.ndarray], method: Literal["sigmoid", "minmax"] = "minmax") -> List[float]:
    """Normalize raw cross-encoder scores using specified method.

    Args:
        scores: List or NumPy array of raw logits from cross-encoder model.
        method: Normalization method ('sigmoid' or 'minmax').

    Returns:
        List of normalized scores in [0, 1].

    Raises:
        ValueError: If method is unsupported or scores are invalid for minmax.
        ValueError: If scores contain NaN or infinite values.
    """
    if isinstance(scores, np.ndarray):
        scores = scores.tolist()

    if len(scores) == 0:
        logger.warning("Empty scores list provided, returning empty list")
        return []

    if any(not np.isfinite(score) for score in scores):
        raise ValueError("Scores contain NaN or infinite values")

    if method == "sigmoid":
        logger.debug("Applying sigmoid normalization")
        return [1 / (1 + np.exp(-score)) for score in scores]
    elif method == "minmax":
        logger.debug("Applying min-max normalization")
        min_score, max_score = min(scores), max(scores)
        if max_score == min_score:
            logger.warning(
                "Max and min scores are equal, returning 0.5 for all scores")
            return [0.5] * len(scores)
        return [(score - min_score) / (max_score - min_score) for score in scores]
    else:
        raise ValueError(f"Unsupported normalization method: {method}")


def rerank_results(results: List[SearchResult], query: str, cross_encoder_model: str = "ms-marco-MiniLM-L6-v2") -> List[SearchResult]:
    """Rerank search results using a cross-encoder model.

    Args:
        results: List of initial search results from vector search.
        query: The search query used to rank results.
        cross_encoder_model: Name of the cross-encoder model to use.

    Returns:
        List of reranked search results with updated scores and ranks.

    Raises:
        ValueError: If cross-encoder model fails to load or predict.
    """
    if not results:
        logger.warning(
            "Empty results list provided for reranking, returning empty list")
        return []

    try:
        # Load cross-encoder model
        model = CrossEncoderRegistry.load_model(cross_encoder_model)
        logger.info(f"Loaded cross-encoder model: {cross_encoder_model}")
    except Exception as e:
        logger.error(f"Failed to load cross-encoder model: {str(e)}")
        raise ValueError(f"Failed to load cross-encoder model: {str(e)}")

    # Create query-document pairs
    pairs = [(query, f"{result['header']}\n{result['content']}")
             for result in results]
    logger.info(f"Created {len(pairs)} query-document pairs for reranking")

    # Compute cross-encoder scores
    try:
        raw_scores = model.predict(pairs)
        logger.info("Computed cross-encoder scores")
    except Exception as e:
        logger.error(f"Cross-encoder prediction failed: {str(e)}")
        raise ValueError(f"Cross-encoder prediction failed: {str(e)}")

    # Normalize scores
    try:
        normalized_scores = normalize_scores(raw_scores, method="minmax")
    except ValueError as e:
        logger.error(f"Normalization failed: {str(e)}")
        raise

    # Create reranked results with updated scores and ranks
    reranked_results = []
    for i, (result, score) in enumerate(zip(results, normalized_scores)):
        reranked_result = result.copy()
        reranked_result["score"] = score
        reranked_result["rank"] = i + 1
        reranked_results.append(reranked_result)

    # Sort by normalized score in descending order
    reranked_results.sort(key=lambda x: x["score"], reverse=True)

    # Update ranks after sorting
    for i, result in enumerate(reranked_results):
        result["rank"] = i + 1

    logger.info(f"Reranked {len(reranked_results)} results")
    return reranked_results


def cross_encoder_search(chunks: List[ChunkResult], query: str, cross_encoder_model: str = "ms-marco-MiniLM-L6-v2", top_k: int = 20) -> List[SearchResult]:
    """Perform search on resume chunks using a cross-encoder model.

    Args:
        chunks: List of chunked resume documents.
        query: The search query.
        cross_encoder_model: Name of the cross-encoder model to use.
        top_k: Number of top results to return.

    Returns:
        List of search results with cross-encoder scores and ranks.

    Raises:
        ValueError: If cross-encoder model fails to load or predict.
    """
    if not chunks:
        logger.warning("Empty chunks list provided, returning empty list")
        return []

    try:
        # Load cross-encoder model
        model = CrossEncoderRegistry.load_model(cross_encoder_model)
        logger.info(f"Loaded cross-encoder model: {cross_encoder_model}")
    except Exception as e:
        logger.error(f"Failed to load cross-encoder model: {str(e)}")
        raise ValueError(f"Failed to load cross-encoder model: {str(e)}")

    # Create query-document pairs
    pairs = [(query, f"{chunk['header']}\n{chunk['content']}")
             for chunk in chunks]
    logger.info(
        f"Created {len(pairs)} query-document pairs for cross-encoder search")

    # Compute cross-encoder scores
    try:
        raw_scores = model.predict(pairs)
        logger.info("Computed cross-encoder scores")
    except Exception as e:
        logger.error(f"Cross-encoder prediction failed: {str(e)}")
        raise ValueError(f"Cross-encoder prediction failed: {str(e)}")

    # Normalize scores
    try:
        scores = normalize_scores(raw_scores, method="minmax")
    except ValueError as e:
        logger.error(f"Normalization failed: {str(e)}")
        raise

    # Create search results
    results = []
    for i, chunk in enumerate(chunks):
        results.append(SearchResult(
            rank=0,  # Temporary rank, updated after sorting
            score=float(scores[i]),
            header_score=float(scores[i]),  # Cross-encoder scores entire pair
            content_score=float(scores[i]),  # Same score for consistency
            num_tokens=chunk["num_tokens"],
            parent_header=chunk["parent_header"],
            header=chunk["header"],
            content=chunk["content"],
            level=chunk["level"],
            parent_level=chunk["parent_level"],
            metadata={
                **chunk["metadata"],
                "chunk_index": chunk["chunk_index"],
                "doc_index": chunk["doc_index"],
                "doc_id": chunk["shared_doc_id"],
                "source": chunk["metadata"]["source"],
                "context": chunk["metadata"]["source"].replace('.md', '').replace('_', ' ').title()
            },
        ))

    # Sort by score and assign ranks
    results.sort(key=lambda x: x["score"], reverse=True)
    for i, result in enumerate(results[:top_k]):
        result["rank"] = i + 1

    logger.info(f"Returning top {min(top_k, len(results))} search results")
    return results[:top_k]


def group_results_by_source_for_llm_context(
    documents: List[SearchResult],
    llm_model: 'LLMModelType',
    max_tokens: int = 2000,
    buffer: int = 100
) -> str:
    def strip_hashtags(text: str) -> str:
        if text:
            return text.lstrip('#').strip()
        return text

    filtered_documents = []
    total_tokens = 0

    for doc in documents:
        doc_tokens = doc.get("num_tokens", 0)
        if total_tokens + doc_tokens > max_tokens - buffer:
            break
        filtered_documents.append(doc)
        total_tokens += doc_tokens

    documents = filtered_documents

    tokenizer = get_tokenizer_fn(
        llm_model, add_special_tokens=False, remove_pad_tokens=True)
    separator = "\n\n"
    separator_tokens = len(tokenizer.encode(separator))
    sorted_docs = sorted(
        documents, key=lambda x: x.get("score", 0), reverse=True)
    filtered_docs = []
    total_tokens = 0
    grouped_temp: DefaultDict[str, List[Dict]] = defaultdict(list)
    seen_header_text: DefaultDict[str, set] = defaultdict(set)

    for doc in sorted_docs:
        text = doc.get("content", "")
        source = doc["metadata"]["source"]
        parent_header = doc.get("parent_header", "None")
        header = doc.get("header", None)
        level = doc.get("level", 0)
        parent_level = doc.get("parent_level", None)

        if not isinstance(text, str):
            logger.debug(
                f"Non-string content found for source: {source}, doc_index: {doc.get('doc_index', 0)}, type: {type(text)}. Converting to string.")
            text = str(text) if text else ""

        doc_tokens = doc.get("num_tokens", len(tokenizer.encode(text)))
        header_tokens = 0

        if not grouped_temp[source]:
            header_tokens += len(tokenizer.encode(
                f"<!-- Source: {source} -->\n\n"))
            header_tokens += separator_tokens if filtered_docs else 0

        parent_header_key = strip_hashtags(
            parent_header) if parent_header and parent_header != "None" else None
        header_key = strip_hashtags(header) if header else None

        if header_key and header_key not in seen_header_text[source] and level >= 0:
            header_tokens += len(tokenizer.encode(f"{header}\n\n"))
            seen_header_text[source].add(header_key)

        additional_tokens = doc_tokens + header_tokens + separator_tokens

        filtered_docs.append(doc)
        grouped_temp[source].append(doc)
        total_tokens += additional_tokens

    context_blocks = []
    total_tokens = 0
    for source, docs in grouped_temp.items():
        block = f"<!-- Source: {source} -->\n\n"
        block_tokens = len(tokenizer.encode(block))
        seen_header_text_in_block = set()
        docs = sorted(docs, key=lambda x: (
            x.get("doc_index", 0),
            x.get("chunk_index", 0)
        ))
        for doc in docs:
            header = doc.get("header", None)
            parent_header = doc.get("parent_header", "None")
            text = doc.get("content", "")

            if not isinstance(text, str):
                logger.debug(
                    f"Non-string content in block for source: {source}, doc_index: {doc.get('doc_index', 0)}, type: {type(text)}. Converting to string.")
                text = str(text) if text else ""

            doc_tokens = doc.get("num_tokens", len(tokenizer.encode(text)))
            doc_level = doc.get("level", 0) if doc.get(
                "level") is not None else 0
            parent_level = doc.get("parent_level", None)
            parent_header_key = strip_hashtags(
                parent_header) if parent_header and parent_header != "None" else None
            header_key = strip_hashtags(header) if header else None

            has_matching_child = any(
                strip_hashtags(d.get("header", "")) == parent_header_key
                for d in docs
                if d.get("header") and d.get("level", 0) >= 0
            )
            if parent_header_key and parent_level is not None and not has_matching_child and parent_header_key not in seen_header_text_in_block:
                parent_header_text = f"{parent_header}\n\n"
                block += parent_header_text
                block_tokens += len(tokenizer.encode(parent_header_text))
                seen_header_text_in_block.add(parent_header_key)

            if header_key and header_key not in seen_header_text_in_block and doc_level >= 0:
                subheader_text = f"{header}\n\n"
                block += subheader_text
                block_tokens += len(tokenizer.encode(subheader_text))
                seen_header_text_in_block.add(header_key)

            block += text + "\n\n"
            block_tokens += doc_tokens + separator_tokens

        if block_tokens > len(tokenizer.encode(f"<!-- Source: {source} -->\n\n")):
            context_blocks.append(block.strip())
            total_tokens += block_tokens
        else:
            logger.warning(
                f"Empty block for {source} after processing; skipping.")

    result = "\n\n".join(context_blocks)
    final_token_count = len(tokenizer.encode(result))
    contexts_data = {
        "query": documents[0].get("query", "") if documents else "",
        "count": len(filtered_docs),
        "total_tokens": sum(doc.get("num_tokens", 0) for doc in filtered_docs),
        "results": [
            {
                "rank": doc.get("rank"),
                "merged_doc_id": doc.get("merged_doc_id"),
                "chunk_id": doc.get("chunk_id"),
                "doc_index": doc.get("doc_index"),
                "chunk_index": doc.get("chunk_index", 0),
                "header": doc.get("header"),
                "content": doc.get("content"),
                "source": doc.get("source"),
                "score": doc.get("score"),
                "mtld": doc.get("mtld"),
                "mtld_category": doc.get("mtld_category"),
                "word_count": doc.get("word_count"),
                "link_to_text_ratio": doc.get("link_to_text_ratio"),
                "num_tokens": doc.get("num_tokens"),
                "parent_header": doc.get("parent_header"),
                "parent_level": doc.get("parent_level"),
                "level": doc.get("level"),
                "selected_doc_ids": doc.get("selected_doc_ids"),
                "num_tokens": doc.get("num_tokens")
            } for doc in filtered_docs
        ]
    }

    if final_token_count > max_tokens:
        logger.warning(
            f"Final context exceeds max_tokens: {final_token_count} > {max_tokens}")
    else:
        logger.debug(
            f"Grouped context created with {final_token_count} tokens for {len(grouped_temp)} sources")
    return contexts_data, result


if __name__ == '__main__':
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    resume_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-resume/complete_jet_resume.md"

    query = "Tell me about yourself."
    llm_model: LLMModelType = "qwen3-1.7b-4bit-dwq-053125"
    chunk_size = 150
    max_tokens = 1500

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load and process markdown files
    logger.info(f"Loading markdown files from {resume_path}")
    analysis = analyze_markdown(resume_path)
    save_file(analysis, f"{output_dir}/analysis.json")

    text_file_tuples = load_markdown_files(resume_path)
    for md_content, source in text_file_tuples:
        save_file(md_content, f"{output_dir}/data/{source}")

    original_docs = preprocess_markdown_texts(text_file_tuples)
    save_file(original_docs, f"{output_dir}/docs.json")

    chunks = get_chunks(original_docs, chunk_size=chunk_size)
    save_file(chunks, f"{output_dir}/chunks.json")

    # Perform initial vector search
    logger.info(f"Performing vector search for query: {query}")
    results = search_resume(chunks, query, top_k=20)
    save_file(results, f"{output_dir}/search_results.json")

    # # Rerank results using cross-encoder
    # logger.info("Reranking search results with cross-encoder")
    # results = rerank_results(results, query)
    # save_file(results, f"{output_dir}/reranked_results.json")

    # # Perform cross-encoder search
    # logger.info(f"Performing cross-encoder search for query: {query}")
    # results = cross_encoder_search(chunks, query, top_k=20)
    # save_file(results, f"{output_dir}/search_results.json")

    # Print top results
    for i, result in enumerate(results[:5], 1):
        print(f"Result {i}:")
        print(f"Score: {result['score']:.4f}")
        print(f"Header: {result['header']}")
        print(f"Content: {result['content'][:200]}...")
        print(f"Rank: {result['rank']}")
        print(f"Level: {result['level']}")
        print(f"Parent Header: {result['parent_header']}")
        print(f"Parent Level: {result['parent_level']}")
        print(f"Number of Tokens: {result['num_tokens']}")
        print(f"Metadata: {result['metadata']}")
        print("-" * 50)

    # Group results for LLM context
    logger.info("Grouping results for LLM context")
    contexts_data, context_md = group_results_by_source_for_llm_context(
        results, llm_model, max_tokens=max_tokens)
    save_file(contexts_data, f"{output_dir}/contexts.json")
    save_file(context_md, f"{output_dir}/context.md")
    save_file({"num_tokens": count_tokens(llm_model, context_md)},
              f"{output_dir}/context_info.json")

    # Generate LLM response
    logger.info("Generating LLM response")
    mlx = MLXModelRegistry.load_model(llm_model)
    prompt = PROMPT_TEMPLATE.format(query=query, context=context_md)
    llm_response = mlx.chat(prompt, llm_model, temperature=0.7, verbose=True)
    save_file(llm_response["content"], f"{output_dir}/response.md")
