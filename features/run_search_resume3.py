import os
import shutil
import numpy as np

from typing import List, Set, Tuple, TypedDict
from jet.code.markdown_utils._markdown_parser import derive_by_header_hierarchy, parse_markdown
from jet.data.header_docs import HeaderDocs
from jet.file.utils import load_file, save_file
from jet.llm.mlx.generation import chat
from jet.logger import logger
from jet.models.embeddings.base import generate_embeddings
from jet.models.embeddings.chunking import ChunkResult, chunk_headers_by_hierarchy
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType, LLMModelType, ModelType
from jet.models.tokenizer.base import get_tokenizer_fn, count_tokens
from jet.wordnet.text_chunker import chunk_headers

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# --- Configuration ---
RESUME_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-resume/complete_jet_resume.md"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
MAX_CONTEXT_TOKENS = 900
EMBEDDING_MODEL: EmbedModelType = "mxbai-embed-large"
MODEL_NAME: LLMModelType = "qwen3-1.7b-4bit"

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "generated",
    os.path.splitext(os.path.basename(__file__))[0]
)

# --- Tokenizer ---
tokenizer = get_tokenizer_fn(MODEL_NAME)


class SearchResult(TypedDict):
    rank: int
    score: float
    doc_id: str
    doc_index: int
    text: str


class ResponseData(TypedDict):
    response: str
    system_prompt: str
    user_prompt: str


def chunk_text(text: str, chunk_size: int, overlap: int = 30) -> List[ChunkResult]:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")
    if overlap < 0:
        raise ValueError("Overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk size")

    tokens = tokenize_text(text)
    if not tokens:
        logger.warning("Input text is empty or could not be tokenized")
        return []

    chunks: List[ChunkResult] = []
    start = 0
    index = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        if chunk_text.strip():
            chunks.append({
                "text": chunk_text,
                "index": index,
                "token_count": len(chunk_tokens)
            })
        start += chunk_size - overlap
        index += 1

    if not chunks:
        logger.warning("No valid chunks created from input text")
    return chunks


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def rank_chunks_by_similarity(query: str, chunks: List[ChunkResult]) -> List[SearchResult]:
    if not query.strip():
        raise ValueError("Query cannot be empty")
    if not chunks:
        logger.warning("No chunks provided for ranking")
        return []

    SentenceTransformerRegistry.load_model(EMBEDDING_MODEL)
    query_embedding = generate_embeddings(
        query, EMBEDDING_MODEL, return_format="numpy")
    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]
    chunk_embeddings = generate_embeddings(
        chunk_texts, EMBEDDING_MODEL, return_format="numpy", show_progress=True)

    scored_chunks = []
    for chunk, emb in zip(chunks, chunk_embeddings):
        score = cosine_similarity(query_embedding, emb)

        text = chunk["text"]
        scored_chunks.append((text, score, chunk["doc_index"]))

    # Sort by score in descending order and assign ranks
    sorted_chunks = sorted(scored_chunks, key=lambda x: x[1], reverse=True)
    search_results = [
        {
            "rank": i + 1,
            "score": score,
            **chunks[index],
            "text": text,
        }
        for i, (text, score, index) in enumerate(sorted_chunks)
    ]

    return search_results


def select_relevant_chunks(query: str, ranked_chunks: List[SearchResult], chunks: List[ChunkResult], max_tokens: int, tokenizer_model: ModelType) -> Tuple[List[str], int]:
    if max_tokens <= 0:
        raise ValueError("Max tokens must be positive")

    if not ranked_chunks:
        logger.warning("No ranked chunks available for selection")
        return [], 0

    # Map chunks by text for quick lookup
    chunk_map = {
        chunk["doc_id"]: chunk
        for chunk in chunks
    }
    ranked_chunk_metadata = [
        (chunk_map[result["doc_id"]], result["score"])
        for result in ranked_chunks
        if result["doc_id"] in chunk_map
    ]

    selected = []
    total_tokens = 0
    used_indices: Set[int] = set()

    def merge_adjacent_chunks(start_chunk: ChunkResult, start_score: float) -> Tuple[str, int, Set[int]]:
        merged_text = start_chunk["text"]
        merged_tokens = count_tokens(tokenizer_model, merged_text)
        merged_indices = {start_chunk["doc_index"]}
        current_index = start_chunk["doc_index"]

        for offset in [-1, 1]:
            adj_index = current_index + offset
            while adj_index not in used_indices:
                for chunk, score in ranked_chunk_metadata:
                    text = chunk["text"]
                    token_count = count_tokens(tokenizer_model, text)
                    if chunk["doc_index"] == adj_index and total_tokens + merged_tokens + token_count <= max_tokens:
                        if score >= start_score * 0.8:
                            merged_text = merged_text + " " + \
                                text if offset == 1 else text + \
                                " " + merged_text
                            merged_tokens += token_count
                            merged_indices.add(adj_index)
                        break
                else:
                    break
                adj_index += offset

        return merged_text, merged_tokens, merged_indices

    for chunk, score in ranked_chunk_metadata:
        if chunk["doc_index"] in used_indices:
            continue

        merged_text, chunk_tokens, merged_indices = merge_adjacent_chunks(
            chunk, score)
        if total_tokens + chunk_tokens <= max_tokens:
            selected.append(merged_text)
            total_tokens += chunk_tokens
            used_indices.update(merged_indices)
        if total_tokens >= max_tokens:
            break

    logger.info(
        f"Selected {len(selected)} merged chunks with {total_tokens} total tokens")
    return selected, total_tokens


def get_response(query: str, context_chunks: List[str]) -> ResponseData:
    context = "\n---\n".join(context_chunks)
    system_prompt = (
        "You are an expert resume assistant. "
        "Use the given resume to answer the user's question concisely, clearly, and professionally."
    )

    user_prompt = f"{query}\n\nResume:\n{context}"

    response = chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        verbose=True,
    )

    return {
        "response": response["content"],
        "system_prompt": system_prompt,
        "user_prompt": user_prompt
    }


def format_sub_dir(text: str) -> str:
    return text.lower().strip('.,!?').replace(' ', '_').replace('.', '_').replace(',', '_').replace('!', '_').replace('?', '_').strip()


def main(query):
    output_dir = f"{OUTPUT_DIR}/{format_sub_dir(query)}"
    shutil.rmtree(output_dir, ignore_errors=True)

    chunk_size = 300

    resume_md = load_file(RESUME_PATH)

    markdown_tokens = parse_markdown(resume_md)
    save_file({"query": query, "markdown_tokens": markdown_tokens},
              f"{output_dir}/markdown_tokens.json")

    header_docs = HeaderDocs.from_tokens(markdown_tokens)
    header_docs.calculate_num_tokens(EMBEDDING_MODEL)
    save_file({"query": query, "header_docs": header_docs},
              f"{output_dir}/header_docs.json")

    header_nodes = [node for node in header_docs.as_nodes()
                    if node.type == "header"]
    save_file(header_nodes, f"{output_dir}/header_nodes.json")

    header_recursive_nodes = [{
        "id": node.id,
        "doc_id": node.doc_id,
        "doc_index": node.doc_index,
        "chunk_index": node.chunk_index,
        "line": node.line,
        "level": node.level,
        "num_tokens": count_tokens(EMBEDDING_MODEL, node.get_recursive_text()),
        "parent_id": node.parent_id,
        "parent_headers": "\n".join(node.get_parent_headers()).strip(),
        "parent_header": node.parent_header,
        "header": node.header,
        "content": node.content,
        "metadata": node.metadata,
        "text": node.get_recursive_text()
    } for node in header_nodes]
    save_file(header_recursive_nodes,
              f"{output_dir}/header_recursive_nodes.json")

    chunk_headers = chunk_headers_by_hierarchy(resume_md, chunk_size)
    save_file({"query": query, "chunk_headers": chunk_headers},
              f"{output_dir}/chunk_headers.json")

    derived_chunks = derive_by_header_hierarchy(resume_md)
    save_file({"query": query, "derived_chunks": derived_chunks},
              f"{output_dir}/derived_chunks.json")

    chunks = header_recursive_nodes

    search_results = rank_chunks_by_similarity(query, chunks)
    save_file({"query": query, "search_results": search_results},
              f"{output_dir}/search_results.json")

    top_chunks, total_tokens = select_relevant_chunks(
        query, search_results, chunks, MAX_CONTEXT_TOKENS, tokenizer_model=MODEL_NAME)
    save_file({"query": query, "selected_chunks": top_chunks, "total_tokens": total_tokens},
              f"{output_dir}/selected_chunks.json")

    response_data = get_response(query, top_chunks)
    prompt_content = (
        f"# Prompts for Query\n\n"
        f"## Query\n{query}\n\n"
        f"## System Prompt\n{response_data['system_prompt']}\n\n"
        f"## User Prompt\n{response_data['user_prompt']}"
    )
    save_file(prompt_content, f"{output_dir}/prompts.md")

    query_response = f"## Query\n\n{query}\n\n## Response\n\n{response_data['response']}"
    save_file(query_response, f"{output_dir}/query_response.md")


if __name__ == "__main__":
    query = "Tell me about yourself."
    main(query)
    query = "Tell me about your recent achievements."
    main(query)
