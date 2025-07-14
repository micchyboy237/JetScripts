from typing import List, TypedDict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import re

from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType


class SearchResult(TypedDict):
    rank: int
    score: float
    text: str
    file_path: str


def load_markdown_file(file_path: str) -> Optional[str]:
    """
    Load and read the content of a Markdown file.

    Args:
        file_path: Path to the Markdown file

    Returns:
        File content as a string, or None if file cannot be read
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return None


def chunk_text(text: str, min_chunk_size: int = 50, max_chunk_size: int = 1000) -> List[str]:
    """
    Split Markdown text into chunks based on logical sections (headers).

    Args:
        text: Input Markdown text to chunk
        min_chunk_size: Minimum characters for a chunk to avoid overly small sections
        max_chunk_size: Maximum characters for a chunk to prevent overly large embeddings

    Returns:
        List of text chunks, each containing a header and its associated content
    """
    if not text:
        return []

    # Split text by Markdown headers (e.g., #, ##, ###)
    header_pattern = r'^(#+ .+)$'
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_size = 0

    for line in lines:
        is_header = bool(re.match(header_pattern, line))

        if is_header and current_chunk:
            # Start a new chunk if we hit a header and have content
            chunk_text = '\n'.join(current_chunk).strip()
            if len(chunk_text) >= min_chunk_size:
                chunks.append(chunk_text)
            elif chunks and len(chunk_text) > 0:
                # Merge small chunk with previous if it's too small
                chunks[-1] = chunks[-1] + '\n\n' + chunk_text
            current_chunk = [line]
            current_size = len(line)
        else:
            # Add line to current chunk
            current_chunk.append(line)
            current_size += len(line) + 1  # +1 for newline

            # Split if chunk exceeds max size
            if current_size > max_chunk_size:
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current_chunk = []
                current_size = 0

    # Append final chunk if it exists
    if current_chunk:
        chunk_text = '\n'.join(current_chunk).strip()
        if len(chunk_text) >= min_chunk_size:
            chunks.append(chunk_text)
        elif chunks and len(chunk_text) > 0:
            chunks[-1] = chunks[-1] + '\n\n' + chunk_text

    return [chunk for chunk in chunks if chunk]


def search_resume(query: str, file_path: str, embed_model: EmbedModelType = "all-MiniLM-L12-v2", top_k: int = 5) -> List[SearchResult]:
    """
    Perform vector search on a resume Markdown file.

    Args:
        query: Search query from interviewer
        file_path: Path to the resume Markdown file
        embed_model: SentenceTransformer model name
        top_k: Number of top results to return

    Returns:
        List of SearchResult dictionaries containing rank, score, text, and file_path
    """
    # Load resume content
    content = load_markdown_file(file_path)
    if not content:
        return []

    # Chunk the resume for better granularity
    chunks = chunk_text(content)
    if not chunks:
        return []

    # Initialize model and FAISS index
    model = SentenceTransformerRegistry.load_model(embed_model)
    embeddings = model.encode(chunks, convert_to_numpy=True)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Encode query and search
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    # Prepare results
    results: List[SearchResult] = []
    for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), 1):
        if idx < len(chunks):
            results.append({
                "rank": rank,
                # Convert distance to similarity score
                "score": float(1 / (1 + score)),
                "text": chunks[idx],
                "file_path": file_path
            })

    return results


if __name__ == "__main__":
    resume_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/data/complete_jet_resume.md"
    # Example usage with real-world interviewer queries
    queries = [
        "Experience with React Native mobile development",
        "Proficiency in Node.js backend development",
        "Projects involving AI or machine learning",
        "Use of AWS services in development"
    ]

    for query in queries:
        print(f"\nSearching for: {query}")
        results = search_resume(query, resume_file, top_k=5)
        for result in results:
            print(f"Rank {result['rank']}: Score {result['score']:.3f}")
            print(f"Content: {result['text']}")
            print(f"File: {result['file_path']}")
