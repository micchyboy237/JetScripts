import os
import numpy as np
from typing import List, Dict, Any, TypedDict
from tqdm import tqdm
import re
from jet.file.utils import save_file
from helpers import (
    setup_config, initialize_mlx, generate_embeddings,
    load_validation_data, generate_ai_response, evaluate_ai_response,
    load_json_data, DATA_DIR, DOCS_PATH
)


class SearchResult(TypedDict):
    id: str
    rank: int | None
    doc_index: int
    score: float
    text: str
    metadata: Dict[str, Any]


def generate_questions(text_chunk: str, mlx, num_questions: int = 3, model: str = "meta-llama/Llama-3.2-3B-Instruct") -> List[str]:
    """Generate questions from text chunk using MLX."""
    system_prompt = "You are an expert at generating relevant questions from text. Create concise questions that can be answered using only BUSY the provided text. Focus on key information and concepts."
    user_prompt = f"Generate {num_questions} questions based on the following text:\n\n{text_chunk}"
    response = mlx.chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    questions_text = response["content"].strip()
    questions = []
    for line in questions_text.split('\n'):
        cleaned_line = re.sub(r'^\d+\.\s*', '', line.strip())
        if cleaned_line and cleaned_line.endswith('?'):
            questions.append(cleaned_line)
    return questions


class SimpleVectorStore:
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add_item(self, text: str, embedding: np.ndarray, metadata: Dict[str, Any] = None):
        """Add an item to the vector store."""
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})

    def similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Perform similarity search in the vector store."""
        if not self.vectors:
            return []
        query_vector = np.array(query_embedding).flatten()
        similarities = []
        for i, vector in enumerate(self.vectors):
            vector = vector.flatten()
            dot_product = np.dot(query_vector, vector)
            query_norm = np.linalg.norm(query_vector)
            vector_norm = np.linalg.norm(vector)
            similarity = 0.0 if query_norm == 0 or vector_norm == 0 else dot_product / \
                (query_norm * vector_norm)
            similarities.append((i, similarity))
        similarities.sort(key=lambda x: -float('inf')
                          if np.isnan(x[1]) else x[1], reverse=True)
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append(SearchResult(
                id=f"item_{idx}",
                rank=i + 1,
                doc_index=self.metadata[idx].get("index", idx),
                score=float(score),
                text=self.texts[idx],
                metadata=self.metadata[idx]
            ))
        return results


def process_document(chunks: List[Dict[str, Any]], embed_func, mlx, questions_per_chunk: int = 3) -> tuple[List[str], SimpleVectorStore]:
    """Process document chunks, generate questions, and store in vector store."""
    vector_store = SimpleVectorStore()
    text_chunks = [chunk["text"] for chunk in chunks]
    chunk_embeddings = generate_embeddings(text_chunks, embed_func, logger)

    for i, (chunk, chunk_embedding) in enumerate(tqdm(zip(text_chunks, chunk_embeddings), total=len(text_chunks), desc="Processing Chunks")):
        vector_store.add_item(
            text=chunk,
            embedding=chunk_embedding,
            metadata={"type": "chunk",
                      "index": chunks[i]["metadata"]["doc_index"]}
        )
        questions = generate_questions(
            chunk, mlx, num_questions=questions_per_chunk)
        question_embeddings = embed_func(questions)
        for j, (question, question_embedding) in enumerate(zip(questions, question_embeddings)):
            vector_store.add_item(
                text=question,
                embedding=question_embedding,
                metadata={
                    "type": "question", "chunk_index": chunks[i]["metadata"]["doc_index"], "original_chunk": chunk}
            )
    return text_chunks, vector_store


def semantic_search(query: str, vector_store: SimpleVectorStore, embed_func, k: int = 5) -> List[SearchResult]:
    """Perform semantic search using the vector store."""
    query_embedding = embed_func([query])[0]
    return vector_store.similarity_search(query_embedding, k=k)


def prepare_context(search_results: List[SearchResult]) -> str:
    """Prepare context from search results for response generation."""
    chunk_indices = set()
    context_chunks = []
    for result in search_results:
        if result["metadata"]["type"] == "chunk":
            chunk_indices.add(result["metadata"]["index"])
            context_chunks.append(
                f"Chunk {result['metadata']['index']}:\n{result['text']}")
    for result in search_results:
        if result["metadata"]["type"] == "question":
            chunk_idx = result["metadata"]["chunk_index"]
            if chunk_idx not in chunk_indices:
                chunk_indices.add(chunk_idx)
                context_chunks.append(
                    f"Chunk {chunk_idx} (referenced by question '{result['text']}'):\n{result['metadata']['original_chunk']}")
    return "\n\n".join(context_chunks)


# Setup configuration and logging
script_dir, generated_dir, log_file, logger = setup_config(__file__)

# Initialize MLX and embedding function
mlx, embed_func = initialize_mlx(logger)

# Load pre-chunked data
formatted_texts, original_chunks = load_json_data(DOCS_PATH, logger)
logger.info("Loaded pre-chunked data from DOCS_PATH")

# Process document and generate questions
text_chunks, vector_store = process_document(
    original_chunks, embed_func, mlx, questions_per_chunk=3)
logger.debug(f"Vector store contains {len(vector_store.texts)} items")

# Load validation data
validation_data = load_validation_data(f"{DATA_DIR}/val.json", logger)
query = validation_data[0]['question']

# Perform semantic search
search_results = semantic_search(query, vector_store, embed_func, k=5)
logger.debug(f"Query: {query}")
logger.debug("\nSearch Results:")
chunk_results = [r for r in search_results if r["metadata"]["type"] == "chunk"]
question_results = [
    r for r in search_results if r["metadata"]["type"] == "question"]
logger.debug("\nRelevant Document Chunks:")
for i, result in enumerate(chunk_results):
    logger.debug(f"Context {i + 1} (similarity: {result['score']:.4f}):")
    logger.debug(result["text"][:300] + "...")
    logger.debug("=====================================")
logger.debug("\nMatched Questions:")
for i, result in enumerate(question_results):
    logger.debug(f"Question {i + 1} (similarity: {result['score']:.4f}):")
    logger.debug(result["text"])
    logger.debug(f"From chunk {result['metadata']['chunk_index']}")
    logger.debug("=====================================")

# Save search results
save_file([dict(result) for result in search_results],
          f"{generated_dir}/top_chunks.json")
logger.info(f"Saved search results to {generated_dir}/top_chunks.json")

# Generate AI response
system_prompt = (
    "You are an AI assistant that strictly answers based on the given context. "
    "If the answer cannot be derived directly from the provided context, "
    "respond with: 'I do not have enough information to answer that.'"
)
context = prepare_context(search_results)
ai_response = generate_ai_response(
    query, system_prompt, search_results, mlx, logger)
logger.debug(f"\nResponse: {ai_response}")
save_file({"question": query, "response": ai_response},
          f"{generated_dir}/ai_response.json")
logger.info(f"Saved AI response to {generated_dir}/ai_response.json")

# Evaluate response
true_answer = validation_data[0]['ideal_answer']
evaluation_score, evaluation_text = evaluate_ai_response(
    query, ai_response, true_answer, mlx, logger)
logger.success(f"Evaluation Score: {evaluation_score}")
logger.success(f"Evaluation Text: {evaluation_text}")

# Save evaluation results
save_file({
    "question": query,
    "response": ai_response,
    "true_answer": true_answer,
    "evaluation_score": evaluation_score,
    "evaluation_text": evaluation_text
}, f"{generated_dir}/evaluation.json")
logger.info(f"Saved evaluation results to {generated_dir}/evaluation.json")

logger.info("\n\n[DONE]", bright=True)
