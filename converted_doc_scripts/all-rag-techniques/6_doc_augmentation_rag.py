import asyncio
import platform
from jet.logger import CustomLogger
from jet.llm.mlx.base import MLX
from jet.llm.utils.embeddings import get_embedding_function
from tqdm import tqdm
import pypdf
import json
import numpy as np
import os
import re

# Setup logging and directories
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")
file_name = os.path.splitext(os.path.basename(__file__))[0]
DATA_DIR = os.path.join(script_dir, "data")
logger.info("Initializing document processing pipeline")

# Initialize MLX and embedding function
logger.info("Initializing MLX and embedding function")
mlx = MLX()
embed_func = get_embedding_function("mxbai-embed-large")


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using pypdf."""
    logger.debug(f"Extracting text from {pdf_path}")
    all_text = ""
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text() or ""
            all_text += text
    return all_text


def chunk_text(text, n, overlap):
    """Split text into chunks with specified size and overlap."""
    chunks = []
    for i in range(0, len(text), n - overlap):
        chunks.append(text[i:i + n])
    return chunks


def generate_questions(text_chunk, num_questions=5, model="meta-llama/Llama-3.2-3B-Instruct"):
    """Generate questions from text chunk using MLX."""
    system_prompt = "You are an expert at generating relevant questions from text. Create concise questions that can be answered using only the provided text. Focus on key information and concepts."
    user_prompt = f"Generate {num_questions} questions based on the following text:\n\n{text_chunk}"
    response = mlx.chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    questions_text = response.strip()
    questions = []
    for line in questions_text.split('\n'):
        cleaned_line = re.sub(r'^\d+\.\s*', '', line.strip())
        if cleaned_line and cleaned_line.endswith('?'):
            questions.append(cleaned_line)
    return questions


def create_embeddings(texts):
    """Create embeddings using the embedding function."""
    return embed_func(texts)


class SimpleVectorStore:
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add_item(self, text, embedding, metadata=None):
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})

    def similarity_search(self, query_embedding, k=5):
        if not self.vectors:
            return []
        query_vector = np.array(
            query_embedding).flatten()  # Ensure query is 1D
        similarities = []
        for i, vector in enumerate(self.vectors):
            vector = vector.flatten()  # Ensure stored vector is 1D
            # Calculate cosine similarity
            dot_product = np.dot(query_vector, vector)
            query_norm = np.linalg.norm(query_vector)
            vector_norm = np.linalg.norm(vector)
            # Check for zero norms to avoid division by zero
            if query_norm == 0 or vector_norm == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (query_norm * vector_norm)
            similarities.append((i, similarity))
        # Sort similarities, handling NaN or invalid values
        similarities.sort(key=lambda x: -float('inf')
                          if np.isnan(x[1]) else x[1], reverse=True)
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": score
            })
        return results


def process_document(pdf_path, chunk_size=1000, chunk_overlap=200, questions_per_chunk=5):
    logger.debug("Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_path)
    logger.debug("Chunking text...")
    text_chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)
    logger.debug(f"Created {len(text_chunks)} text chunks")
    vector_store = SimpleVectorStore()
    logger.debug("Processing chunks and generating questions...")

    # Generate embeddings for all chunks at once
    chunk_embeddings = create_embeddings(text_chunks)

    for i, (chunk, chunk_embedding) in enumerate(tqdm(zip(text_chunks, chunk_embeddings), total=len(text_chunks), desc="Processing Chunks")):
        vector_store.add_item(
            text=chunk,
            embedding=chunk_embedding,
            metadata={"type": "chunk", "index": i}
        )
        questions = generate_questions(
            chunk, num_questions=questions_per_chunk)
        question_embeddings = create_embeddings(questions)
        for j, (question, question_embedding) in enumerate(zip(questions, question_embeddings)):
            vector_store.add_item(
                text=question,
                embedding=question_embedding,
                metadata={"type": "question",
                          "chunk_index": i, "original_chunk": chunk}
            )
    return text_chunks, vector_store


def semantic_search(query, vector_store, k=5):
    query_embedding = create_embeddings([query])[0]
    results = vector_store.similarity_search(query_embedding, k=k)
    return results


def prepare_context(search_results):
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
    full_context = "\n\n".join(context_chunks)
    return full_context


def generate_response(query, context, model="meta-llama/Llama-3.2-3B-Instruct"):
    system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"
    user_prompt = f"{context}\n\nQuestion: {query}"
    response = mlx.chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    return response


def evaluate_response(query, response, reference_answer, model="meta-llama/Llama-3.2-3B-Instruct"):
    evaluate_system_prompt = "You are an intelligent evaluation system tasked with assessing the AI assistant's responses. If the AI assistant's response is very close to the true response, assign a score of 1. If the response is incorrect or unsatisfactory in relation to the true response, assign a score of 0. If the response is partially aligned with the true response, assign a score of 0.5."
    evaluation_prompt = f"User Query: {query}\nAI Response:\n{response}\nTrue Response: {reference_answer}\n{evaluate_system_prompt}"
    eval_response = mlx.chat([
        {"role": "system", "content": evaluate_system_prompt},
        {"role": "user", "content": evaluation_prompt}
    ])
    try:
        score = float(eval_response.strip())
    except ValueError:
        logger.debug(
            "Warning: Could not parse evaluation score, defaulting to 0")
        score = 0.0
    return score


async def main():
    # Process document
    pdf_path = f"{DATA_DIR}/AI_Information.pdf"
    text_chunks, vector_store = process_document(
        pdf_path,
        chunk_size=1000,
        chunk_overlap=200,
        questions_per_chunk=3
    )
    logger.debug(f"Vector store contains {len(vector_store.texts)} items")

    # Load validation data and perform semantic search
    with open('data/val.json') as f:
        data = json.load(f)
    query = data[0]['question']
    search_results = semantic_search(query, vector_store, k=5)

    # Log search results
    logger.debug("Query:", query)
    logger.debug("\nSearch Results:")
    chunk_results = []
    question_results = []
    for result in search_results:
        if result["metadata"]["type"] == "chunk":
            chunk_results.append(result)
        else:
            question_results.append(result)

    logger.debug("\nRelevant Document Chunks:")
    for i, result in enumerate(chunk_results):
        logger.debug(
            f"Context {i + 1} (similarity: {result['similarity']:.4f}):")
        logger.debug(result["text"][:300] + "...")
        logger.debug("=====================================")

    logger.debug("\nMatched Questions:")
    for i, result in enumerate(question_results):
        logger.debug(
            f"Question {i + 1} (similarity: {result['similarity']:.4f}):")
        logger.debug(result["text"])
        chunk_idx = result["metadata"]["chunk_index"]
        logger.debug(f"From chunk {chunk_idx}")
        logger.debug("=====================================")

    # Generate and evaluate response
    context = prepare_context(search_results)
    response_text = generate_response(query, context)
    logger.debug("\nQuery:", query)
    logger.debug("\nResponse:")
    logger.debug(response_text)

    reference_answer = data[0]['ideal_answer']
    evaluation = evaluate_response(query, response_text, reference_answer)
    logger.debug("\nEvaluation:")
    logger.debug(f"Score: {evaluation}")

    logger.info("\n\n[DONE]", bright=True)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
