from jet.logger import CustomLogger
from jet.llm.mlx.base import MLX
from jet.llm.utils.embeddings import get_embedding_function
import pypdf
import json
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

DATA_DIR = os.path.join(script_dir, "data")
logger.info("Initializing MLX and embedding function")
mlx = MLX()
embed_func = get_embedding_function("mxbai-embed-large")


def extract_text_from_pdf(pdf_path):
    all_text = ""
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text() or ""
            all_text += text
    return all_text


def chunk_text(text, chunk_size=800, overlap=0):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)
    return chunks


class SimpleVectorStore:
    def __init__(self):
        self.vectors = []
        self.documents = []
        self.metadata = []

    def add_documents(self, documents, vectors, metadata=None):
        if metadata is None:
            metadata = [{} for _ in range(len(documents))]
        for doc, vec, meta in zip(documents, vectors, metadata):
            self.documents.append(doc)
            self.vectors.append(np.array(vec))
            self.metadata.append(meta)

    def search(self, query_vector, top_k=5):
        if not self.vectors or not self.documents:
            return []
        query_array = np.array(query_vector).flatten()
        similarities = []
        for i, vector in enumerate(self.vectors):
            vector = vector.flatten()
            similarity = np.dot(query_array, vector) / (
                np.linalg.norm(query_array) * np.linalg.norm(vector)
            )
            similarities.append((i, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i, score in similarities[:top_k]:
            results.append({
                "document": self.documents[i],
                "score": float(score),
                "metadata": self.metadata[i]
            })
        return results


def create_embeddings(texts):
    if not texts:
        return []
    return embed_func(texts)


def process_document(pdf_path, chunk_size=800):
    logger.debug("Extracting text from document...")
    text = extract_text_from_pdf(pdf_path)
    logger.debug("Chunking text into non-overlapping segments...")
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=0)
    logger.debug(f"Created {len(chunks)} chunks")
    logger.debug("Generating embeddings for chunks...")
    chunk_embeddings = create_embeddings(chunks)
    vector_store = SimpleVectorStore()
    metadata = [{"chunk_index": i, "source": pdf_path}
                for i in range(len(chunks))]
    vector_store.add_documents(chunks, chunk_embeddings, metadata)
    doc_info = {
        "chunks": chunks,
        "source": pdf_path,
    }
    return chunks, vector_store, doc_info


def calculate_chunk_values(query, chunks, vector_store, irrelevant_chunk_penalty=0.2):
    query_embedding = create_embeddings([query])[0]
    num_chunks = len(chunks)
    results = vector_store.search(query_embedding, top_k=num_chunks)
    relevance_scores = {result["metadata"]["chunk_index"]                        : result["score"] for result in results}
    chunk_values = []
    for i in range(num_chunks):
        score = relevance_scores.get(i, 0.0)
        value = score - irrelevant_chunk_penalty
        chunk_values.append(value)
    return chunk_values


def find_best_segments(chunk_values, max_segment_length=20, total_max_length=30, min_segment_value=0.2):
    logger.debug("Finding optimal continuous text segments...")
    best_segments = []
    segment_scores = []
    total_included_chunks = 0
    while total_included_chunks < total_max_length:
        best_score = min_segment_value
        best_segment = None
        for start in range(len(chunk_values)):
            if any(start >= s[0] and start < s[1] for s in best_segments):
                continue
            for length in range(1, min(max_segment_length, len(chunk_values) - start) + 1):
                end = start + length
                if any(end > s[0] and end <= s[1] for s in best_segments):
                    continue
                segment_value = sum(chunk_values[start:end])
                if segment_value > best_score:
                    best_score = segment_value
                    best_segment = (start, end)
        if best_segment:
            best_segments.append(best_segment)
            segment_scores.append(best_score)
            total_included_chunks += best_segment[1] - best_segment[0]
            logger.debug(
                f"Found segment {best_segment} with score {best_score:.4f}")
        else:
            break
    best_segments = sorted(best_segments, key=lambda x: x[0])
    return best_segments, segment_scores


def reconstruct_segments(chunks, best_segments):
    reconstructed_segments = []
    for start, end in best_segments:
        segment_text = " ".join(chunks[start:end])
        reconstructed_segments.append({
            "text": segment_text,
            "segment_range": (start, end),
        })
    return reconstructed_segments


def format_segments_for_context(segments):
    context = []
    for i, segment in enumerate(segments):
        segment_header = f"SEGMENT {i+1} (Chunks {segment['segment_range'][0]}-{segment['segment_range'][1]-1}):"
        context.append(segment_header)
        context.append(segment['text'])
        context.append("-" * 80)
    return "\n\n".join(context)


def generate_response(query, context, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    logger.debug("Generating response using relevant segments as context...")
    system_prompt = "You are a helpful AI assistant. Answer the user's question based only on the provided context. If you cannot find the answer in the context, state that you don't have enough information."
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{context}\n\nQuestion: {query}"}
        ],
        model=model,
        temperature=0
    )
    return response["choices"][0]["message"]["content"]


def rag_with_rse(pdf_path, query, chunk_size=800, irrelevant_chunk_penalty=0.2):
    logger.debug("\n=== STARTING RAG WITH RELEVANT SEGMENT EXTRACTION ===")
    logger.debug(f"Query: {query}")
    chunks, vector_store, doc_info = process_document(pdf_path, chunk_size)
    logger.debug("\nCalculating relevance scores and chunk values...")
    chunk_values = calculate_chunk_values(
        query, chunks, vector_store, irrelevant_chunk_penalty)
    best_segments, scores = find_best_segments(
        chunk_values,
        max_segment_length=20,
        total_max_length=30,
        min_segment_value=0.2
    )
    logger.debug("\nReconstructing text segments from chunks...")
    segments = reconstruct_segments(chunks, best_segments)
    context = format_segments_for_context(segments)
    response = generate_response(query, context)
    result = {
        "query": query,
        "segments": segments,
        "response": response
    }
    logger.debug("\n=== FINAL RESPONSE ===")
    logger.debug(response)
    return result


def standard_top_k_retrieval(pdf_path, query, k=10, chunk_size=800):
    logger.debug("\n=== STARTING STANDARD TOP-K RETRIEVAL ===")
    logger.debug(f"Query: {query}")
    chunks, vector_store, doc_info = process_document(pdf_path, chunk_size)
    logger.debug("Creating query embedding and retrieving chunks...")
    query_embedding = create_embeddings([query])[0]
    results = vector_store.search(query_embedding, top_k=k)
    retrieved_chunks = [result["document"] for result in results]
    context = "\n\n".join([
        f"CHUNK {i+1}:\n{chunk}"
        for i, chunk in enumerate(retrieved_chunks)
    ])
    response = generate_response(query, context)
    result = {
        "query": query,
        "chunks": retrieved_chunks,
        "response": response
    }
    logger.debug("\n=== FINAL RESPONSE ===")
    logger.debug(response)
    return result


def evaluate_methods(pdf_path, query, reference_answer=None):
    logger.debug("\n========= EVALUATION =========\n")
    rse_result = rag_with_rse(pdf_path, query)
    standard_result = standard_top_k_retrieval(pdf_path, query)
    if reference_answer:
        logger.debug("\n=== COMPARING RESULTS ===")
        logger.debug("Evaluating responses against reference answer...")
        system_prompt = "You are an objective evaluator of RAG system responses."
        evaluation_prompt = (
            f"Reference Answer: {reference_answer}\n\n"
            f"RSE Response: {rse_result['response']}\n\n"
            f"Standard Top-K Response: {standard_result['response']}\n\n"
            "Compare the responses and provide a concise evaluation."
        )
        evaluation = mlx.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": evaluation_prompt}
            ],
            model="mlx-community/Llama-3.2-3B-Instruct-4bit",
            temperature=0
        )
        logger.debug("\n=== EVALUATION RESULTS ===")
        logger.debug(evaluation["choices"][0]["message"]["content"])
    return {
        "rse_result": rse_result,
        "standard_result": standard_result
    }


with open(os.path.join(DATA_DIR, 'val.json')) as f:
    data = json.load(f)
query = data[0]['question']
reference_answer = data[0]['ideal_answer']
pdf_path = os.path.join(DATA_DIR, 'AI_Information.pdf')
results = evaluate_methods(pdf_path, query, reference_answer)
logger.info("\n\n[DONE]", bright=True)
