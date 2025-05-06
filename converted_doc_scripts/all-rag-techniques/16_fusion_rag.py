from jet.llm.mlx.base import MLX
from jet.llm.utils.embeddings import get_embedding_function
from jet.logger import CustomLogger
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import pypdf
import json
import numpy as np
import os
import re
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")
DATA_DIR = os.path.join(script_dir, "data")
os.makedirs(DATA_DIR, exist_ok=True)
logger.info("Initializing MLX and embedding function")
mlx = MLX()
embed_func = get_embedding_function("mxbai-embed-large")


def extract_text_from_pdf(pdf_path):
    logger.debug(f"Extracting text from {pdf_path}...")
    all_text = ""
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text() or ""
            all_text += text
    return all_text


def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk = text[i:i + chunk_size]
        if chunk:
            chunk_data = {
                "text": chunk,
                "metadata": {
                    "start_char": i,
                    "end_char": i + len(chunk)
                }
            }
            chunks.append(chunk_data)
    logger.debug(f"Created {len(chunks)} text chunks")
    return chunks


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\\t', ' ')
    text = text.replace('\\n', ' ')
    text = ' '.join(text.split())
    return text


def create_embeddings(texts):
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

    def add_items(self, items, embeddings):
        for i, (item, embedding) in enumerate(zip(items, embeddings)):
            self.add_item(
                text=item["text"],
                embedding=embedding,
                metadata={**item.get("metadata", {}), "index": i}
            )

    def similarity_search_with_scores(self, query_embedding, k=5):
        if not self.vectors:
            return []
        query_vector = np.array(query_embedding).flatten()
        similarities = []
        for i, vector in enumerate(self.vectors):
            vector = vector.flatten()
            similarity = cosine_similarity([query_vector], [vector])[0][0]
            similarities.append((i, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": float(score)
            })
        return results

    def get_all_documents(self):
        return [{"text": text, "metadata": meta} for text, meta in zip(self.texts, self.metadata)]


def create_bm25_index(chunks):
    texts = [chunk["text"] for chunk in chunks]
    tokenized_docs = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized_docs)
    logger.debug(f"Created BM25 index with {len(texts)} documents")
    return bm25


def bm25_search(bm25, chunks, query, k=5):
    query_tokens = query.split()
    scores = bm25.get_scores(query_tokens)
    results = []
    for i, score in enumerate(scores):
        metadata = chunks[i].get("metadata", {}).copy()
        metadata["index"] = i
        results.append({
            "text": chunks[i]["text"],
            "metadata": metadata,
            "bm25_score": float(score)
        })
    results.sort(key=lambda x: x["bm25_score"], reverse=True)
    return results[:k]


def fusion_retrieval(query, chunks, vector_store, bm25_index, k=5, alpha=0.5):
    logger.debug(f"Performing fusion retrieval for query: {query}")
    epsilon = 1e-8
    query_embedding = create_embeddings(query)
    vector_results = vector_store.similarity_search_with_scores(
        query_embedding, k=len(chunks))
    bm25_results = bm25_search(bm25_index, chunks, query, k=len(chunks))
    vector_scores_dict = {result["metadata"]["index"]: result["similarity"] for result in vector_results}
    bm25_scores_dict = {result["metadata"]["index"]: result["bm25_score"] for result in bm25_results}
    all_docs = vector_store.get_all_documents()
    combined_results = []
    for i, doc in enumerate(all_docs):
        vector_score = vector_scores_dict.get(i, 0.0)
        bm25_score = bm25_scores_dict.get(i, 0.0)
        combined_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "vector_score": vector_score,
            "bm25_score": bm25_score,
            "index": i
        })
    vector_scores = np.array([doc["vector_score"] for doc in combined_results])
    bm25_scores = np.array([doc["bm25_score"] for doc in combined_results])
    norm_vector_scores = (vector_scores - np.min(vector_scores)) / \
        (np.max(vector_scores) - np.min(vector_scores) + epsilon)
    norm_bm25_scores = (bm25_scores - np.min(bm25_scores)) / \
        (np.max(bm25_scores) - np.min(bm25_scores) + epsilon)
    combined_scores = alpha * norm_vector_scores + \
        (1 - alpha) * norm_bm25_scores
    for i, score in enumerate(combined_scores):
        combined_results[i]["combined_score"] = float(score)
    combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
    top_results = combined_results[:k]
    logger.debug(
        f"Retrieved {len(top_results)} documents with fusion retrieval")
    return top_results


def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(text)
    chunks = chunk_text(cleaned_text, chunk_size, chunk_overlap)
    chunk_texts = [chunk["text"] for chunk in chunks]
    logger.debug("Creating embeddings for chunks...")
    embeddings = create_embeddings(chunk_texts)
    vector_store = SimpleVectorStore()
    vector_store.add_items(chunks, embeddings)
    logger.debug(f"Added {len(chunks)} items to vector store")
    bm25_index = create_bm25_index(chunks)
    return chunks, vector_store, bm25_index


def generate_response(query, context, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    system_prompt = "You are a helpful AI assistant. Answer the question based on the provided context. If the context is insufficient, acknowledge the limitation."
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0.1
    )
    return response["choices"][0]["message"]["content"]


def answer_with_fusion_rag(query, chunks, vector_store, bm25_index, k=5, alpha=0.5, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    retrieved_docs = fusion_retrieval(
        query, chunks, vector_store, bm25_index, k=k, alpha=alpha)
    context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_docs])
    response = generate_response(query, context, model)
    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "response": response
    }


def vector_only_rag(query, vector_store, k=5, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    query_embedding = create_embeddings(query)
    retrieved_docs = vector_store.similarity_search_with_scores(
        query_embedding, k=k)
    context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_docs])
    response = generate_response(query, context, model)
    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "response": response
    }


def bm25_only_rag(query, chunks, bm25_index, k=5, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    retrieved_docs = bm25_search(bm25_index, chunks, query, k=k)
    context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_docs])
    response = generate_response(query, context, model)
    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "response": response
    }


def compare_retrieval_methods(query, chunks, vector_store, bm25_index, k=5, alpha=0.5, reference_answer=None, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    logger.debug(f"\n=== Comparing retrieval methods for query: {query} ===\n")
    logger.debug("\nRunning vector-only RAG...")
    vector_result = vector_only_rag(query, vector_store, k, model)
    logger.debug("\nRunning BM25-only RAG...")
    bm25_result = bm25_only_rag(query, chunks, bm25_index, k, model)
    logger.debug("\nRunning fusion RAG...")
    fusion_result = answer_with_fusion_rag(
        query, chunks, vector_store, bm25_index, k, alpha, model)
    logger.debug("\nComparing responses...")
    comparison = evaluate_responses(
        query,
        vector_result["response"],
        bm25_result["response"],
        fusion_result["response"],
        reference_answer,
        model
    )
    return {
        "query": query,
        "vector_result": vector_result,
        "bm25_result": bm25_result,
        "fusion_result": fusion_result,
        "comparison": comparison
    }


def evaluate_responses(query, vector_response, bm25_response, fusion_response, reference_answer=None, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    system_prompt = "You are an objective evaluator. Compare the three responses to the query and provide a concise evaluation. If a reference answer is provided, use it to assess accuracy and completeness."
    user_prompt = f"Query: {query}\n\nVector-only Response:\n{vector_response}\n\nBM25 Response:\n{bm25_response}\n\nFusion Response:\n{fusion_response}"
    if reference_answer:
        user_prompt += f"\n\nReference Answer:\n{reference_answer}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0
    )
    return response["choices"][0]["message"]["content"]


def evaluate_fusion_retrieval(pdf_path, test_queries, reference_answers=None, k=5, alpha=0.5, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    logger.debug("=== EVALUATING FUSION RETRIEVAL ===\n")
    chunks, vector_store, bm25_index = process_document(pdf_path)
    results = []
    for i, query in enumerate(test_queries):
        logger.debug(f"\n\n=== Evaluating Query {i+1}/{len(test_queries)} ===")
        logger.debug(f"Query: {query}")
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
        comparison = compare_retrieval_methods(
            query,
            chunks,
            vector_store,
            bm25_index,
            k=k,
            alpha=alpha,
            reference_answer=reference,
            model=model
        )
        results.append(comparison)
        logger.debug("\n=== Vector-based Response ===")
        logger.debug(comparison["vector_result"]["response"])
        logger.debug("\n=== BM25 Response ===")
        logger.debug(comparison["bm25_result"]["response"])
        logger.debug("\n=== Fusion Response ===")
        logger.debug(comparison["fusion_result"]["response"])
        logger.debug("\n=== Comparison ===")
        logger.debug(comparison["comparison"])
    overall_analysis = generate_overall_analysis(results, model)
    return {
        "results": results,
        "overall_analysis": overall_analysis
    }


def generate_overall_analysis(results, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        evaluations_summary += f"Comparison Summary: {result['comparison'][:200]}...\n\n"
    system_prompt = "Provide an overall analysis of the performance of vector-only, BM25, and fusion retrieval methods based on the provided summaries."
    user_prompt = f"Evaluations Summary:\n{evaluations_summary}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0
    )
    return response["choices"][0]["message"]["content"]


pdf_path = os.path.join(DATA_DIR, "AI_Information.pdf")
test_queries = [
    "What are the main applications of transformer models in natural language processing?"
]
reference_answers = [
    "Transformer models have revolutionized natural language processing with applications including machine translation, text summarization, question answering, sentiment analysis, and text generation. They excel at capturing long-range dependencies in text and have become the foundation for models like BERT, GPT, and T5.",
]
k = 5
alpha = 0.5
evaluation_results = evaluate_fusion_retrieval(
    pdf_path=pdf_path,
    test_queries=test_queries,
    reference_answers=reference_answers,
    k=k,
    alpha=alpha
)
logger.debug("\n\n=== OVERALL ANALYSIS ===\n")
logger.debug(evaluation_results["overall_analysis"])
logger.info("\n\n[DONE]", bright=True)
