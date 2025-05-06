from jet.llm.mlx.base import MLX
from jet.llm.utils.embeddings import get_embedding_function
from jet.logger import CustomLogger
import pypdf
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re

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
    pages = []
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text = page.extract_text() or ""
            if len(text.strip()) > 50:
                pages.append({
                    "text": text,
                    "metadata": {
                        "source": pdf_path,
                        "page": page_num + 1
                    }
                })
    logger.debug(f"Extracted {len(pages)} pages with content")
    return pages


def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "start_pos": i,
                    "end_pos": i + len(chunk_text)
                }
            })
    logger.debug(f"Created {len(chunks)} text chunks")
    return chunks


class SimpleVectorStore:
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add_item(self, text, embedding, metadata=None):
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})

    def similarity_search(self, query_embedding, k=5, filter_func=None):
        if not self.vectors:
            return []
        query_vector = np.array(query_embedding).flatten()
        similarities = []
        for i, vector in enumerate(self.vectors):
            vector = vector.flatten()
            if filter_func and not filter_func(self.metadata[i]):
                continue
            dot_product = np.dot(query_vector, vector)
            query_norm = np.linalg.norm(query_vector)
            vector_norm = np.linalg.norm(vector)
            if query_norm == 0 or vector_norm == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (query_norm * vector_norm)
            similarities.append((i, similarity))
        similarities.sort(key=lambda x: -float('inf')
                          if np.isnan(x[1]) else x[1], reverse=True)
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": float(score)
            })
        return results


def create_embeddings(texts):
    if not texts:
        return []
    return embed_func(texts)


def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    pages = extract_text_from_pdf(pdf_path)
    all_chunks = []
    for page in pages:
        page_chunks = chunk_text(page["text"], chunk_size, chunk_overlap)
        for chunk in page_chunks:
            chunk["metadata"].update(page["metadata"])
        all_chunks.extend(page_chunks)
    logger.debug("Creating embeddings for chunks...")
    chunk_texts = [chunk["text"] for chunk in all_chunks]
    chunk_embeddings = create_embeddings(chunk_texts)
    vector_store = SimpleVectorStore()
    for i, chunk in enumerate(all_chunks):
        vector_store.add_item(
            text=chunk["text"],
            embedding=chunk_embeddings[i],
            metadata=chunk["metadata"]
        )
    logger.debug(f"Vector store created with {len(all_chunks)} chunks")
    return vector_store


def generate_hypothetical_document(query, desired_length=1000, model="llama-3.2-1b-instruct-4bit"):
    system_prompt = "You are a helpful AI assistant. Generate a detailed document that answers the given question comprehensively."
    user_prompt = f"Question: {query}\n\nGenerate a document that fully answers this question:"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0.1
    )
    return response["choices"][0]["message"]["content"]


def hyde_rag(query, vector_store, k=5, should_generate_response=True, model="llama-3.2-1b-instruct-4bit"):
    logger.debug(f"\n=== Processing query with HyDE: {query} ===\n")
    logger.debug("Generating hypothetical document...")
    hypothetical_doc = generate_hypothetical_document(query, model=model)
    logger.debug(
        f"Generated hypothetical document of {len(hypothetical_doc)} characters")
    logger.debug("Creating embedding for hypothetical document...")
    hypothetical_embedding = create_embeddings([hypothetical_doc])[0]
    logger.debug(f"Retrieving {k} most similar chunks...")
    retrieved_chunks = vector_store.similarity_search(
        hypothetical_embedding, k=k)
    results = {
        "query": query,
        "hypothetical_document": hypothetical_doc,
        "retrieved_chunks": retrieved_chunks
    }
    if should_generate_response:
        logger.debug("Generating final response...")
        response = generate_response(query, retrieved_chunks, model)
        results["response"] = response
    return results


def standard_rag(query, vector_store, k=5, should_generate_response=True, model="llama-3.2-1b-instruct-4bit"):
    logger.debug(f"\n=== Processing query with Standard RAG: {query} ===\n")
    logger.debug("Creating embedding for query...")
    query_embedding = create_embeddings([query])[0]
    logger.debug(f"Retrieving {k} most similar chunks...")
    retrieved_chunks = vector_store.similarity_search(query_embedding, k=k)
    results = {
        "query": query,
        "retrieved_chunks": retrieved_chunks
    }
    if should_generate_response:
        logger.debug("Generating final response...")
        response = generate_response(query, retrieved_chunks, model)
        results["response"] = response
    return results


def generate_response(query, relevant_chunks, model="llama-3.2-1b-instruct-4bit"):
    context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
    system_prompt = "You are a helpful AI assistant. Answer the question based on the provided context. If the context is insufficient, acknowledge the limitation."
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0.5,
        max_tokens=500
    )
    return response["choices"][0]["message"]["content"]


def compare_approaches(query, vector_store, reference_answer=None, model="llama-3.2-1b-instruct-4bit"):
    hyde_result = hyde_rag(query, vector_store, model=model)
    hyde_response = hyde_result["response"]
    standard_result = standard_rag(query, vector_store, model=model)
    standard_response = standard_result["response"]
    comparison = compare_responses(
        query, hyde_response, standard_response, reference_answer, model)
    return {
        "query": query,
        "hyde_response": hyde_response,
        "hyde_hypothetical_doc": hyde_result["hypothetical_document"],
        "standard_response": standard_response,
        "reference_answer": reference_answer,
        "comparison": comparison
    }


def compare_responses(query, hyde_response, standard_response, reference=None, model="llama-3.2-1b-instruct-4bit"):
    system_prompt = "You are an objective evaluator. Compare the two responses to the query and provide a concise evaluation. If a reference answer is provided, use it to assess accuracy and completeness."
    user_prompt = f"Query: {query}\n\nHyDE Response:\n{hyde_response}\n\nStandard RAG Response:\n{standard_response}"
    if reference:
        user_prompt += f"\n\nReference Answer:\n{reference}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0
    )
    return response["choices"][0]["message"]["content"]


def run_evaluation(pdf_path, test_queries, reference_answers=None, chunk_size=1000, chunk_overlap=200, model="llama-3.2-1b-instruct-4bit"):
    vector_store = process_document(pdf_path, chunk_size, chunk_overlap)
    results = []
    for i, query in enumerate(test_queries):
        logger.debug(
            f"\n\n===== Evaluating Query {i+1}/{len(test_queries)} =====")
        logger.debug(f"Query: {query}")
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
        result = compare_approaches(query, vector_store, reference, model)
        results.append(result)
    overall_analysis = generate_overall_analysis(results, model)
    return {
        "results": results,
        "overall_analysis": overall_analysis
    }


def generate_overall_analysis(results, model="llama-3.2-1b-instruct-4bit"):
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        evaluations_summary += f"Comparison summary: {result['comparison'][:200]}...\n\n"
    system_prompt = "Provide an overall analysis of the performance of HyDE versus standard RAG based on the provided summaries."
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


def visualize_results(query, hyde_result, standard_result):
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    axs[0].text(0.5, 0.5, f"Query:\n\n{query}",
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, wrap=True)
    axs[0].axis('off')
    hypothetical_doc = hyde_result["hypothetical_document"]
    shortened_doc = hypothetical_doc[:500] + \
        "..." if len(hypothetical_doc) > 500 else hypothetical_doc
    axs[1].text(0.5, 0.5, f"Hypothetical Document:\n\n{shortened_doc}",
                horizontalalignment='center', verticalalignment='center',
                fontsize=10, wrap=True)
    axs[1].axis('off')
    hyde_chunks = [chunk["text"][:100] +
                   "..." for chunk in hyde_result["retrieved_chunks"]]
    std_chunks = [chunk["text"][:100] +
                  "..." for chunk in standard_result["retrieved_chunks"]]
    comparison_text = "Retrieved by HyDE:\n\n"
    for i, chunk in enumerate(hyde_chunks):
        comparison_text += f"{i+1}. {chunk}\n\n"
    comparison_text += "\nRetrieved by Standard RAG:\n\n"
    for i, chunk in enumerate(std_chunks):
        comparison_text += f"{i+1}. {chunk}\n\n"
    axs[2].text(0.5, 0.5, comparison_text,
                horizontalalignment='center', verticalalignment='center',
                fontsize=8, wrap=True)
    axs[2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'hyde_vs_standard_rag_comparison.png'))


pdf_path = os.path.join(DATA_DIR, "AI_Information.pdf")
vector_store = process_document(pdf_path)
query = "What are the main ethical considerations in artificial intelligence development?"
hyde_result = hyde_rag(query, vector_store)
logger.debug("\n=== HyDE Response ===")
logger.debug(hyde_result["response"])
standard_result = standard_rag(query, vector_store)
logger.debug("\n=== Standard RAG Response ===")
logger.debug(standard_result["response"])
visualize_results(query, hyde_result, standard_result)
test_queries = [
    "How does neural network architecture impact AI performance?"
]
reference_answers = [
    "Neural network architecture significantly impacts AI performance through factors like depth (number of layers), width (neurons per layer), connectivity patterns, and activation functions. Different architectures like CNNs, RNNs, and Transformers are optimized for specific tasks such as image recognition, sequence processing, and natural language understanding respectively.",
]
evaluation_results = run_evaluation(
    pdf_path=pdf_path,
    test_queries=test_queries,
    reference_answers=reference_answers
)
logger.debug("\n=== OVERALL ANALYSIS ===")
logger.debug(evaluation_results["overall_analysis"])
logger.info("\n\n[DONE]", bright=True)
