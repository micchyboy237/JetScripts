from jet.llm.mlx.base import MLX
from jet.llm.utils.embeddings import get_embedding_function
from jet.logger import CustomLogger
import pypdf
import json
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
    all_text = ""
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text() or ""
            all_text += text
    return all_text


def chunk_text(text, n, overlap):
    chunks = []
    for i in range(0, len(text), n - overlap):
        chunks.append(text[i:i + n])
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

    def similarity_search(self, query_embedding, k=5):
        if not self.vectors:
            return []
        query_vector = np.array(query_embedding).flatten()
        similarities = []
        for i, vector in enumerate(self.vectors):
            vector = vector.flatten()
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
                "similarity": score
            })
        return results


def create_embeddings(text):
    return embed_func(text)


def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    logger.debug("Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_path)
    logger.debug("Chunking text...")
    chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)
    logger.debug(f"Created {len(chunks)} text chunks")
    logger.debug("Creating embeddings for chunks...")
    chunk_embeddings = create_embeddings(chunks)
    store = SimpleVectorStore()
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={"index": i, "source": pdf_path}
        )
    logger.debug(f"Added {len(chunks)} chunks to the vector store")
    return chunks, store


def classify_query(query, model="llama-3.2-1b-instruct-4bit"):
    system_prompt = "Classify the query as Factual, Analytical, Opinion, or Contextual. Respond with only the category name."
    user_prompt = f"Query: {query}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0
    )
    category = response["choices"][0]["message"]["content"].strip()
    valid_categories = ["Factual", "Analytical", "Opinion", "Contextual"]
    return category if category in valid_categories else "Factual"


def factual_retrieval_strategy(query, vector_store, k=4, model="llama-3.2-1b-instruct-4bit"):
    logger.debug(f"Executing Factual retrieval strategy for: '{query}'")
    system_prompt = "Enhance this factual query to improve retrieval accuracy. Respond with only the enhanced query."
    user_prompt = f"Query: {query}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0
    )
    enhanced_query = response["choices"][0]["message"]["content"].strip()
    logger.debug(f"Enhanced query: {enhanced_query}")
    query_embedding = create_embeddings(enhanced_query)
    initial_results = vector_store.similarity_search(query_embedding, k=k*2)
    ranked_results = []
    for doc in initial_results:
        relevance_score = score_document_relevance(
            enhanced_query, doc["text"], model)
        ranked_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "similarity": doc["similarity"],
            "relevance_score": relevance_score
        })
    ranked_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return ranked_results[:k]


def analytical_retrieval_strategy(query, vector_store, k=4, model="llama-3.2-1b-instruct-4bit"):
    logger.debug(f"Executing Analytical retrieval strategy for: '{query}'")
    system_prompt = "Generate a list of sub-questions for this analytical query, one per line."
    user_prompt = f"Query: {query}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0.3
    )
    sub_queries = response["choices"][0]["message"]["content"].strip().split(
        '\n')
    sub_queries = [q.strip() for q in sub_queries if q.strip()]
    logger.debug(f"Generated sub-queries: {sub_queries}")
    all_results = []
    for sub_query in sub_queries:
        sub_query_embedding = create_embeddings(sub_query)
        results = vector_store.similarity_search(sub_query_embedding, k=2)
        all_results.extend(results)
    unique_texts = set()
    diverse_results = []
    for result in all_results:
        if result["text"] not in unique_texts:
            unique_texts.add(result["text"])
            diverse_results.append(result)
    if len(diverse_results) < k:
        main_query_embedding = create_embeddings(query)
        main_results = vector_store.similarity_search(
            main_query_embedding, k=k)
        for result in main_results:
            if result["text"] not in unique_texts and len(diverse_results) < k:
                unique_texts.add(result["text"])
                diverse_results.append(result)
    return diverse_results[:k]


def opinion_retrieval_strategy(query, vector_store, k=4, model="llama-3.2-1b-instruct-4bit"):
    logger.debug(f"Executing Opinion retrieval strategy for: '{query}'")
    system_prompt = "Identify different perspectives on this query, one per line."
    user_prompt = f"Query: {query}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0.3
    )
    viewpoints = response["choices"][0]["message"]["content"].strip().split(
        '\n')
    viewpoints = [v.strip() for v in viewpoints if v.strip()]
    logger.debug(f"Identified viewpoints: {viewpoints}")
    all_results = []
    for viewpoint in viewpoints:
        combined_query = f"{query} {viewpoint}"
        viewpoint_embedding = create_embeddings(combined_query)
        results = vector_store.similarity_search(viewpoint_embedding, k=2)
        for result in results:
            result["viewpoint"] = viewpoint
        all_results.extend(results)
    selected_results = []
    for viewpoint in viewpoints:
        viewpoint_docs = [r for r in all_results if r.get(
            "viewpoint") == viewpoint]
        if viewpoint_docs:
            selected_results.append(viewpoint_docs[0])
    remaining_slots = k - len(selected_results)
    if remaining_slots > 0:
        remaining_docs = [r for r in all_results if r not in selected_results]
        remaining_docs.sort(key=lambda x: x["similarity"], reverse=True)
        selected_results.extend(remaining_docs[:remaining_slots])
    return selected_results[:k]


def contextual_retrieval_strategy(query, vector_store, k=4, user_context=None, model="llama-3.2-1b-instruct-4bit"):
    logger.debug(f"Executing Contextual retrieval strategy for: '{query}'")
    if not user_context:
        system_prompt = "Infer the implied context in this query. Respond with only the inferred context."
        user_prompt = f"Query: {query}"
        response = mlx.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=model,
            temperature=0.1
        )
        user_context = response["choices"][0]["message"]["content"].strip()
        logger.debug(f"Inferred context: {user_context}")
    system_prompt = "Combine the query with the provided context to create a contextualized query."
    user_prompt = f"Query: {query}\nContext: {user_context}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0
    )
    contextualized_query = response["choices"][0]["message"]["content"].strip()
    logger.debug(f"Contextualized query: {contextualized_query}")
    query_embedding = create_embeddings(contextualized_query)
    initial_results = vector_store.similarity_search(query_embedding, k=k*2)
    ranked_results = []
    for doc in initial_results:
        context_relevance = score_document_context_relevance(
            query, user_context, doc["text"], model)
        ranked_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "similarity": doc["similarity"],
            "context_relevance": context_relevance
        })
    ranked_results.sort(key=lambda x: x["context_relevance"], reverse=True)
    return ranked_results[:k]


def score_document_relevance(query, document, model="llama-3.2-1b-instruct-4bit"):
    doc_preview = document[:1500] + "..." if len(document) > 1500 else document
    system_prompt = "Score the relevance of the document to the query from 0 to 10, where 10 is highly relevant. Provide only the score."
    user_prompt = f"Query: {query}\nDocument: {doc_preview}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0
    )
    score_text = response["choices"][0]["message"]["content"].strip()
    score_match = re.search(r'\b(10|[0-9])\b', score_text)
    return float(score_match.group(1)) if score_match else 5.0


def score_document_context_relevance(query, context, document, model="llama-3.2-1b-instruct-4bit"):
    doc_preview = document[:1500] + "..." if len(document) > 1500 else document
    system_prompt = "Score the relevance of the document to the query and context from 0 to 10, where 10 is highly relevant. Provide only the score."
    user_prompt = f"Query: {query}\nContext: {context}\nDocument: {doc_preview}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0
    )
    score_text = response["choices"][0]["message"]["content"].strip()
    score_match = re.search(r'\b(10|[0-9])\b', score_text)
    return float(score_match.group(1)) if score_match else 5.0


def adaptive_retrieval(query, vector_store, k=4, user_context=None, model="llama-3.2-1b-instruct-4bit"):
    query_type = classify_query(query, model)
    logger.debug(f"Query classified as: {query_type}")
    if query_type == "Factual":
        results = factual_retrieval_strategy(query, vector_store, k, model)
    elif query_type == "Analytical":
        results = analytical_retrieval_strategy(query, vector_store, k, model)
    elif query_type == "Opinion":
        results = opinion_retrieval_strategy(query, vector_store, k, model)
    elif query_type == "Contextual":
        results = contextual_retrieval_strategy(
            query, vector_store, k, user_context, model)
    else:
        results = factual_retrieval_strategy(query, vector_store, k, model)
    return results


def generate_response(query, results, query_type, model="llama-3.2-1b-instruct-4bit"):
    context = "\n\n---\n\n".join([r["text"] for r in results])
    system_prompt = "You are a helpful assistant. Answer the question based on the provided context. If you cannot answer from the context, acknowledge the limitations."
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0.2
    )
    return response["choices"][0]["message"]["content"]


def rag_with_adaptive_retrieval(pdf_path, query, k=4, user_context=None, model="llama-3.2-1b-instruct-4bit"):
    logger.debug("\n=== RAG WITH ADAPTIVE RETRIEVAL ===")
    logger.debug(f"Query: {query}")
    chunks, vector_store = process_document(pdf_path)
    query_type = classify_query(query, model)
    logger.debug(f"Query classified as: {query_type}")
    retrieved_docs = adaptive_retrieval(
        query, vector_store, k, user_context, model)
    response = generate_response(query, retrieved_docs, query_type, model)
    result = {
        "query": query,
        "query_type": query_type,
        "retrieved_documents": retrieved_docs,
        "response": response
    }
    logger.debug("\n=== RESPONSE ===")
    logger.debug(response)
    return result


def evaluate_adaptive_vs_standard(pdf_path, test_queries, reference_answers=None, model="llama-3.2-1b-instruct-4bit"):
    logger.debug("=== EVALUATING ADAPTIVE VS. STANDARD RETRIEVAL ===")
    chunks, vector_store = process_document(pdf_path)
    results = []
    for i, query in enumerate(test_queries):
        logger.debug(f"\n\nQuery {i+1}: {query}")
        logger.debug("\n--- Standard Retrieval ---")
        query_embedding = create_embeddings(query)
        standard_docs = vector_store.similarity_search(query_embedding, k=4)
        standard_response = generate_response(
            query, standard_docs, "General", model)
        logger.debug("\n--- Adaptive Retrieval ---")
        query_type = classify_query(query, model)
        adaptive_docs = adaptive_retrieval(
            query, vector_store, k=4, model=model)
        adaptive_response = generate_response(
            query, adaptive_docs, query_type, model)
        result = {
            "query": query,
            "query_type": query_type,
            "standard_retrieval": {
                "documents": standard_docs,
                "response": standard_response
            },
            "adaptive_retrieval": {
                "documents": adaptive_docs,
                "response": adaptive_response
            }
        }
        if reference_answers and i < len(reference_answers):
            result["reference_answer"] = reference_answers[i]
        results.append(result)
        logger.debug("\n--- Responses ---")
        logger.debug(f"Standard: {standard_response[:200]}...")
        logger.debug(f"Adaptive: {adaptive_response[:200]}...")
    if reference_answers:
        comparison = compare_responses(results, model)
        logger.debug("\n=== EVALUATION RESULTS ===")
        logger.debug(comparison)
    return {
        "results": results,
        "comparison": comparison if reference_answers else "No reference answers provided for evaluation"
    }


def compare_responses(results, model="llama-3.2-1b-instruct-4bit"):
    comparison_text = ""
    system_prompt = "You are an objective evaluator. Compare the responses and provide a concise evaluation."
    for i, result in enumerate(results):
        if "reference_answer" not in result:
            continue
        comparison_text += f"\n\n**Query {i+1}: {result['query']}**\n"
        comparison_text += f"*Query Type: {result['query_type']}*\n\n"
        comparison_text += f"**Reference Answer:**\n{result['reference_answer']}\n\n"
        comparison_text += f"**Standard Retrieval Response:**\n{result['standard_retrieval']['response']}\n\n"
        comparison_text += f"**Adaptive Retrieval Response:**\n{result['adaptive_retrieval']['response']}\n\n"
        user_prompt = f"Reference Answer: {result['reference_answer']}\n\nStandard Response:\n{result['standard_retrieval']['response']}\n\nAdaptive Response:\n{result['adaptive_retrieval']['response']}"
        response = mlx.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=model,
            temperature=0.2
        )
        comparison_text += f"**Comparison Analysis:**\n{response['choices'][0]['message']['content']}\n\n"
    return comparison_text


pdf_path = os.path.join(DATA_DIR, "AI_Information.pdf")
test_queries = [
    "What is Explainable AI (XAI)?",
]
reference_answers = [
    "Explainable AI (XAI) aims to make AI systems transparent and understandable by providing clear explanations of how decisions are made. This helps users trust and effectively manage AI technologies.",
]
evaluation_results = evaluate_adaptive_vs_standard(
    pdf_path=pdf_path,
    test_queries=test_queries,
    reference_answers=reference_answers
)
logger.debug(evaluation_results["comparison"])
logger.info("\n\n[DONE]", bright=True)
