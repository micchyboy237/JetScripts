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
    return store


def determine_if_retrieval_needed(query, model="llama-3.2-1b-instruct-4bit"):
    system_prompt = "Determine if retrieval from a document is necessary to answer this query accurately. Respond with 'yes' or 'no'."
    user_prompt = f"Query: {query}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0
    )
    return response["choices"][0]["message"]["content"].strip().lower() == "yes"


def evaluate_relevance(query, context, model="llama-3.2-1b-instruct-4bit"):
    max_context_length = 2000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "... [truncated]"
    system_prompt = "Evaluate if the provided document context is relevant to the query. Respond with 'relevant' or 'not relevant'."
    user_prompt = f"Query: {query}\nDocument: {context}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0
    )
    return response["choices"][0]["message"]["content"].strip().lower()


def assess_support(response, context, model="llama-3.2-1b-instruct-4bit"):
    max_context_length = 2000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "... [truncated]"
    system_prompt = "Assess if the response is supported by the provided context. Respond with 'fully supported', 'partially supported', or 'no support'."
    user_prompt = f"Response: {response}\nContext: {context}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0
    )
    return response["choices"][0]["message"]["content"].strip().lower()


def rate_utility(query, response, model="llama-3.2-1b-instruct-4bit"):
    system_prompt = "Rate the utility of the response to the query on a scale of 1 to 5, where 5 is highly useful. Provide only the number."
    user_prompt = f"Query: {query}\nResponse: {response}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0
    )
    rating = response["choices"][0]["message"]["content"].strip()
    rating_match = re.search(r'[1-5]', rating)
    return int(rating_match.group()) if rating_match else 3


def generate_response(query, context=None, model="llama-3.2-1b-instruct-4bit"):
    system_prompt = "You are a helpful AI assistant. Provide a clear, accurate, and informative response to the query."
    if context:
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
    else:
        user_prompt = f"Question: {query}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0.2
    )
    return response["choices"][0]["message"]["content"].strip()


def self_rag(query, vector_store, top_k=3, model="llama-3.2-1b-instruct-4bit"):
    logger.debug(f"\n=== Starting Self-RAG for query: {query} ===\n")
    logger.debug("Step 1: Determining if retrieval is necessary...")
    retrieval_needed = determine_if_retrieval_needed(query, model)
    logger.debug(f"Retrieval needed: {retrieval_needed}")
    metrics = {
        "retrieval_needed": retrieval_needed,
        "documents_retrieved": 0,
        "relevant_documents": 0,
        "response_support_ratings": [],
        "utility_ratings": []
    }
    best_response = None
    best_score = -1
    if retrieval_needed:
        logger.debug("\nStep 2: Retrieving relevant documents...")
        query_embedding = create_embeddings(query)
        results = vector_store.similarity_search(query_embedding, k=top_k)
        metrics["documents_retrieved"] = len(results)
        logger.debug(f"Retrieved {len(results)} documents")
        logger.debug("\nStep 3: Evaluating document relevance...")
        relevant_contexts = []
        for i, result in enumerate(results):
            context = result["text"]
            relevance = evaluate_relevance(query, context, model)
            logger.debug(f"Document {i+1} relevance: {relevance}")
            if relevance == "relevant":
                relevant_contexts.append(context)
        metrics["relevant_documents"] = len(relevant_contexts)
        logger.debug(f"Found {len(relevant_contexts)} relevant documents")
        if relevant_contexts:
            logger.debug("\nStep 4: Processing relevant contexts...")
            for i, context in enumerate(relevant_contexts):
                logger.debug(
                    f"\nProcessing context {i+1}/{len(relevant_contexts)}...")
                logger.debug("Generating response...")
                response = generate_response(query, context, model)
                logger.debug("Assessing support...")
                support_rating = assess_support(response, context, model)
                logger.debug(f"Support rating: {support_rating}")
                metrics["response_support_ratings"].append(support_rating)
                logger.debug("Rating utility...")
                utility_rating = rate_utility(query, response, model)
                logger.debug(f"Utility rating: {utility_rating}/5")
                metrics["utility_ratings"].append(utility_rating)
                support_score = {
                    "fully supported": 3,
                    "partially supported": 1,
                    "no support": 0
                }.get(support_rating, 0)
                overall_score = support_score * 5 + utility_rating
                logger.debug(f"Overall score: {overall_score}")
                if overall_score > best_score:
                    best_response = response
                    best_score = overall_score
                    logger.debug("New best response found!")
        if not relevant_contexts or best_score <= 0:
            logger.debug(
                "\nNo suitable context found or poor responses, generating without retrieval...")
            best_response = generate_response(query, model=model)
    else:
        logger.debug("\nNo retrieval needed, generating response directly...")
        best_response = generate_response(query, model=model)
    metrics["best_score"] = best_score
    metrics["used_retrieval"] = retrieval_needed and best_score > 0
    logger.debug("\n=== Self-RAG Completed ===")
    return {
        "query": query,
        "response": best_response,
        "metrics": metrics
    }


def run_self_rag_example(model="llama-3.2-1b-instruct-4bit"):
    pdf_path = os.path.join(DATA_DIR, "AI_Information.pdf")
    logger.debug(f"Processing document: {pdf_path}")
    vector_store = process_document(pdf_path)
    query1 = "What are the main ethical concerns in AI development?"
    logger.debug("\n" + "="*80)
    logger.debug(f"EXAMPLE 1: {query1}")
    result1 = self_rag(query1, vector_store, model=model)
    logger.debug("\nFinal response:")
    logger.debug(result1["response"])
    logger.debug("\nMetrics:")
    logger.debug(json.dumps(result1["metrics"], indent=2))
    query2 = "Can you write a short poem about artificial intelligence?"
    logger.debug("\n" + "="*80)
    logger.debug(f"EXAMPLE 2: {query2}")
    result2 = self_rag(query2, vector_store, model=model)
    logger.debug("\nFinal response:")
    logger.debug(result2["response"])
    logger.debug("\nMetrics:")
    logger.debug(json.dumps(result2["metrics"], indent=2))
    query3 = "How might AI impact economic growth in developing countries?"
    logger.debug("\n" + "="*80)
    logger.debug(f"EXAMPLE 3: {query3}")
    result3 = self_rag(query3, vector_store, model=model)
    logger.debug("\nFinal response:")
    logger.debug(result3["response"])
    logger.debug("\nMetrics:")
    logger.debug(json.dumps(result3["metrics"], indent=2))
    return {
        "example1": result1,
        "example2": result2,
        "example3": result3
    }


def traditional_rag(query, vector_store, top_k=3, model="llama-3.2-1b-instruct-4bit"):
    logger.debug(f"\n=== Running traditional RAG for query: {query} ===\n")
    logger.debug("Retrieving documents...")
    query_embedding = create_embeddings(query)
    results = vector_store.similarity_search(query_embedding, k=top_k)
    logger.debug(f"Retrieved {len(results)} documents")
    contexts = [result["text"] for result in results]
    combined_context = "\n\n".join(contexts)
    logger.debug("Generating response...")
    response = generate_response(query, combined_context, model)
    return response


def evaluate_rag_approaches(pdf_path, test_queries, reference_answers=None, model="llama-3.2-1b-instruct-4bit"):
    logger.debug("=== Evaluating RAG Approaches ===")
    vector_store = process_document(pdf_path)
    results = []
    for i, query in enumerate(test_queries):
        logger.debug(f"\nProcessing query {i+1}: {query}")
        self_rag_result = self_rag(query, vector_store, model=model)
        self_rag_response = self_rag_result["response"]
        trad_rag_response = traditional_rag(query, vector_store, model=model)
        reference = reference_answers[i] if reference_answers and i < len(
            reference_answers) else None
        comparison = compare_responses(
            query, self_rag_response, trad_rag_response, reference, model)
        results.append({
            "query": query,
            "self_rag_response": self_rag_response,
            "traditional_rag_response": trad_rag_response,
            "reference_answer": reference,
            "comparison": comparison,
            "self_rag_metrics": self_rag_result["metrics"]
        })
    overall_analysis = generate_overall_analysis(results, model)
    return {
        "results": results,
        "overall_analysis": overall_analysis
    }


def compare_responses(query, self_rag_response, trad_rag_response, reference=None, model="llama-3.2-1b-instruct-4bit"):
    system_prompt = "You are an objective evaluator. Compare the two responses to the query and provide a concise evaluation. If a reference answer is provided, use it to assess accuracy and completeness."
    user_prompt = f"Query: {query}\n\nSelf-RAG Response:\n{self_rag_response}\n\nTraditional RAG Response:\n{trad_rag_response}"
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


def generate_overall_analysis(results, model="llama-3.2-1b-instruct-4bit"):
    comparisons_summary = ""
    for i, result in enumerate(results):
        comparisons_summary += f"Query {i+1}: {result['query']}\n"
        comparisons_summary += f"Self-RAG metrics: Retrieval needed: {result['self_rag_metrics']['retrieval_needed']}, "
        comparisons_summary += f"Relevant docs: {result['self_rag_metrics']['relevant_documents']}/{result['self_rag_metrics']['documents_retrieved']}\n"
        comparisons_summary += f"Comparison summary: {result['comparison'][:200]}...\n\n"
    system_prompt = "Provide an overall analysis of the performance of Self-RAG versus Traditional RAG based on the provided summaries."
    user_prompt = f"Comparisons Summary:\n{comparisons_summary}"
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
    "What are the main ethical concerns in AI development?",
]
reference_answers = [
    "The main ethical concerns in AI development include bias and fairness, privacy, transparency, accountability, safety, and the potential for misuse or harmful applications.",
]
evaluation_results = evaluate_rag_approaches(
    pdf_path=pdf_path,
    test_queries=test_queries,
    reference_answers=reference_answers
)
logger.debug("\n=== OVERALL ANALYSIS ===\n")
logger.debug(evaluation_results["overall_analysis"])
logger.info("\n\n[DONE]", bright=True)
