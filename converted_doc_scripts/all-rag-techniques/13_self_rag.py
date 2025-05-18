import numpy as np
import re
import json
from typing import List, Dict, Any
from jet.file.utils import save_file
from helpers import (
    setup_config, initialize_mlx, generate_embeddings,
    load_validation_data, generate_ai_response,
    load_json_data, SearchResult, SimpleVectorStore, DATA_DIR, DOCS_PATH
)


def chunk_text(text: str, n: int, overlap: int) -> List[str]:
    """Chunk text into overlapping segments."""
    chunks = []
    for i in range(0, len(text), n - overlap):
        chunks.append(text[i:i + n])
    return chunks


def process_document(chunks: List[Dict[str, Any]], embed_func) -> SimpleVectorStore:
    """Process document chunks and store in vector store."""
    logger.debug("Processing chunks...")
    text_chunks = [chunk["text"] for chunk in chunks]
    logger.debug(f"Created {len(text_chunks)} text chunks")
    chunk_embeddings = generate_embeddings(text_chunks, embed_func, logger)
    store = SimpleVectorStore()
    for i, (chunk, embedding) in enumerate(zip(text_chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={"index": i, "source": chunks[i]["metadata"]["doc_index"]}
        )
    logger.debug(f"Added {len(text_chunks)} chunks to the vector store")
    return store


def determine_if_retrieval_needed(query: str, mlx, model: str = "llama-3.2-1b-instruct-4bit") -> bool:
    """Determine if document retrieval is necessary."""
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


def evaluate_relevance(query: str, context: str, mlx, model: str = "llama-3.2-1b-instruct-4bit") -> str:
    """Evaluate document relevance to query."""
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


def assess_support(response: str, context: str, mlx, model: str = "llama-3.2-1b-instruct-4bit") -> str:
    """Assess if response is supported by context."""
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


def rate_utility(query: str, response: str, mlx, model: str = "llama-3.2-1b-instruct-4bit") -> int:
    """Rate response utility."""
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


def self_rag(query: str, vector_store: SimpleVectorStore, embed_func, mlx, top_k: int = 3, model: str = "llama-3.2-1b-instruct-4bit") -> Dict[str, Any]:
    """Run Self-RAG with relevance and support checks."""
    logger.debug(f"\n=== Starting Self-RAG for query: {query} ===\n")
    logger.debug("Step 1: Determining if retrieval is necessary...")
    retrieval_needed = determine_if_retrieval_needed(query, mlx, model)
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
        query_embedding = embed_func(query)
        results = vector_store.search(query_embedding, top_k=top_k)
        metrics["documents_retrieved"] = len(results)
        logger.debug(f"Retrieved {len(results)} documents")
        logger.debug("\nStep 3: Evaluating document relevance...")
        relevant_contexts = []
        for i, result in enumerate(results):
            context = result["text"]
            relevance = evaluate_relevance(query, context, mlx, model)
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
                response = generate_ai_response(query, "You are a helpful AI assistant. Provide a clear, accurate, and informative response to the query.", [
                                                {"text": context}], mlx, logger, model=model)
                logger.debug("Assessing support...")
                support_rating = assess_support(response, context, mlx, model)
                logger.debug(f"Support rating: {support_rating}")
                metrics["response_support_ratings"].append(support_rating)
                logger.debug("Rating utility...")
                utility_rating = rate_utility(query, response, mlx, model)
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
            best_response = generate_ai_response(
                query, "You are a helpful AI assistant. Provide a clear, accurate, and informative response to the query.", [], mlx, logger, model=model)
    else:
        logger.debug("\nNo retrieval needed, generating response directly...")
        best_response = generate_ai_response(
            query, "You are a helpful AI assistant. Provide a clear, accurate, and informative response to the query.", [], mlx, logger, model=model)
    metrics["best_score"] = best_score
    metrics["used_retrieval"] = retrieval_needed and best_score > 0
    logger.debug("\n=== Self-RAG Completed ===")
    return {
        "query": query,
        "response": best_response,
        "metrics": metrics
    }


def run_self_rag_example(chunks: List[Dict[str, Any]], embed_func, mlx, model: str = "llama-3.2-1b-instruct-4bit") -> Dict[str, Any]:
    """Run Self-RAG examples."""
    logger.debug("Processing document...")
    vector_store = process_document(chunks, embed_func)
    query1 = "What are the main ethical concerns in AI development?"
    logger.debug("\n" + "="*80)
    logger.debug(f"EXAMPLE 1: {query1}")
    result1 = self_rag(query1, vector_store, embed_func, mlx, model=model)
    logger.debug("\nFinal response:")
    logger.debug(result1["response"])
    logger.debug("\nMetrics:")
    logger.debug(json.dumps(result1["metrics"], indent=2))
    query2 = "Can you write a short poem about artificial intelligence?"
    logger.debug("\n" + "="*80)
    logger.debug(f"EXAMPLE 2: {query2}")
    result2 = self_rag(query2, vector_store, embed_func, mlx, model=model)
    logger.debug("\nFinal response:")
    logger.debug(result2["response"])
    logger.debug("\nMetrics:")
    logger.debug(json.dumps(result2["metrics"], indent=2))
    query3 = "How might AI impact economic growth in developing countries?"
    logger.debug("\n" + "="*80)
    logger.debug(f"EXAMPLE 3: {query3}")
    result3 = self_rag(query3, vector_store, embed_func, mlx, model=model)
    logger.debug("\nFinal response:")
    logger.debug(result3["response"])
    logger.debug("\nMetrics:")
    logger.debug(json.dumps(result3["metrics"], indent=2))
    return {
        "example1": result1,
        "example2": result2,
        "example3": result3
    }


def traditional_rag(query: str, vector_store: SimpleVectorStore, embed_func, mlx, top_k: int = 3, model: str = "llama-3.2-1b-instruct-4bit") -> str:
    """Run traditional RAG."""
    logger.debug(f"\n=== Running traditional RAG for query: {query} ===\n")
    logger.debug("Retrieving documents...")
    query_embedding = embed_func(query)
    results = vector_store.search(query_embedding, top_k=top_k)
    logger.debug(f"Retrieved {len(results)} documents")
    return generate_ai_response(query, "You are a helpful AI assistant. Provide a clear, accurate, and informative response to the query.", results, mlx, logger, model=model)


def evaluate_rag_approaches(chunks: List[Dict[str, Any]], test_queries: List[str], embed_func, mlx, reference_answers: List[str] = None, model: str = "llama-3.2-1b-instruct-4bit") -> Dict[str, Any]:
    """Evaluate Self-RAG vs Traditional RAG."""
    logger.debug("=== Evaluating RAG Approaches ===")
    vector_store = process_document(chunks, embed_func)
    results = []
    for i, query in enumerate(test_queries):
        logger.debug(f"\nProcessing query {i+1}: {query}")
        self_rag_result = self_rag(
            query, vector_store, embed_func, mlx, model=model)
        self_rag_response = self_rag_result["response"]
        trad_rag_response = traditional_rag(
            query, vector_store, embed_func, mlx, model=model)
        reference = reference_answers[i] if reference_answers and i < len(
            reference_answers) else None
        comparison = compare_responses(
            query, self_rag_response, trad_rag_response, reference, mlx, model)
        results.append({
            "query": query,
            "self_rag_response": self_rag_response,
            "traditional_rag_response": trad_rag_response,
            "reference_answer": reference,
            "comparison": comparison,
            "self_rag_metrics": self_rag_result["metrics"]
        })
    overall_analysis = generate_overall_analysis(results, mlx, model)
    return {
        "results": results,
        "overall_analysis": overall_analysis
    }


def compare_responses(query: str, self_rag_response: str, trad_rag_response: str, reference: str = None, mlx=None, model: str = "llama-3.2-1b-instruct-4bit") -> str:
    """Compare Self-RAG and Traditional RAG responses."""
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


def generate_overall_analysis(results: List[Dict[str, Any]], mlx, model: str = "llama-3.2-1b-instruct-4bit") -> str:
    """Generate overall analysis of RAG approaches."""
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


script_dir, generated_dir, log_file, logger = setup_config(__file__)
mlx, embed_func = initialize_mlx(logger)
formatted_texts, original_chunks = load_json_data(DOCS_PATH, logger)
logger.info("Loaded pre-chunked data from DOCS_PATH")
test_queries = [
    "What are the main ethical concerns in AI development?",
]
reference_answers = [
    "The main ethical concerns in AI development include bias and fairness, privacy, transparency, accountability, safety, and the potential for misuse or harmful applications.",
]
evaluation_results = evaluate_rag_approaches(
    chunks=original_chunks,
    test_queries=test_queries,
    embed_func=embed_func,
    mlx=mlx,
    reference_answers=reference_answers
)
save_file(evaluation_results, f"{generated_dir}/evaluation_results.json")
logger.info(
    f"Saved evaluation results to {generated_dir}/evaluation_results.json")
logger.debug("\n=== OVERALL ANALYSIS ===\n")
logger.debug(evaluation_results["overall_analysis"])
logger.info("\n\n[DONE]", bright=True)
