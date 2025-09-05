import os
import numpy as np
from typing import List, Dict, Any, TypedDict
from datetime import datetime
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


def rewrite_query(original_query: str, mlx, model: str = "meta-llama/Llama-3.2-3B-Instruct") -> str:
    """Rewrite query to be more specific and detailed."""
    system_prompt = "You are an AI assistant specialized in improving search queries. Your task is to rewrite user queries to be more specific, detailed, and likely to retrieve relevant information."
    response = ""
    for chunk in mlx.stream_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Rewrite this query: {original_query}"}
        ],
        model=model,
        temperature=0
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content
        logger.success(content, flush=True)
    return response


def generate_step_back_query(original_query: str, mlx, model: str = "meta-llama/Llama-3.2-3B-Instruct") -> str:
    """Generate a broader version of the query."""
    system_prompt = "You are an AI assistant specialized in search strategies. Your task is to generate broader, more general versions of specific queries to retrieve relevant background information."
    response = ""
    for chunk in mlx.stream_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate a broader version of this query: {original_query}"}
        ],
        model=model,
        temperature=0
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content
        logger.success(content, flush=True)
    return response


def decompose_query(original_query: str, mlx, num_subqueries: int = None, model: str = "meta-llama/Llama-3.2-3B-Instruct") -> List[str]:
    """Decompose query into sub-questions."""
    current_date = datetime.now().strftime("%B %d, %Y")
    system_prompt = (
        f"You are an AI assistant specialized in breaking down complex queries into simpler sub-questions. "
        f"Decompose the user's query into {'exactly ' + str(num_subqueries) if num_subqueries else 'a reasonable number of'} clear sub-questions. "
        f"Each sub-question should be concise, specific, and end with a question mark. "
        f"List the sub-questions in a numbered format (e.g., '1. ...?'). "
        f"Today's date is {current_date}."
    )
    logger.debug(original_query)
    response = ""
    for chunk in mlx.stream_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Decompose this query: {original_query}"}
        ],
        model=model,
        temperature=0.7
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content
        logger.success(content, flush=True)
    logger.newline()
    lines = response.split("\n")
    sub_queries = []
    for line in lines:
        line = line.strip()
        if line and any(line.startswith(f"{i}.") for i in range(1, (num_subqueries or 4) + 1)):
            query = line[line.find(".") + 1:].strip()
            if query.endswith("?"):
                sub_queries.append(query)
    return sub_queries[:num_subqueries] if num_subqueries else sub_queries


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

    def similarity_search(self, query_embedding: np.ndarray, k: int = 3) -> List[SearchResult]:
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
                id=f"chunk_{idx}",
                rank=i + 1,
                doc_index=self.metadata[idx].get("index", idx),
                score=float(score),
                text=self.texts[idx],
                metadata=self.metadata[idx]
            ))
        return results


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
            metadata={"index": chunks[i]["metadata"]
                      ["doc_index"], "source": DOCS_PATH}
        )
    logger.debug(f"Added {len(text_chunks)} chunks to the vector store")
    return store


def transformed_search(query: str, vector_store: SimpleVectorStore, embed_func, mlx, transformation_type: str = None, top_k: int = 3) -> List[SearchResult]:
    """Perform search with query transformation."""
    logger.debug(f"Transformation type: {transformation_type or 'original'}")
    logger.debug(f"Original query: {query}")
    results = []
    if transformation_type == "rewrite":
        transformed_query = rewrite_query(query, mlx)
        logger.debug(f"Rewritten query: {transformed_query}")
        query_embedding = embed_func([transformed_query])[0]
        results = vector_store.similarity_search(query_embedding, k=top_k)
    elif transformation_type == "step_back":
        transformed_query = generate_step_back_query(query, mlx)
        logger.debug(f"Step-back query: {transformed_query}")
        query_embedding = embed_func([transformed_query])[0]
        results = vector_store.similarity_search(query_embedding, k=top_k)
    elif transformation_type == "decompose":
        sub_queries = decompose_query(query, mlx)
        logger.debug("Decomposed into sub-queries:")
        for i, sub_q in enumerate(sub_queries, 1):
            logger.debug(f"{i}. {sub_q}")
        sub_query_embeddings = embed_func(sub_queries)
        all_results = []
        for embedding in sub_query_embeddings:
            sub_results = vector_store.similarity_search(embedding, k=2)
            all_results.extend(sub_results)
        seen_texts = {}
        for result in all_results:
            text = result["text"]
            if text not in seen_texts or result["score"] > seen_texts[text]["score"]:
                seen_texts[text] = result
        results = sorted(seen_texts.values(),
                         key=lambda x: x["score"], reverse=True)[:top_k]
    else:
        query_embedding = embed_func(query)
        results = vector_store.similarity_search(query_embedding, k=top_k)
    return results


def compare_responses(results: Dict[str, Any], reference_answer: str, mlx, model: str = "meta-llama/Llama-3.2-3B-Instruct") -> str:
    """Compare responses from different transformations."""
    comparison_text = f"Reference Answer: {reference_answer}\n\n"
    for technique, result in results.items():
        comparison_text += f"{technique.capitalize()} Query Response:\n{result['response']}\n\n"
    response = ""
    for chunk in mlx.stream_chat(
        messages=[
            {"role": "system", "content": "You are an objective evaluator. Compare the responses and provide a concise evaluation."},
            {"role": "user", "content": comparison_text}
        ],
        model=model,
        temperature=0
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content
        logger.success(content, flush=True)
    logger.debug("\n===== EVALUATION RESULTS =====")
    logger.debug(response)
    logger.debug("=============================")
    return response


def rag_with_query_transformation(query: str, vector_store: SimpleVectorStore, embed_func, mlx, transformation_type: str = None) -> Dict[str, Any]:
    """Run RAG with query transformation."""
    results = transformed_search(
        query, vector_store, embed_func, mlx, transformation_type)
    context = "\n\n".join(
        [f"PASSAGE {i+1}:\n{result['text']}" for i, result in enumerate(results)])
    system_prompt = (
        "You are a helpful AI assistant. Answer the user's question based only on the provided context. "
        "If you cannot find the answer in the context, state that you don't have enough information."
    )
    response = generate_ai_response(query, system_prompt, results, mlx, logger)
    return {
        "original_query": query,
        "transformation_type": transformation_type,
        "context": context,
        "response": response
    }


def evaluate_transformations(query: str, vector_store: SimpleVectorStore, embed_func, mlx, reference_answer: str = None) -> Dict[str, Any]:
    """Evaluate different query transformations."""
    transformation_types = [None, "rewrite", "step_back", "decompose"]
    results = {}
    for transformation_type in transformation_types:
        type_name = transformation_type if transformation_type else "original"
        logger.debug(f"\n===== Running RAG with {type_name} query =====")
        result = rag_with_query_transformation(
            query, vector_store, embed_func, mlx, transformation_type)
        results[type_name] = result
        logger.debug(f"Response with {type_name} query: {result['response']}")
        logger.debug("=" * 50)
    save_file(results, f"{generated_dir}/transformation_results.json")
    logger.info(
        f"Saved transformation results to {generated_dir}/transformation_results.json")
    if reference_answer:
        comparison_text = compare_responses(results, reference_answer, mlx)
        save_file({"comparison": comparison_text},
                  f"{generated_dir}/comparison.json")
        logger.info(f"Saved comparison to {generated_dir}/comparison.json")
    return results


# Setup configuration and logging
script_dir, generated_dir, log_file, logger = setup_config(__file__)

# Initialize MLX and embedding function
mlx, embed_func = initialize_mlx(logger)

# Load pre-chunked data
formatted_texts, original_chunks = load_json_data(DOCS_PATH, logger)
logger.info("Loaded pre-chunked data from DOCS_PATH")

# Process document
vector_store = process_document(original_chunks, embed_func)

# Load validation data
validation_data = load_validation_data(f"{DATA_DIR}/val.json", logger)
query = validation_data[0]['question']
reference_answer = validation_data[0]['ideal_answer']

# Run query transformations and log results
logger.debug(f"Original Query: {query}")
rewritten_query = rewrite_query(query, mlx)
logger.info("\n1. Rewritten Query:")
logger.success(rewritten_query)
step_back_query = generate_step_back_query(query, mlx)
logger.info("\n2. Step-back Query:")
logger.success(step_back_query)
sub_queries = decompose_query(query, mlx, num_subqueries=4)
logger.info("\n3. Sub-queries:")
for i, query in enumerate(sub_queries, 1):
    logger.success(f"   {i}. {query}")

# Evaluate transformations
evaluation_results = evaluate_transformations(
    query, vector_store, embed_func, mlx, reference_answer)

# Save final evaluation
save_file({
    "question": query,
    "reference_answer": reference_answer,
    "evaluation_results": evaluation_results
}, f"{generated_dir}/evaluation.json")
logger.info(f"Saved evaluation results to {generated_dir}/evaluation.json")

logger.info("\n\n[DONE]", bright=True)
