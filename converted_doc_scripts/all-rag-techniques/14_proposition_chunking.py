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


def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk:
            chunks.append({
                "text": chunk,
                "chunk_id": len(chunks) + 1,
                "start_char": i,
                "end_char": i + len(chunk)
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

    def add_items(self, texts, embeddings, metadata_list=None):
        if metadata_list is None:
            metadata_list = [{} for _ in range(len(texts))]
        for text, embedding, metadata in zip(texts, embeddings, metadata_list):
            self.add_item(text, embedding, metadata)

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
                "similarity": float(score)
            })
        return results


def create_embeddings(texts):
    return embed_func(texts)


def generate_propositions(chunk, model="llama-3.2-1b-instruct-4bit"):
    system_prompt = "Convert the provided text into concise propositions, each representing a single fact or idea. List each proposition on a new line."
    user_prompt = f"Text to convert into propositions:\n\n{chunk['text']}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0
    )
    raw_propositions = response["choices"][0]["message"]["content"].strip().split(
        '\n')
    clean_propositions = []
    for prop in raw_propositions:
        cleaned = re.sub(r'^\s*(\d+\.|\-|\*)\s*', '', prop).strip()
        if cleaned and len(cleaned) > 10:
            clean_propositions.append(cleaned)
    return clean_propositions


def evaluate_proposition(proposition, original_text, model="llama-3.2-1b-instruct-4bit"):
    system_prompt = """Evaluate the proposition based on the original text. Provide a JSON object with scores (0-10) for:
- accuracy: How factually correct is the proposition?
- clarity: How clear and understandable is the proposition?
- completeness: Does it capture a complete idea from the text?
- conciseness: Is it succinct without losing meaning?"""
    user_prompt = f"Proposition: {proposition}\nOriginal Text: {original_text}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0
    )
    try:
        scores = json.loads(response["choices"][0]
                            ["message"]["content"].strip())
        return scores
    except json.JSONDecodeError:
        return {
            "accuracy": 5,
            "clarity": 5,
            "completeness": 5,
            "conciseness": 5
        }


def process_document_into_propositions(pdf_path, chunk_size=800, chunk_overlap=100,
                                       quality_thresholds=None, model="llama-3.2-1b-instruct-4bit"):
    if quality_thresholds is None:
        quality_thresholds = {
            "accuracy": 7,
            "clarity": 7,
            "completeness": 7,
            "conciseness": 7
        }
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    all_propositions = []
    logger.debug("Generating propositions from chunks...")
    for i, chunk in enumerate(chunks):
        logger.debug(f"Processing chunk {i+1}/{len(chunks)}...")
        chunk_propositions = generate_propositions(chunk, model)
        logger.debug(f"Generated {len(chunk_propositions)} propositions")
        for prop in chunk_propositions:
            proposition_data = {
                "text": prop,
                "source_chunk_id": chunk["chunk_id"],
                "source_text": chunk["text"]
            }
            all_propositions.append(proposition_data)
    logger.debug("\nEvaluating proposition quality...")
    quality_propositions = []
    for i, prop in enumerate(all_propositions):
        if i % 10 == 0:
            logger.debug(
                f"Evaluating proposition {i+1}/{len(all_propositions)}...")
        scores = evaluate_proposition(prop["text"], prop["source_text"], model)
        prop["quality_scores"] = scores
        passes_quality = True
        for metric, threshold in quality_thresholds.items():
            if scores.get(metric, 0) < threshold:
                passes_quality = False
                break
        if passes_quality:
            quality_propositions.append(prop)
        else:
            logger.debug(
                f"Proposition failed quality check: {prop['text'][:50]}...")
    logger.debug(
        f"\nRetained {len(quality_propositions)}/{len(all_propositions)} propositions after quality filtering")
    return chunks, quality_propositions


def build_vector_stores(chunks, propositions):
    chunk_store = SimpleVectorStore()
    chunk_texts = [chunk["text"] for chunk in chunks]
    logger.debug(f"Creating embeddings for {len(chunk_texts)} chunks...")
    chunk_embeddings = create_embeddings(chunk_texts)
    chunk_metadata = [{"chunk_id": chunk["chunk_id"],
                       "type": "chunk"} for chunk in chunks]
    chunk_store.add_items(chunk_texts, chunk_embeddings, chunk_metadata)
    prop_store = SimpleVectorStore()
    prop_texts = [prop["text"] for prop in propositions]
    logger.debug(f"Creating embeddings for {len(prop_texts)} propositions...")
    prop_embeddings = create_embeddings(prop_texts)
    prop_metadata = [
        {
            "type": "proposition",
            "source_chunk_id": prop["source_chunk_id"],
            "quality_scores": prop["quality_scores"]
        }
        for prop in propositions
    ]
    prop_store.add_items(prop_texts, prop_embeddings, prop_metadata)
    return chunk_store, prop_store


def retrieve_from_store(query, vector_store, k=5):
    query_embedding = create_embeddings(query)
    results = vector_store.similarity_search(query_embedding, k=k)
    return results


def compare_retrieval_approaches(query, chunk_store, prop_store, k=5):
    logger.debug(f"\ gravityn=== Query: {query} ===")
    logger.debug("\nRetrieving with proposition-based approach...")
    prop_results = retrieve_from_store(query, prop_store, k)
    logger.debug("Retrieving with chunk-based approach...")
    chunk_results = retrieve_from_store(query, chunk_store, k)
    logger.debug("\n=== Proposition-Based Results ===")
    for i, result in enumerate(prop_results):
        logger.debug(
            f"{i+1}) {result['text']} (Score: {result['similarity']:.4f})")
    logger.debug("\n=== Chunk-Based Results ===")
    for i, result in enumerate(chunk_results):
        truncated_text = result['text'][:150] + \
            "..." if len(result['text']) > 150 else result['text']
        logger.debug(
            f"{i+1}) {truncated_text} (Score: {result['similarity']:.4f})")
    return {
        "query": query,
        "proposition_results": prop_results,
        "chunk_results": chunk_results
    }


def generate_response(query, results, result_type="proposition", model="llama-3.2-1b-instruct-4bit"):
    context = "\n\n".join([result["text"] for result in results])
    system_prompt = "You are a helpful AI assistant. Answer the question based on the provided context. If the context is insufficient, acknowledge the limitation."
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


def evaluate_responses(query, prop_response, chunk_response, reference_answer=None, model="llama-3.2-1b-instruct-4bit"):
    system_prompt = "You are an objective evaluator. Compare the two responses to the query and provide a concise evaluation. If a reference answer is provided, use it to assess accuracy and completeness."
    user_prompt = f"Query: {query}\n\nProposition-Based Response:\n{prop_response}\n\nChunk-Based Response:\n{chunk_response}"
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


def run_proposition_chunking_evaluation(pdf_path, test_queries, reference_answers=None, model="llama-3.2-1b-instruct-4bit"):
    logger.debug("=== Starting Proposition Chunking Evaluation ===\n")
    chunks, propositions = process_document_into_propositions(
        pdf_path, model=model)
    chunk_store, prop_store = build_vector_stores(chunks, propositions)
    results = []
    for i, query in enumerate(test_queries):
        logger.debug(f"\n\n=== Testing Query {i+1}/{len(test_queries)} ===")
        logger.debug(f"Query: {query}")
        retrieval_results = compare_retrieval_approaches(
            query, chunk_store, prop_store)
        logger.debug("\nGenerating response from proposition-based results...")
        prop_response = generate_response(
            query,
            retrieval_results["proposition_results"],
            "proposition",
            model
        )
        logger.debug("Generating response from chunk-based results...")
        chunk_response = generate_response(
            query,
            retrieval_results["chunk_results"],
            "chunk",
            model
        )
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
        logger.debug("\nEvaluating responses...")
        evaluation = evaluate_responses(
            query, prop_response, chunk_response, reference, model)
        query_result = {
            "query": query,
            "proposition_results": retrieval_results["proposition_results"],
            "chunk_results": retrieval_results["chunk_results"],
            "proposition_response": prop_response,
            "chunk_response": chunk_response,
            "reference_answer": reference,
            "evaluation": evaluation
        }
        results.append(query_result)
        logger.debug("\n=== Proposition-Based Response ===")
        logger.debug(prop_response)
        logger.debug("\n=== Chunk-Based Response ===")
        logger.debug(chunk_response)
        logger.debug("\n=== Evaluation ===")
        logger.debug(evaluation)
    logger.debug("\n\n=== Generating Overall Analysis ===")
    overall_analysis = generate_overall_analysis(results, model)
    logger.debug("\n" + overall_analysis)
    return {
        "results": results,
        "overall_analysis": overall_analysis,
        "proposition_count": len(propositions),
        "chunk_count": len(chunks)
    }


def generate_overall_analysis(results, model="llama-3.2-1b-instruct-4bit"):
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        evaluations_summary += f"Evaluation Summary: {result['evaluation'][:200]}...\n\n"
    system_prompt = "Provide an overall analysis of the performance of proposition-based versus chunk-based retrieval based on the provided summaries."
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
    "What are the main ethical concerns in AI development?",
]
reference_answers = [
    "The main ethical concerns in AI development include bias and fairness, privacy, transparency, accountability, safety, and the potential for misuse or harmful applications.",
]
evaluation_results = run_proposition_chunking_evaluation(
    pdf_path=pdf_path,
    test_queries=test_queries,
    reference_answers=reference_answers
)
logger.debug("\n\n=== Overall Analysis ===")
logger.debug(evaluation_results["overall_analysis"])
logger.info("\n\n[DONE]", bright=True)
