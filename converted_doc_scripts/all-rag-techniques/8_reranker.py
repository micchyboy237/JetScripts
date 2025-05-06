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


def rerank_with_llm(query, results, top_n=3, model="llama-3.2-1b-instruct-4bit"):
    logger.debug(f"Reranking {len(results)} documents...")
    system_prompt = "You are an AI assistant. Score the relevance of the document to the query from 0 to 10, where 10 is highly relevant. Provide only the score."
    scored_results = []
    for i, result in enumerate(results):
        if i % 5 == 0:
            logger.debug(f"Scoring document {i+1}/{len(results)}...")
        user_prompt = f"Query: {query}\nDocument: {result['text']}"
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
        if score_match:
            score = float(score_match.group(1))
        else:
            logger.debug(
                f"Warning: Could not extract score from response: '{score_text}', using similarity score instead")
            score = result["similarity"] * 10
        scored_results.append({
            "text": result["text"],
            "metadata": result["metadata"],
            "similarity": result["similarity"],
            "relevance_score": score
        })
    reranked_results = sorted(
        scored_results, key=lambda x: x["relevance_score"], reverse=True)
    return reranked_results[:top_n]


def rerank_with_keywords(query, results, top_n=3):
    keywords = [word.lower() for word in query.split() if len(word) > 3]
    scored_results = []
    for result in results:
        document_text = result["text"].lower()
        base_score = result["similarity"] * 0.5
        keyword_score = 0
        for keyword in keywords:
            if keyword in document_text:
                keyword_score += 0.1
                first_position = document_text.find(keyword)
                if first_position < len(document_text) / 4:
                    keyword_score += 0.1
                frequency = document_text.count(keyword)
                keyword_score += min(0.05 * frequency, 0.2)
        final_score = base_score + keyword_score
        scored_results.append({
            "text": result["text"],
            "metadata": result["metadata"],
            "similarity": result["similarity"],
            "relevance_score": final_score
        })
    reranked_results = sorted(
        scored_results, key=lambda x: x["relevance_score"], reverse=True)
    return reranked_results[:top_n]


def generate_response(query, context, model="llama-3.2-1b-instruct-4bit"):
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


def rag_with_reranking(query, vector_store, reranking_method="llm", top_n=3, model="llama-3.2-1b-instruct-4bit"):
    query_embedding = create_embeddings(query)
    initial_results = vector_store.similarity_search(query_embedding, k=10)
    if reranking_method == "llm":
        reranked_results = rerank_with_llm(
            query, initial_results, top_n=top_n, model=model)
    elif reranking_method == "keywords":
        reranked_results = rerank_with_keywords(
            query, initial_results, top_n=top_n)
    else:
        reranked_results = initial_results[:top_n]
    context = "\n\n===\n\n".join([result["text"]
                                 for result in reranked_results])
    response = generate_response(query, context, model)
    return {
        "query": query,
        "reranking_method": reranking_method,
        "initial_results": initial_results[:top_n],
        "reranked_results": reranked_results,
        "context": context,
        "response": response
    }


def evaluate_reranking(query, standard_results, reranked_results, reference_answer=None, model="llama-3.2-1b-instruct-4bit"):
    if reference_answer:
        system_prompt = "You are an objective evaluator. Compare the responses and provide a concise evaluation."
        user_prompt = f"Reference Answer: {reference_answer}\n\nStandard Response:\n{standard_results['response']}\n\nReranked Response:\n{reranked_results['response']}"
        response = mlx.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=model,
            temperature=0
        )
        return response["choices"][0]["message"]["content"]
    return "No reference answer provided."


with open(os.path.join(DATA_DIR, 'val.json')) as f:
    data = json.load(f)
query = data[0]['question']
reference_answer = data[0]['ideal_answer']
pdf_path = os.path.join(DATA_DIR, 'AI_Information.pdf')
vector_store = process_document(pdf_path)
query = "Does AI have the potential to transform the way we live and work?"

logger.debug("Comparing retrieval methods...")
logger.debug("\n=== STANDARD RETRIEVAL ===")
standard_results = rag_with_reranking(
    query, vector_store, reranking_method="none")
logger.debug(f"\nQuery: {query}")
logger.debug(f"\nResponse:\n{standard_results['response']}")

logger.debug("\n=== LLM-BASED RERANKING ===")
llm_results = rag_with_reranking(query, vector_store, reranking_method="llm")
logger.debug(f"\nQuery: {query}")
logger.debug(f"\nResponse:\n{llm_results['response']}")

logger.debug("\n=== KEYWORD-BASED RERANKING ===")
keyword_results = rag_with_reranking(
    query, vector_store, reranking_method="keywords")
logger.debug(f"\nQuery: {query}")
logger.debug(f"\nResponse:\n{keyword_results['response']}")

evaluation = evaluate_reranking(
    query=query,
    standard_results=standard_results,
    reranked_results=llm_results,
    reference_answer=reference_answer
)
logger.debug("\n=== EVALUATION RESULTS ===")
logger.debug(evaluation)
logger.info("\n\n[DONE]", bright=True)
