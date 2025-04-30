from datetime import datetime
from jet.llm.mlx.base import MLX
from jet.llm.utils.embeddings import get_embedding_function
from jet.logger import CustomLogger
import pypdf
import json
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
DATA_DIR = os.path.join(script_dir, "data")

logger.info("Initializing MLX and embedding function")
mlx = MLX()
embed_func = get_embedding_function("mxbai-embed-large")


def rewrite_query(original_query, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    system_prompt = "You are an AI assistant specialized in improving search queries. Your task is to rewrite user queries to be more specific, detailed, and likely to retrieve relevant information."
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Rewrite this query: {original_query}"}
        ],
        model=model,
        temperature=0.0
    )
    return response["choices"][0]["message"]["content"]


def generate_step_back_query(original_query, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    system_prompt = "You are an AI assistant specialized in search strategies. Your task is to generate broader, more general versions of specific queries to retrieve relevant background information."
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate a broader version of this query: {original_query}"}
        ],
        model=model,
        temperature=0.1
    )
    return response["choices"][0]["message"]["content"]


def decompose_query(original_query, num_subqueries=None, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    # Get current date dynamically
    current_date = datetime.now().strftime("%B %d, %Y")

    # Define system prompt based on whether num_subqueries is provided
    if num_subqueries is None:
        system_prompt = """
        You are an AI assistant specialized in breaking down complex browser-based queries for web search and information retrieval, as of {current_date}. Your task is to decompose a query into simpler sub-questions that, when answered together, fully and comprehensively address all components of the original query. Each sub-question should target a specific, answerable aspect of the query to guide web searches or scraping efforts, ensuring no part of the original query is overlooked.

        Format your response as a numbered list of sub-questions, with a reasonable number of sub-questions (typically 2 to 4, depending on the query's complexity), each on a new line, with the following structure:
        1. Sub-question text
        2. Sub-question text
        ...

        Ensure each sub-question:
        - Starts with a number followed by a period (e.g., '1.', '2.').
        - Is a clear, standalone question ending with a question mark.
        - Targets a distinct aspect of the original query, collectively covering all its components, including explicit and implicit elements.
        - Is specific and designed for web search or content analysis, avoiding vague or overly broad questions.
        - Reflects the current date ({current_date}) when relevant, especially for queries about trends, recent events, or time-sensitive information.
        - Contains no additional text, headings, or explanations outside the numbered list.
        """.format(current_date=current_date)
    else:
        system_prompt = """
        You are an AI assistant specialized in breaking down complex browser-based queries for web search and information retrieval, as of {current_date}. Your task is to decompose a query into simpler sub-questions that, when answered together, fully and comprehensively address all components of the original query. Each sub-question should target a specific, answerable aspect of the query to guide web searches or scraping efforts, ensuring no part of the original query is overlooked.

        Format your response as a numbered list of exactly {num_subqueries} sub-questions, each on a new line, with the following structure:
        1. Sub-question text
        2. Sub-question text
        ...

        Ensure each sub-question:
        - Starts with a number followed by a period (e.g., '1.', '2.').
        - Is a clear, standalone question ending with a question mark.
        - Targets a distinct aspect of the original query, collectively covering all its components, including explicit and implicit elements.
        - Is specific and designed for web search or content analysis, avoiding vague or overly broad questions.
        - Reflects the current date ({current_date}) when relevant, especially for queries about trends, recent events, or time-sensitive information.
        - Contains no additional text, headings, or explanations outside the numbered list.
        """.format(current_date=current_date, num_subqueries=num_subqueries)

    logger.debug(original_query)
    stream_response = mlx.stream_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user",
                "content": f"Decompose this query into {'exactly ' + str(num_subqueries) if num_subqueries else 'a reasonable number of'} sub-questions: {original_query}"}
        ],
        model=model,
        temperature=0.2
    )
    content = ""
    for chunk in stream_response:
        chunk_content = chunk["choices"][0]["delta"]["content"]
        logger.success(chunk_content, flush=True)
        content += chunk_content
    logger.newline()
    lines = content.split("\n")
    sub_queries = []
    for line in lines:
        line = line.strip()
        if line and any(line.startswith(f"{i}.") for i in range(1, (num_subqueries or 4) + 1)):
            query = line[line.find(".") + 1:].strip()
            if query.endswith("?"):  # Ensure it's a question
                sub_queries.append(query)
    # If num_subqueries is specified, ensure exactly that number are returned
    return sub_queries[:num_subqueries] if num_subqueries else sub_queries


original_query = "What are the impacts of AI on job automation and employment?"
logger.debug("Original Query:", original_query)

rewritten_query = rewrite_query(original_query)
logger.info("\n1. Rewritten Query:")
logger.success(rewritten_query)

step_back_query = generate_step_back_query(original_query)
logger.info("\n2. Step-back Query:")
logger.success(step_back_query)

sub_queries = decompose_query(original_query, num_subqueries=4)
logger.info("\n3. Sub-queries:")
for i, query in enumerate(sub_queries, 1):
    logger.success(f"   {i}. {query}")


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


def create_embeddings(text):
    return embed_func(text)


def extract_text_from_pdf(pdf_path):
    all_text = ""
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text() or ""
            all_text += text
    return all_text


def chunk_text(text, n=1000, overlap=200):
    chunks = []
    for i in range(0, len(text), n - overlap):
        chunks.append(text[i:i + n])
    return chunks


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


def transformed_search(query, vector_store, transformation_type, top_k=3):
    logger.debug(f"Transformation type: {transformation_type}")
    logger.debug(f"Original query: {query}")
    results = []
    if transformation_type == "rewrite":
        transformed_query = rewrite_query(query)
        logger.debug(f"Rewritten query: {transformed_query}")
        query_embedding = create_embeddings(transformed_query)
        results = vector_store.similarity_search(query_embedding, k=top_k)
    elif transformation_type == "step_back":
        transformed_query = generate_step_back_query(query)
        logger.debug(f"Step-back query: {transformed_query}")
        query_embedding = create_embeddings(transformed_query)
        results = vector_store.similarity_search(query_embedding, k=top_k)
    elif transformation_type == "decompose":
        sub_queries = decompose_query(query)
        logger.debug("Decomposed into sub-queries:")
        for i, sub_q in enumerate(sub_queries, 1):
            logger.debug(f"{i}. {sub_q}")
        sub_query_embeddings = create_embeddings(sub_queries)
        all_results = []
        for i, embedding in enumerate(sub_query_embeddings):
            sub_results = vector_store.similarity_search(embedding, k=2)
            all_results.extend(sub_results)
        seen_texts = {}
        for result in all_results:
            text = result["text"]
            if text not in seen_texts or result["similarity"] > seen_texts[text]["similarity"]:
                seen_texts[text] = result
        results = sorted(seen_texts.values(),
                         key=lambda x: x["similarity"], reverse=True)[:top_k]
    else:
        query_embedding = create_embeddings(query)
        results = vector_store.similarity_search(query_embedding, k=top_k)
    return results


def generate_response(query, context, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
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


def rag_with_query_transformation(pdf_path, query, transformation_type=None):
    vector_store = process_document(pdf_path)
    if transformation_type:
        results = transformed_search(query, vector_store, transformation_type)
    else:
        query_embedding = create_embeddings(query)
        results = vector_store.similarity_search(query_embedding, k=3)
    context = "\n\n".join(
        [f"PASSAGE {i+1}:\n{result['text']}" for i, result in enumerate(results)])
    response = generate_response(query, context)
    return {
        "original_query": query,
        "transformation_type": transformation_type,
        "context": context,
        "response": response
    }


def compare_responses(results, reference_answer, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    comparison_text = f"Reference Answer: {reference_answer}\n\n"
    for technique, result in results.items():
        comparison_text += f"{technique.capitalize()} Query Response:\n{result['response']}\n\n"
    response = mlx.chat(
        [
            {"role": "system", "content": "You are an objective evaluator. Compare the responses and provide a concise evaluation."},
            {"role": "user", "content": comparison_text}
        ],
        model=model,
        temperature=0
    )
    logger.debug("\n===== EVALUATION RESULTS =====")
    logger.debug(response["choices"][0]["message"]["content"])
    logger.debug("=============================")


def evaluate_transformations(pdf_path, query, reference_answer=None):
    transformation_types = [None, "rewrite", "step_back", "decompose"]
    results = {}
    for transformation_type in transformation_types:
        type_name = transformation_type if transformation_type else "original"
        logger.debug(f"\n===== Running RAG with {type_name} query =====")
        result = rag_with_query_transformation(
            pdf_path, query, transformation_type)
        results[type_name] = result
        logger.debug(f"Response with {type_name} query:")
        logger.debug(result["response"])
        logger.debug("=" * 50)
    if reference_answer:
        compare_responses(results, reference_answer)
    return results


with open('data/val.json') as f:
    data = json.load(f)
query = data[0]['question']
reference_answer = data[0]['ideal_answer']
pdf_path = f"{DATA_DIR}/AI_Information.pdf"
evaluation_results = evaluate_transformations(
    pdf_path, query, reference_answer)
logger.info("\n\n[DONE]", bright=True)
