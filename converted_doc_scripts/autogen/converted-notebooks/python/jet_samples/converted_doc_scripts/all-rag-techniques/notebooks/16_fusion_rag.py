from jet.logger import CustomLogger
from openai import Ollama
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import fitz
import json
import numpy as np
import os
import re
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

"""
# Fusion Retrieval: Combining Vector and Keyword Search

In this notebook, I implement a fusion retrieval system that combines the strengths of semantic vector search with keyword-based BM25 retrieval. This approach improves retrieval quality by capturing both conceptual similarity and exact keyword matches.

## Why Fusion Retrieval Matters

Traditional RAG systems typically rely on vector search alone, but this has limitations:

- Vector search excels at semantic similarity but may miss exact keyword matches
- Keyword search is great for specific terms but lacks semantic understanding
- Different queries perform better with different retrieval methods

Fusion retrieval gives us the best of both worlds by:

- Performing both vector-based and keyword-based retrieval
- Normalizing the scores from each approach
- Combining them with a weighted formula
- Ranking documents based on the combined score

## Setting Up the Environment
We begin by importing necessary libraries.
"""
logger.info("# Fusion Retrieval: Combining Vector and Keyword Search")


"""
## Setting Up the Ollama API Client
We initialize the Ollama client to generate embeddings and responses.
"""
logger.info("## Setting Up the Ollama API Client")

client = Ollama(
    base_url="https://api.studio.nebius.com/v1/",
#     api_key=os.getenv("OPENAI_API_KEY")  # Retrieve the API key from environment variables
)

"""
## Document Processing Functions
"""
logger.info("## Document Processing Functions")

def extract_text_from_pdf(pdf_path):
    """
    Extract text content from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        str: Extracted text content
    """
    logger.debug(f"Extracting text from {pdf_path}...")  # Print the path of the PDF being processed
    pdf_document = fitz.open(pdf_path)  # Open the PDF file using PyMuPDF
    text = ""  # Initialize an empty string to store the extracted text

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]  # Get the page object
        text += page.get_text()  # Extract text from the page and append to the text string

    return text  # Return the extracted text content

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Split text into overlapping chunks.

    Args:
        text (str): Input text to chunk
        chunk_size (int): Size of each chunk in characters
        chunk_overlap (int): Overlap between chunks in characters

    Returns:
        List[Dict]: List of chunks with text and metadata
    """
    chunks = []  # Initialize an empty list to store chunks

    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk = text[i:i + chunk_size]  # Extract a chunk of the specified size
        if chunk:  # Ensure we don't add empty chunks
            chunk_data = {
                "text": chunk,  # The chunk text
                "metadata": {
                    "start_char": i,  # Start character index of the chunk
                    "end_char": i + len(chunk)  # End character index of the chunk
                }
            }
            chunks.append(chunk_data)  # Add the chunk data to the list

    logger.debug(f"Created {len(chunks)} text chunks")  # Print the number of created chunks
    return chunks  # Return the list of chunks

def clean_text(text):
    """
    Clean text by removing extra whitespace and special characters.

    Args:
        text (str): Input text

    Returns:
        str: Cleaned text
    """
    text = re.sub(r'\s+', ' ', text)

    text = text.replace('\\t', ' ')
    text = text.replace('\\n', ' ')

    text = ' '.join(text.split())

    return text

"""
## Creating Our Vector Store
"""
logger.info("## Creating Our Vector Store")

def create_embeddings(texts, model="BAAI/bge-en-icl"):
    """
    Create embeddings for the given texts.

    Args:
        texts (str or List[str]): Input text(s)
        model (str): Embedding model name

    Returns:
        List[List[float]]: Embedding vectors
    """
    input_texts = texts if isinstance(texts, list) else [texts]

    batch_size = 100
    all_embeddings = []

    for i in range(0, len(input_texts), batch_size):
        batch = input_texts[i:i + batch_size]  # Get the current batch of texts

        response = client.embeddings.create(
            model=model,
            input=batch
        )

        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)  # Add the batch embeddings to the list

    if isinstance(texts, str):
        return all_embeddings[0]

    return all_embeddings

class SimpleVectorStore:
    """
    A simple vector store implementation using NumPy.
    """
    def __init__(self):
        self.vectors = []  # List to store embedding vectors
        self.texts = []  # List to store text content
        self.metadata = []  # List to store metadata

    def add_item(self, text, embedding, metadata=None):
        """
        Add an item to the vector store.

        Args:
            text (str): The text content
            embedding (List[float]): The embedding vector
            metadata (Dict, optional): Additional metadata
        """
        self.vectors.append(np.array(embedding))  # Append the embedding vector
        self.texts.append(text)  # Append the text content
        self.metadata.append(metadata or {})  # Append the metadata (or empty dict if None)

    def add_items(self, items, embeddings):
        """
        Add multiple items to the vector store.

        Args:
            items (List[Dict]): List of text items
            embeddings (List[List[float]]): List of embedding vectors
        """
        for i, (item, embedding) in enumerate(zip(items, embeddings)):
            self.add_item(
                text=item["text"],  # Extract text from item
                embedding=embedding,  # Use corresponding embedding
                metadata={**item.get("metadata", {}), "index": i}  # Merge item metadata with index
            )

    def similarity_search_with_scores(self, query_embedding, k=5):
        """
        Find the most similar items to a query embedding with similarity scores.

        Args:
            query_embedding (List[float]): Query embedding vector
            k (int): Number of results to return

        Returns:
            List[Tuple[Dict, float]]: Top k most similar items with scores
        """
        if not self.vectors:
            return []  # Return empty list if no vectors are stored

        query_vector = np.array(query_embedding)

        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = cosine_similarity([query_vector], [vector])[0][0]  # Compute cosine similarity
            similarities.append((i, similarity))  # Append index and similarity score

        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],  # Retrieve text by index
                "metadata": self.metadata[idx],  # Retrieve metadata by index
                "similarity": float(score)  # Add similarity score
            })

        return results

    def get_all_documents(self):
        """
        Get all documents in the store.

        Returns:
            List[Dict]: All documents
        """
        return [{"text": text, "metadata": meta} for text, meta in zip(self.texts, self.metadata)]  # Combine texts and metadata

"""
## BM25 Implementation
"""
logger.info("## BM25 Implementation")

def create_bm25_index(chunks):
    """
    Create a BM25 index from the given chunks.

    Args:
        chunks (List[Dict]): List of text chunks

    Returns:
        BM25Okapi: A BM25 index
    """
    texts = [chunk["text"] for chunk in chunks]

    tokenized_docs = [text.split() for text in texts]

    bm25 = BM25Okapi(tokenized_docs)

    logger.debug(f"Created BM25 index with {len(texts)} documents")

    return bm25

def bm25_search(bm25, chunks, query, k=5):
    """
    Search the BM25 index with a query.

    Args:
        bm25 (BM25Okapi): BM25 index
        chunks (List[Dict]): List of text chunks
        query (str): Query string
        k (int): Number of results to return

    Returns:
        List[Dict]: Top k results with scores
    """
    query_tokens = query.split()

    scores = bm25.get_scores(query_tokens)

    results = []

    for i, score in enumerate(scores):
        metadata = chunks[i].get("metadata", {}).copy()
        metadata["index"] = i

        results.append({
            "text": chunks[i]["text"],
            "metadata": metadata,  # Add metadata with index
            "bm25_score": float(score)
        })

    results.sort(key=lambda x: x["bm25_score"], reverse=True)

    return results[:k]

"""
## Fusion Retrieval Function
"""
logger.info("## Fusion Retrieval Function")

def fusion_retrieval(query, chunks, vector_store, bm25_index, k=5, alpha=0.5):
    """
    Perform fusion retrieval combining vector-based and BM25 search.

    Args:
        query (str): Query string
        chunks (List[Dict]): Original text chunks
        vector_store (SimpleVectorStore): Vector store
        bm25_index (BM25Okapi): BM25 index
        k (int): Number of results to return
        alpha (float): Weight for vector scores (0-1), where 1-alpha is BM25 weight

    Returns:
        List[Dict]: Top k results based on combined scores
    """
    logger.debug(f"Performing fusion retrieval for query: {query}")

    epsilon = 1e-8

    query_embedding = create_embeddings(query)  # Create embedding for the query
    vector_results = vector_store.similarity_search_with_scores(query_embedding, k=len(chunks))  # Perform vector search

    bm25_results = bm25_search(bm25_index, chunks, query, k=len(chunks))  # Perform BM25 search

    vector_scores_dict = {result["metadata"]["index"]: result["similarity"] for result in vector_results}
    bm25_scores_dict = {result["metadata"]["index"]: result["bm25_score"] for result in bm25_results}

    all_docs = vector_store.get_all_documents()
    combined_results = []

    for i, doc in enumerate(all_docs):
        vector_score = vector_scores_dict.get(i, 0.0)  # Get vector score or 0 if not found
        bm25_score = bm25_scores_dict.get(i, 0.0)  # Get BM25 score or 0 if not found
        combined_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "vector_score": vector_score,
            "bm25_score": bm25_score,
            "index": i
        })

    vector_scores = np.array([doc["vector_score"] for doc in combined_results])
    bm25_scores = np.array([doc["bm25_score"] for doc in combined_results])

    norm_vector_scores = (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + epsilon)
    norm_bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + epsilon)

    combined_scores = alpha * norm_vector_scores + (1 - alpha) * norm_bm25_scores

    for i, score in enumerate(combined_scores):
        combined_results[i]["combined_score"] = float(score)

    combined_results.sort(key=lambda x: x["combined_score"], reverse=True)

    top_results = combined_results[:k]

    logger.debug(f"Retrieved {len(top_results)} documents with fusion retrieval")
    return top_results

"""
## Document Processing Pipeline
"""
logger.info("## Document Processing Pipeline")

def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    Process a document for fusion retrieval.

    Args:
        pdf_path (str): Path to the PDF file
        chunk_size (int): Size of each chunk in characters
        chunk_overlap (int): Overlap between chunks in characters

    Returns:
        Tuple[List[Dict], SimpleVectorStore, BM25Okapi]: Chunks, vector store, and BM25 index
    """
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

"""
## Response Generation
"""
logger.info("## Response Generation")

def generate_response(query, context):
    """
    Generate a response based on the query and context.

    Args:
        query (str): User query
        context (str): Context from retrieved documents

    Returns:
        str: Generated response
    """
    system_prompt = """You are a helpful AI assistant. Answer the user's question based on the provided context.
    If the context doesn't contain relevant information to answer the question fully, acknowledge this limitation."""

    user_prompt = f"""Context:
    {context}

    Question: {query}

    Please answer the question based on the provided context."""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # Specify the model to use
        messages=[
            {"role": "system", "content": system_prompt},  # System message to guide the assistant
            {"role": "user", "content": user_prompt}  # User message with context and query
        ],
        temperature=0.1  # Set the temperature for response generation
    )

    return response.choices[0].message.content

"""
## Main Retrieval Function
"""
logger.info("## Main Retrieval Function")

def answer_with_fusion_rag(query, chunks, vector_store, bm25_index, k=5, alpha=0.5):
    """
    Answer a query using fusion RAG.

    Args:
        query (str): User query
        chunks (List[Dict]): Text chunks
        vector_store (SimpleVectorStore): Vector store
        bm25_index (BM25Okapi): BM25 index
        k (int): Number of documents to retrieve
        alpha (float): Weight for vector scores

    Returns:
        Dict: Query results including retrieved documents and response
    """
    retrieved_docs = fusion_retrieval(query, chunks, vector_store, bm25_index, k=k, alpha=alpha)

    context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_docs])

    response = generate_response(query, context)

    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "response": response
    }

"""
## Comparing Retrieval Methods
"""
logger.info("## Comparing Retrieval Methods")

def vector_only_rag(query, vector_store, k=5):
    """
    Answer a query using only vector-based RAG.

    Args:
        query (str): User query
        vector_store (SimpleVectorStore): Vector store
        k (int): Number of documents to retrieve

    Returns:
        Dict: Query results
    """
    query_embedding = create_embeddings(query)

    retrieved_docs = vector_store.similarity_search_with_scores(query_embedding, k=k)

    context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_docs])

    response = generate_response(query, context)

    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "response": response
    }

def bm25_only_rag(query, chunks, bm25_index, k=5):
    """
    Answer a query using only BM25-based RAG.

    Args:
        query (str): User query
        chunks (List[Dict]): Text chunks
        bm25_index (BM25Okapi): BM25 index
        k (int): Number of documents to retrieve

    Returns:
        Dict: Query results
    """
    retrieved_docs = bm25_search(bm25_index, chunks, query, k=k)

    context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_docs])

    response = generate_response(query, context)

    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "response": response
    }

"""
## Evaluation Functions
"""
logger.info("## Evaluation Functions")

def compare_retrieval_methods(query, chunks, vector_store, bm25_index, k=5, alpha=0.5, reference_answer=None):
    """
    Compare different retrieval methods for a query.

    Args:
        query (str): User query
        chunks (List[Dict]): Text chunks
        vector_store (SimpleVectorStore): Vector store
        bm25_index (BM25Okapi): BM25 index
        k (int): Number of documents to retrieve
        alpha (float): Weight for vector scores in fusion retrieval
        reference_answer (str, optional): Reference answer for comparison

    Returns:
        Dict: Comparison results
    """
    logger.debug(f"\n=== Comparing retrieval methods for query: {query} ===\n")

    logger.debug("\nRunning vector-only RAG...")
    vector_result = vector_only_rag(query, vector_store, k)

    logger.debug("\nRunning BM25-only RAG...")
    bm25_result = bm25_only_rag(query, chunks, bm25_index, k)

    logger.debug("\nRunning fusion RAG...")
    fusion_result = answer_with_fusion_rag(query, chunks, vector_store, bm25_index, k, alpha)

    logger.debug("\nComparing responses...")
    comparison = evaluate_responses(
        query,
        vector_result["response"],
        bm25_result["response"],
        fusion_result["response"],
        reference_answer
    )

    return {
        "query": query,
        "vector_result": vector_result,
        "bm25_result": bm25_result,
        "fusion_result": fusion_result,
        "comparison": comparison
    }

def evaluate_responses(query, vector_response, bm25_response, fusion_response, reference_answer=None):
    """
    Evaluate the responses from different retrieval methods.

    Args:
        query (str): User query
        vector_response (str): Response from vector-only RAG
        bm25_response (str): Response from BM25-only RAG
        fusion_response (str): Response from fusion RAG
        reference_answer (str, optional): Reference answer

    Returns:
        str: Evaluation of responses
    """
    system_prompt = """You are an expert evaluator of RAG systems. Compare responses from three different retrieval approaches:
    1. Vector-based retrieval: Uses semantic similarity for document retrieval
    2. BM25 keyword retrieval: Uses keyword matching for document retrieval
    3. Fusion retrieval: Combines both vector and keyword approaches

    Evaluate the responses based on:
    - Relevance to the query
    - Factual correctness
    - Comprehensiveness
    - Clarity and coherence"""

    user_prompt = f"""Query: {query}

    Vector-based response:
    {vector_response}

    BM25 keyword response:
    {bm25_response}

    Fusion response:
    {fusion_response}
    """

    if reference_answer:
        user_prompt += f"""
            Reference answer:
            {reference_answer}
        """

    user_prompt += """
    Please provide a detailed comparison of these three responses. Which approach performed best for this query and why?
    Be specific about the strengths and weaknesses of each approach for this particular query.
    """

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # Specify the model to use
        messages=[
            {"role": "system", "content": system_prompt},  # System message to guide the evaluator
            {"role": "user", "content": user_prompt}  # User message with query and responses
        ],
        temperature=0  # Set the temperature for response generation
    )

    return response.choices[0].message.content

"""
## Complete Evaluation Pipeline
"""
logger.info("## Complete Evaluation Pipeline")

def evaluate_fusion_retrieval(pdf_path, test_queries, reference_answers=None, k=5, alpha=0.5):
    """
    Evaluate fusion retrieval compared to other methods.

    Args:
        pdf_path (str): Path to the PDF file
        test_queries (List[str]): List of test queries
        reference_answers (List[str], optional): Reference answers
        k (int): Number of documents to retrieve
        alpha (float): Weight for vector scores in fusion retrieval

    Returns:
        Dict: Evaluation results
    """
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
            reference_answer=reference
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

    overall_analysis = generate_overall_analysis(results)

    return {
        "results": results,
        "overall_analysis": overall_analysis
    }

def generate_overall_analysis(results):
    """
    Generate an overall analysis of fusion retrieval.

    Args:
        results (List[Dict]): Results from evaluating queries

    Returns:
        str: Overall analysis
    """
    system_prompt = """You are an expert at evaluating information retrieval systems.
    Based on multiple test queries, provide an overall analysis comparing three retrieval approaches:
    1. Vector-based retrieval (semantic similarity)
    2. BM25 keyword retrieval (keyword matching)
    3. Fusion retrieval (combination of both)

    Focus on:
    1. Types of queries where each approach performs best
    2. Overall strengths and weaknesses of each approach
    3. How fusion retrieval balances the trade-offs
    4. Recommendations for when to use each approach"""

    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        evaluations_summary += f"Comparison Summary: {result['comparison'][:200]}...\n\n"

    user_prompt = f"""Based on the following evaluations of different retrieval methods across {len(results)} queries,
    provide an overall analysis comparing these three approaches:

    {evaluations_summary}

    Please provide a comprehensive analysis of vector-based, BM25, and fusion retrieval approaches,
    highlighting when and why fusion retrieval provides advantages over the individual methods."""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content

"""
## Evaluating Fusion Retrieval
"""
logger.info("## Evaluating Fusion Retrieval")

pdf_path = f"{GENERATED_DIR}/AI_Information.pdf"

test_queries = [
    "What are the main applications of transformer models in natural language processing?"  # AI-specific query
]

reference_answers = [
    "Transformer models have revolutionized natural language processing with applications including machine translation, text summarization, question answering, sentiment analysis, and text generation. They excel at capturing long-range dependencies in text and have become the foundation for models like BERT, GPT, and T5.",
]

k = 5  # Number of documents to retrieve
alpha = 0.5  # Weight for vector scores (0.5 means equal weight between vector and BM25)

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