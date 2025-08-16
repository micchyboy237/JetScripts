from jet.logger import CustomLogger
from openai import MLX
import fitz
import json
import numpy as np
import os
import pickle
import re
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

"""
# Hierarchical Indices for RAG

In this notebook, I implement a hierarchical indexing approach for RAG systems. This technique improves retrieval by using a two-tier search method: first identifying relevant document sections through summaries, then retrieving specific details from those sections.

Traditional RAG approaches treat all text chunks equally, which can lead to:

- Lost context when chunks are too small
- Irrelevant results when the document collection is large
- Inefficient searches across the entire corpus

Hierarchical retrieval solves these problems by:

- Creating concise summaries for larger document sections
- First searching these summaries to identify relevant sections
- Then retrieving detailed information only from those sections
- Maintaining context while preserving specific details

## Setting Up the Environment
We begin by importing necessary libraries.
"""
logger.info("# Hierarchical Indices for RAG")


"""
## Setting Up the MLX API Client
We initialize the MLX client to generate embeddings and responses.
"""
logger.info("## Setting Up the MLX API Client")

client = MLX(
    base_url="https://api.studio.nebius.com/v1/",
#     api_key=os.getenv("OPENAI_API_KEY")  # Retrieve the API key from environment variables
)

"""
## Document Processing Functions
"""
logger.info("## Document Processing Functions")

def extract_text_from_pdf(pdf_path):
    """
    Extract text content from a PDF file with page separation.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        List[Dict]: List of pages with text content and metadata
    """
    logger.debug(f"Extracting text from {pdf_path}...")  # Print the path of the PDF being processed
    pdf = fitz.open(pdf_path)  # Open the PDF file using PyMuPDF
    pages = []  # Initialize an empty list to store the pages with text content

    for page_num in range(len(pdf)):
        page = pdf[page_num]  # Get the current page
        text = page.get_text()  # Extract text from the current page

        if len(text.strip()) > 50:
            pages.append({
                "text": text,
                "metadata": {
                    "source": pdf_path,  # Source file path
                    "page": page_num + 1  # Page number (1-based index)
                }
            })

    logger.debug(f"Extracted {len(pages)} pages with content")  # Print the number of pages extracted
    return pages  # Return the list of pages with text content and metadata

"""

"""

def chunk_text(text, metadata, chunk_size=1000, overlap=200):
    """
    Split text into overlapping chunks while preserving metadata.

    Args:
        text (str): Input text to chunk
        metadata (Dict): Metadata to preserve
        chunk_size (int): Size of each chunk in characters
        overlap (int): Overlap between chunks in characters

    Returns:
        List[Dict]: List of text chunks with metadata
    """
    chunks = []  # Initialize an empty list to store the chunks

    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]  # Extract the chunk of text

        if chunk_text and len(chunk_text.strip()) > 50:
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": len(chunks),  # Index of the chunk
                "start_char": i,  # Start character index of the chunk
                "end_char": i + len(chunk_text),  # End character index of the chunk
                "is_summary": False  # Flag indicating this is not a summary
            })

            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })

    return chunks  # Return the list of chunks with metadata

"""
## Simple Vector Store Implementation
"""
logger.info("## Simple Vector Store Implementation")

class SimpleVectorStore:
    """
    A simple vector store implementation using NumPy.
    """
    def __init__(self):
        self.vectors = []  # List to store vector embeddings
        self.texts = []  # List to store text content
        self.metadata = []  # List to store metadata

    def add_item(self, text, embedding, metadata=None):
        """
        Add an item to the vector store.

        Args:
            text (str): Text content
            embedding (List[float]): Vector embedding
            metadata (Dict, optional): Additional metadata
        """
        self.vectors.append(np.array(embedding))  # Append the embedding as a numpy array
        self.texts.append(text)  # Append the text content
        self.metadata.append(metadata or {})  # Append the metadata or an empty dict if None

    def similarity_search(self, query_embedding, k=5, filter_func=None):
        """
        Find the most similar items to a query embedding.

        Args:
            query_embedding (List[float]): Query embedding vector
            k (int): Number of results to return
            filter_func (callable, optional): Function to filter results

        Returns:
            List[Dict]: Top k most similar items
        """
        if not self.vectors:
            return []  # Return an empty list if there are no vectors

        query_vector = np.array(query_embedding)

        similarities = []
        for i, vector in enumerate(self.vectors):
            if filter_func and not filter_func(self.metadata[i]):
                continue

            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))  # Append index and similarity score

        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],  # Add the text content
                "metadata": self.metadata[idx],  # Add the metadata
                "similarity": float(score)  # Add the similarity score
            })

        return results  # Return the list of top k results

"""
## Creating Embeddings
"""
logger.info("## Creating Embeddings")

def create_embeddings(texts, model="BAAI/bge-en-icl"):
    """
    Create embeddings for the given texts.

    Args:
        texts (List[str]): Input texts
        model (str): Embedding model name

    Returns:
        List[List[float]]: Embedding vectors
    """
    if not texts:
        return []

    batch_size = 100
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]  # Get the current batch of texts

        response = client.embeddings.create(
            model=model,
            input=batch
        )

        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)  # Add the batch embeddings to the list

    return all_embeddings  # Return all embeddings

"""
## Summarization Function
"""
logger.info("## Summarization Function")

def generate_page_summary(page_text):
    """
    Generate a concise summary of a page.

    Args:
        page_text (str): Text content of the page

    Returns:
        str: Generated summary
    """
    system_prompt = """You are an expert summarization system.
    Create a detailed summary of the provided text.
    Focus on capturing the main topics, key information, and important facts.
    Your summary should be comprehensive enough to understand what the page contains
    but more concise than the original."""

    max_tokens = 6000
    truncated_text = page_text[:max_tokens] if len(page_text) > max_tokens else page_text

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # Specify the model to use
        messages=[
            {"role": "system", "content": system_prompt},  # System message to guide the assistant
            {"role": "user", "content": f"Please summarize this text:\n\n{truncated_text}"}  # User message with the text to summarize
        ],
        temperature=0.3  # Set the temperature for response generation
    )

    return response.choices[0].message.content

"""
## Hierarchical Document Processing
"""
logger.info("## Hierarchical Document Processing")

def process_document_hierarchically(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    Process a document into hierarchical indices.

    Args:
        pdf_path (str): Path to the PDF file
        chunk_size (int): Size of each detailed chunk
        chunk_overlap (int): Overlap between chunks

    Returns:
        Tuple[SimpleVectorStore, SimpleVectorStore]: Summary and detailed vector stores
    """
    pages = extract_text_from_pdf(pdf_path)

    logger.debug("Generating page summaries...")
    summaries = []
    for i, page in enumerate(pages):
        logger.debug(f"Summarizing page {i+1}/{len(pages)}...")
        summary_text = generate_page_summary(page["text"])

        summary_metadata = page["metadata"].copy()
        summary_metadata.update({"is_summary": True})

        summaries.append({
            "text": summary_text,
            "metadata": summary_metadata
        })

    detailed_chunks = []
    for page in pages:
        page_chunks = chunk_text(
            page["text"],
            page["metadata"],
            chunk_size,
            chunk_overlap
        )
        detailed_chunks.extend(page_chunks)

    logger.debug(f"Created {len(detailed_chunks)} detailed chunks")

    logger.debug("Creating embeddings for summaries...")
    summary_texts = [summary["text"] for summary in summaries]
    summary_embeddings = create_embeddings(summary_texts)

    logger.debug("Creating embeddings for detailed chunks...")
    chunk_texts = [chunk["text"] for chunk in detailed_chunks]
    chunk_embeddings = create_embeddings(chunk_texts)

    summary_store = SimpleVectorStore()
    detailed_store = SimpleVectorStore()

    for i, summary in enumerate(summaries):
        summary_store.add_item(
            text=summary["text"],
            embedding=summary_embeddings[i],
            metadata=summary["metadata"]
        )

    for i, chunk in enumerate(detailed_chunks):
        detailed_store.add_item(
            text=chunk["text"],
            embedding=chunk_embeddings[i],
            metadata=chunk["metadata"]
        )

    logger.debug(f"Created vector stores with {len(summaries)} summaries and {len(detailed_chunks)} chunks")
    return summary_store, detailed_store

"""
## Hierarchical Retrieval
"""
logger.info("## Hierarchical Retrieval")

def retrieve_hierarchically(query, summary_store, detailed_store, k_summaries=3, k_chunks=5):
    """
    Retrieve information using hierarchical indices.

    Args:
        query (str): User query
        summary_store (SimpleVectorStore): Store of document summaries
        detailed_store (SimpleVectorStore): Store of detailed chunks
        k_summaries (int): Number of summaries to retrieve
        k_chunks (int): Number of chunks to retrieve per summary

    Returns:
        List[Dict]: Retrieved chunks with relevance scores
    """
    logger.debug(f"Performing hierarchical retrieval for query: {query}")

    query_embedding = create_embeddings(query)

    summary_results = summary_store.similarity_search(
        query_embedding,
        k=k_summaries
    )

    logger.debug(f"Retrieved {len(summary_results)} relevant summaries")

    relevant_pages = [result["metadata"]["page"] for result in summary_results]

    def page_filter(metadata):
        return metadata["page"] in relevant_pages

    detailed_results = detailed_store.similarity_search(
        query_embedding,
        k=k_chunks * len(relevant_pages),
        filter_func=page_filter
    )

    logger.debug(f"Retrieved {len(detailed_results)} detailed chunks from relevant pages")

    for result in detailed_results:
        page = result["metadata"]["page"]
        matching_summaries = [s for s in summary_results if s["metadata"]["page"] == page]
        if matching_summaries:
            result["summary"] = matching_summaries[0]["text"]

    return detailed_results

"""
## Response Generation with Context
"""
logger.info("## Response Generation with Context")

def generate_response(query, retrieved_chunks):
    """
    Generate a response based on the query and retrieved chunks.

    Args:
        query (str): User query
        retrieved_chunks (List[Dict]): Retrieved chunks from hierarchical search

    Returns:
        str: Generated response
    """
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks):
        page_num = chunk["metadata"]["page"]  # Get the page number from metadata
        context_parts.append(f"[Page {page_num}]: {chunk['text']}")  # Format the chunk text with page number

    context = "\n\n".join(context_parts)

    system_message = """You are a helpful AI assistant answering questions based on the provided context.
Use the information from the context to answer the user's question accurately.
If the context doesn't contain relevant information, acknowledge that.
Include page numbers when referencing specific information."""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # Specify the model to use
        messages=[
            {"role": "system", "content": system_message},  # System message to guide the assistant
            {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {query}"}  # User message with context and query
        ],
        temperature=0.2  # Set the temperature for response generation
    )

    return response.choices[0].message.content

"""
## Complete RAG Pipeline with Hierarchical Retrieval
"""
logger.info("## Complete RAG Pipeline with Hierarchical Retrieval")

def hierarchical_rag(query, pdf_path, chunk_size=1000, chunk_overlap=200,
                    k_summaries=3, k_chunks=5, regenerate=False):
    """
    Complete hierarchical RAG pipeline.

    Args:
        query (str): User query
        pdf_path (str): Path to the PDF document
        chunk_size (int): Size of each detailed chunk
        chunk_overlap (int): Overlap between chunks
        k_summaries (int): Number of summaries to retrieve
        k_chunks (int): Number of chunks to retrieve per summary
        regenerate (bool): Whether to regenerate vector stores

    Returns:
        Dict: Results including response and retrieved chunks
    """
    summary_store_file = f"{os.path.basename(pdf_path)}_summary_store.pkl"
    detailed_store_file = f"{os.path.basename(pdf_path)}_detailed_store.pkl"

    if regenerate or not os.path.exists(summary_store_file) or not os.path.exists(detailed_store_file):
        logger.debug("Processing document and creating vector stores...")
        summary_store, detailed_store = process_document_hierarchically(
            pdf_path, chunk_size, chunk_overlap
        )

        with open(summary_store_file, 'wb') as f:
            pickle.dump(summary_store, f)

        with open(detailed_store_file, 'wb') as f:
            pickle.dump(detailed_store, f)
    else:
        logger.debug("Loading existing vector stores...")
        with open(summary_store_file, 'rb') as f:
            summary_store = pickle.load(f)

        with open(detailed_store_file, 'rb') as f:
            detailed_store = pickle.load(f)

    retrieved_chunks = retrieve_hierarchically(
        query, summary_store, detailed_store, k_summaries, k_chunks
    )

    response = generate_response(query, retrieved_chunks)

    return {
        "query": query,
        "response": response,
        "retrieved_chunks": retrieved_chunks,
        "summary_count": len(summary_store.texts),
        "detailed_count": len(detailed_store.texts)
    }

"""
## Standard (Non-Hierarchical) RAG for Comparison
"""
logger.info("## Standard (Non-Hierarchical) RAG for Comparison")

def standard_rag(query, pdf_path, chunk_size=1000, chunk_overlap=200, k=15):
    """
    Standard RAG pipeline without hierarchical retrieval.

    Args:
        query (str): User query
        pdf_path (str): Path to the PDF document
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        k (int): Number of chunks to retrieve

    Returns:
        Dict: Results including response and retrieved chunks
    """
    pages = extract_text_from_pdf(pdf_path)

    chunks = []
    for page in pages:
        page_chunks = chunk_text(
            page["text"],
            page["metadata"],
            chunk_size,
            chunk_overlap
        )
        chunks.extend(page_chunks)

    logger.debug(f"Created {len(chunks)} chunks for standard RAG")

    store = SimpleVectorStore()

    logger.debug("Creating embeddings for chunks...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings(texts)

    for i, chunk in enumerate(chunks):
        store.add_item(
            text=chunk["text"],
            embedding=embeddings[i],
            metadata=chunk["metadata"]
        )

    query_embedding = create_embeddings(query)

    retrieved_chunks = store.similarity_search(query_embedding, k=k)
    logger.debug(f"Retrieved {len(retrieved_chunks)} chunks with standard RAG")

    response = generate_response(query, retrieved_chunks)

    return {
        "query": query,
        "response": response,
        "retrieved_chunks": retrieved_chunks
    }

"""
## Evaluation Functions
"""
logger.info("## Evaluation Functions")

def compare_approaches(query, pdf_path, reference_answer=None):
    """
    Compare hierarchical and standard RAG approaches.

    Args:
        query (str): User query
        pdf_path (str): Path to the PDF document
        reference_answer (str, optional): Reference answer for evaluation

    Returns:
        Dict: Comparison results
    """
    logger.debug(f"\n=== Comparing RAG approaches for query: {query} ===")

    logger.debug("\nRunning hierarchical RAG...")
    hierarchical_result = hierarchical_rag(query, pdf_path)
    hier_response = hierarchical_result["response"]

    logger.debug("\nRunning standard RAG...")
    standard_result = standard_rag(query, pdf_path)
    std_response = standard_result["response"]

    comparison = compare_responses(query, hier_response, std_response, reference_answer)

    return {
        "query": query,  # The original query
        "hierarchical_response": hier_response,  # Response from hierarchical RAG
        "standard_response": std_response,  # Response from standard RAG
        "reference_answer": reference_answer,  # Reference answer for evaluation
        "comparison": comparison,  # Comparison analysis
        "hierarchical_chunks_count": len(hierarchical_result["retrieved_chunks"]),  # Number of chunks retrieved by hierarchical RAG
        "standard_chunks_count": len(standard_result["retrieved_chunks"])  # Number of chunks retrieved by standard RAG
    }

"""

"""

def compare_responses(query, hierarchical_response, standard_response, reference=None):
    """
    Compare responses from hierarchical and standard RAG.

    Args:
        query (str): User query
        hierarchical_response (str): Response from hierarchical RAG
        standard_response (str): Response from standard RAG
        reference (str, optional): Reference answer

    Returns:
        str: Comparison analysis
    """
    system_prompt = """You are an expert evaluator of information retrieval systems.
Compare the two responses to the same query, one generated using hierarchical retrieval
and the other using standard retrieval.

Evaluate them based on:
1. Accuracy: Which response provides more factually correct information?
2. Comprehensiveness: Which response better covers all aspects of the query?
3. Coherence: Which response has better logical flow and organization?
4. Page References: Does either response make better use of page references?

Be specific in your analysis of the strengths and weaknesses of each approach."""

    user_prompt = f"""Query: {query}

Response from Hierarchical RAG:
{hierarchical_response}

Response from Standard RAG:
{standard_response}"""

    if reference:
        user_prompt += f"""

Reference Answer:
{reference}"""

    user_prompt += """

Please provide a detailed comparison of these two responses, highlighting which approach performed better and why."""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},  # System message to guide the assistant
            {"role": "user", "content": user_prompt}  # User message with the query and responses
        ],
        temperature=0  # Set the temperature for response generation
    )

    return response.choices[0].message.content

"""

"""

def run_evaluation(pdf_path, test_queries, reference_answers=None):
    """
    Run a complete evaluation with multiple test queries.

    Args:
        pdf_path (str): Path to the PDF document
        test_queries (List[str]): List of test queries
        reference_answers (List[str], optional): Reference answers for queries

    Returns:
        Dict: Evaluation results
    """
    results = []  # Initialize an empty list to store results

    for i, query in enumerate(test_queries):
        logger.debug(f"Query: {query}")  # Print the current query

        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]  # Retrieve the reference answer for the current query

        result = compare_approaches(query, pdf_path, reference)
        results.append(result)  # Append the result to the results list

    overall_analysis = generate_overall_analysis(results)

    return {
        "results": results,  # Return the individual results
        "overall_analysis": overall_analysis  # Return the overall analysis
    }

"""

"""

def generate_overall_analysis(results):
    """
    Generate an overall analysis of the evaluation results.

    Args:
        results (List[Dict]): Results from individual query evaluations

    Returns:
        str: Overall analysis
    """
    system_prompt = """You are an expert at evaluating information retrieval systems.
Based on multiple test queries, provide an overall analysis comparing hierarchical RAG
with standard RAG.

Focus on:
1. When hierarchical retrieval performs better and why
2. When standard retrieval performs better and why
3. The overall strengths and weaknesses of each approach
4. Recommendations for when to use each approach"""

    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        evaluations_summary += f"Hierarchical chunks: {result['hierarchical_chunks_count']}, Standard chunks: {result['standard_chunks_count']}\n"
        evaluations_summary += f"Comparison summary: {result['comparison'][:200]}...\n\n"

    user_prompt = f"""Based on the following evaluations comparing hierarchical vs standard RAG across {len(results)} queries,
provide an overall analysis of these two approaches:

{evaluations_summary}

Please provide a comprehensive analysis of the relative strengths and weaknesses of hierarchical RAG
compared to standard RAG, with specific focus on retrieval quality and response generation."""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},  # System message to guide the assistant
            {"role": "user", "content": user_prompt}  # User message with the evaluations summary
        ],
        temperature=0  # Set the temperature for response generation
    )

    return response.choices[0].message.content

"""
## Evaluation of Hierarchical and Standard RAG Approaches
"""
logger.info("## Evaluation of Hierarchical and Standard RAG Approaches")

pdf_path = f"{GENERATED_DIR}/AI_Information.pdf"

query = "What are the key applications of transformer models in natural language processing?"
result = hierarchical_rag(query, pdf_path)

logger.debug("\n=== Response ===")
logger.debug(result["response"])

test_queries = [
    "How do transformers handle sequential data compared to RNNs?"
]

reference_answers = [
    "Transformers handle sequential data differently from RNNs by using self-attention mechanisms instead of recurrent connections. This allows transformers to process all tokens in parallel rather than sequentially, capturing long-range dependencies more efficiently and enabling better parallelization during training. Unlike RNNs, transformers don't suffer from vanishing gradient problems with long sequences."
]

evaluation_results = run_evaluation(
    pdf_path=pdf_path,
    test_queries=test_queries,
    reference_answers=reference_answers
)

logger.debug("\n=== OVERALL ANALYSIS ===")
logger.debug(evaluation_results["overall_analysis"])

logger.info("\n\n[DONE]", bright=True)