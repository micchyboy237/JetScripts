from jet.logger import CustomLogger
from openai import MLX
import fitz
import json
import numpy as np
import os
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
# Relevant Segment Extraction (RSE) for Enhanced RAG

In this notebook, we implement a Relevant Segment Extraction (RSE) technique to improve the context quality in our RAG system. Rather than simply retrieving a collection of isolated chunks, we identify and reconstruct continuous segments of text that provide better context to our language model.

## Key Concept

Relevant chunks tend to be clustered together within documents. By identifying these clusters and preserving their continuity, we provide more coherent context for the LLM to work with.

## Setting Up the Environment
We begin by importing necessary libraries.
"""
logger.info("# Relevant Segment Extraction (RSE) for Enhanced RAG")


"""
## Extracting Text from a PDF File
To implement RAG, we first need a source of textual data. In this case, we extract text from a PDF file using the PyMuPDF library.
"""
logger.info("## Extracting Text from a PDF File")

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file and prints the first `num_chars` characters.

    Args:
    pdf_path (str): Path to the PDF file.

    Returns:
    str: Extracted text from the PDF.
    """
    mypdf = fitz.open(pdf_path)
    all_text = ""  # Initialize an empty string to store the extracted text

    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]  # Get the page
        text = page.get_text("text")  # Extract text from the page
        all_text += text  # Append the extracted text to the all_text string

    return all_text  # Return the extracted text

"""
## Chunking the Extracted Text
Once we have the extracted text, we divide it into smaller, overlapping chunks to improve retrieval accuracy.
"""
logger.info("## Chunking the Extracted Text")

def chunk_text(text, chunk_size=800, overlap=0):
    """
    Split text into non-overlapping chunks.
    For RSE, we typically want non-overlapping chunks so we can reconstruct segments properly.

    Args:
        text (str): Input text to chunk
        chunk_size (int): Size of each chunk in characters
        overlap (int): Overlap between chunks in characters

    Returns:
        List[str]: List of text chunks
    """
    chunks = []

    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk:  # Ensure we don't add empty chunks
            chunks.append(chunk)

    return chunks

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
## Building a Simple Vector Store
let's implement a simple vector store.
"""
logger.info("## Building a Simple Vector Store")

class SimpleVectorStore:
    """
    A lightweight vector store implementation using NumPy.
    """
    def __init__(self, dimension=1536):
        """
        Initialize the vector store.

        Args:
            dimension (int): Dimension of embeddings
        """
        self.dimension = dimension
        self.vectors = []
        self.documents = []
        self.metadata = []

    def add_documents(self, documents, vectors=None, metadata=None):
        """
        Add documents to the vector store.

        Args:
            documents (List[str]): List of document chunks
            vectors (List[List[float]], optional): List of embedding vectors
            metadata (List[Dict], optional): List of metadata dictionaries
        """
        if vectors is None:
            vectors = [None] * len(documents)

        if metadata is None:
            metadata = [{} for _ in range(len(documents))]

        for doc, vec, meta in zip(documents, vectors, metadata):
            self.documents.append(doc)
            self.vectors.append(vec)
            self.metadata.append(meta)

    def search(self, query_vector, top_k=5):
        """
        Search for most similar documents.

        Args:
            query_vector (List[float]): Query embedding vector
            top_k (int): Number of results to return

        Returns:
            List[Dict]: List of results with documents, scores, and metadata
        """
        if not self.vectors or not self.documents:
            return []

        query_array = np.array(query_vector)

        similarities = []
        for i, vector in enumerate(self.vectors):
            if vector is not None:
                similarity = np.dot(query_array, vector) / (
                    np.linalg.norm(query_array) * np.linalg.norm(vector)
                )
                similarities.append((i, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, score in similarities[:top_k]:
            results.append({
                "document": self.documents[i],
                "score": float(score),
                "metadata": self.metadata[i]
            })

        return results

"""
## Creating Embeddings for Text Chunks
Embeddings transform text into numerical vectors, which allow for efficient similarity search.
"""
logger.info("## Creating Embeddings for Text Chunks")

def create_embeddings(texts, model="BAAI/bge-en-icl"):
    """
    Generate embeddings for texts.

    Args:
        texts (List[str]): List of texts to embed
        model (str): Embedding model to use

    Returns:
        List[List[float]]: List of embedding vectors
    """
    if not texts:
        return []  # Return an empty list if no texts are provided

    batch_size = 100  # Adjust based on your API limits
    all_embeddings = []  # Initialize a list to store all embeddings

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]  # Get the current batch of texts

        response = client.embeddings.create(
            input=batch,
            model=model
        )

        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)  # Add the batch embeddings to the list

    return all_embeddings  # Return the list of all embeddings

"""
## Processing Documents with RSE
Now let's implement the core RSE functionality.
"""
logger.info("## Processing Documents with RSE")

def process_document(pdf_path, chunk_size=800):
    """
    Process a document for use with RSE.

    Args:
        pdf_path (str): Path to the PDF document
        chunk_size (int): Size of each chunk in characters

    Returns:
        Tuple[List[str], SimpleVectorStore, Dict]: Chunks, vector store, and document info
    """
    logger.debug("Extracting text from document...")
    text = extract_text_from_pdf(pdf_path)

    logger.debug("Chunking text into non-overlapping segments...")
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=0)
    logger.debug(f"Created {len(chunks)} chunks")

    logger.debug("Generating embeddings for chunks...")
    chunk_embeddings = create_embeddings(chunks)

    vector_store = SimpleVectorStore()

    metadata = [{"chunk_index": i, "source": pdf_path} for i in range(len(chunks))]
    vector_store.add_documents(chunks, chunk_embeddings, metadata)

    doc_info = {
        "chunks": chunks,
        "source": pdf_path,
    }

    return chunks, vector_store, doc_info

"""
## RSE Core Algorithm: Computing Chunk Values and Finding Best Segments
Now that we have the necessary functions to process a document and generate embeddings for its chunks, we can implement the core algorithm for RSE.
"""
logger.info("## RSE Core Algorithm: Computing Chunk Values and Finding Best Segments")

def calculate_chunk_values(query, chunks, vector_store, irrelevant_chunk_penalty=0.2):
    """
    Calculate chunk values by combining relevance and position.

    Args:
        query (str): Query text
        chunks (List[str]): List of document chunks
        vector_store (SimpleVectorStore): Vector store containing the chunks
        irrelevant_chunk_penalty (float): Penalty for irrelevant chunks

    Returns:
        List[float]: List of chunk values
    """
    query_embedding = create_embeddings([query])[0]

    num_chunks = len(chunks)
    results = vector_store.search(query_embedding, top_k=num_chunks)

    relevance_scores = {result["metadata"]["chunk_index"]: result["score"] for result in results}

    chunk_values = []
    for i in range(num_chunks):
        score = relevance_scores.get(i, 0.0)
        value = score - irrelevant_chunk_penalty
        chunk_values.append(value)

    return chunk_values

def find_best_segments(chunk_values, max_segment_length=20, total_max_length=30, min_segment_value=0.2):
    """
    Find the best segments using a variant of the maximum sum subarray algorithm.

    Args:
        chunk_values (List[float]): Values for each chunk
        max_segment_length (int): Maximum length of a single segment
        total_max_length (int): Maximum total length across all segments
        min_segment_value (float): Minimum value for a segment to be considered

    Returns:
        List[Tuple[int, int]]: List of (start, end) indices for best segments
    """
    logger.debug("Finding optimal continuous text segments...")

    best_segments = []
    segment_scores = []
    total_included_chunks = 0

    while total_included_chunks < total_max_length:
        best_score = min_segment_value  # Minimum threshold for a segment
        best_segment = None

        for start in range(len(chunk_values)):
            if any(start >= s[0] and start < s[1] for s in best_segments):
                continue

            for length in range(1, min(max_segment_length, len(chunk_values) - start) + 1):
                end = start + length

                if any(end > s[0] and end <= s[1] for s in best_segments):
                    continue

                segment_value = sum(chunk_values[start:end])

                if segment_value > best_score:
                    best_score = segment_value
                    best_segment = (start, end)

        if best_segment:
            best_segments.append(best_segment)
            segment_scores.append(best_score)
            total_included_chunks += best_segment[1] - best_segment[0]
            logger.debug(f"Found segment {best_segment} with score {best_score:.4f}")
        else:
            break

    best_segments = sorted(best_segments, key=lambda x: x[0])

    return best_segments, segment_scores

"""
## Reconstructing and Using Segments for RAG
"""
logger.info("## Reconstructing and Using Segments for RAG")

def reconstruct_segments(chunks, best_segments):
    """
    Reconstruct text segments based on chunk indices.

    Args:
        chunks (List[str]): List of all document chunks
        best_segments (List[Tuple[int, int]]): List of (start, end) indices for segments

    Returns:
        List[str]: List of reconstructed text segments
    """
    reconstructed_segments = []  # Initialize an empty list to store the reconstructed segments

    for start, end in best_segments:
        segment_text = " ".join(chunks[start:end])
        reconstructed_segments.append({
            "text": segment_text,
            "segment_range": (start, end),
        })

    return reconstructed_segments  # Return the list of reconstructed text segments

def format_segments_for_context(segments):
    """
    Format segments into a context string for the LLM.

    Args:
        segments (List[Dict]): List of segment dictionaries

    Returns:
        str: Formatted context text
    """
    context = []  # Initialize an empty list to store the formatted context

    for i, segment in enumerate(segments):
        segment_header = f"SEGMENT {i+1} (Chunks {segment['segment_range'][0]}-{segment['segment_range'][1]-1}):"
        context.append(segment_header)  # Add the segment header to the context list
        context.append(segment['text'])  # Add the segment text to the context list
        context.append("-" * 80)  # Add a separator line for readability

    return "\n\n".join(context)

"""
## Generating Responses with RSE Context
"""
logger.info("## Generating Responses with RSE Context")

def generate_response(query, context, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Generate a response based on the query and context.

    Args:
        query (str): User query
        context (str): Context text from relevant segments
        model (str): LLM model to use

    Returns:
        str: Generated response
    """
    logger.debug("Generating response using relevant segments as context...")

    system_prompt = """You are a helpful assistant that answers questions based on the provided context.
    The context consists of document segments that have been retrieved as relevant to the user's query.
    Use the information from these segments to provide a comprehensive and accurate answer.
    If the context doesn't contain relevant information to answer the question, say so clearly."""

    user_prompt = f"""
Context:
{context}

Question: {query}

Please provide a helpful answer based on the context provided.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content

"""
## Complete RSE Pipeline Function
"""
logger.info("## Complete RSE Pipeline Function")

def rag_with_rse(pdf_path, query, chunk_size=800, irrelevant_chunk_penalty=0.2):
    """
    Complete RAG pipeline with Relevant Segment Extraction.

    Args:
        pdf_path (str): Path to the document
        query (str): User query
        chunk_size (int): Size of chunks
        irrelevant_chunk_penalty (float): Penalty for irrelevant chunks

    Returns:
        Dict: Result with query, segments, and response
    """
    logger.debug("\n=== STARTING RAG WITH RELEVANT SEGMENT EXTRACTION ===")
    logger.debug(f"Query: {query}")

    chunks, vector_store, doc_info = process_document(pdf_path, chunk_size)

    logger.debug("\nCalculating relevance scores and chunk values...")
    chunk_values = calculate_chunk_values(query, chunks, vector_store, irrelevant_chunk_penalty)

    best_segments, scores = find_best_segments(
        chunk_values,
        max_segment_length=20,
        total_max_length=30,
        min_segment_value=0.2
    )

    logger.debug("\nReconstructing text segments from chunks...")
    segments = reconstruct_segments(chunks, best_segments)

    context = format_segments_for_context(segments)

    response = generate_response(query, context)

    result = {
        "query": query,
        "segments": segments,
        "response": response
    }

    logger.debug("\n=== FINAL RESPONSE ===")
    logger.debug(response)

    return result

"""
## Comparing with Standard Retrieval
Let's implement a standard retrieval approach to compare with RSE:
"""
logger.info("## Comparing with Standard Retrieval")

def standard_top_k_retrieval(pdf_path, query, k=10, chunk_size=800):
    """
    Standard RAG with top-k retrieval.

    Args:
        pdf_path (str): Path to the document
        query (str): User query
        k (int): Number of chunks to retrieve
        chunk_size (int): Size of chunks

    Returns:
        Dict: Result with query, chunks, and response
    """
    logger.debug("\n=== STARTING STANDARD TOP-K RETRIEVAL ===")
    logger.debug(f"Query: {query}")

    chunks, vector_store, doc_info = process_document(pdf_path, chunk_size)

    logger.debug("Creating query embedding and retrieving chunks...")
    query_embedding = create_embeddings([query])[0]

    results = vector_store.search(query_embedding, top_k=k)
    retrieved_chunks = [result["document"] for result in results]

    context = "\n\n".join([
        f"CHUNK {i+1}:\n{chunk}"
        for i, chunk in enumerate(retrieved_chunks)
    ])

    response = generate_response(query, context)

    result = {
        "query": query,
        "chunks": retrieved_chunks,
        "response": response
    }

    logger.debug("\n=== FINAL RESPONSE ===")
    logger.debug(response)

    return result

"""
## Evaluation of RSE
"""
logger.info("## Evaluation of RSE")

def evaluate_methods(pdf_path, query, reference_answer=None):
    """
    Compare RSE with standard top-k retrieval.

    Args:
        pdf_path (str): Path to the document
        query (str): User query
        reference_answer (str, optional): Reference answer for evaluation
    """
    logger.debug("\n========= EVALUATION =========\n")

    rse_result = rag_with_rse(pdf_path, query)

    standard_result = standard_top_k_retrieval(pdf_path, query)

    if reference_answer:
        logger.debug("\n=== COMPARING RESULTS ===")

        evaluation_prompt = f"""
            Query: {query}

            Reference Answer:
            {reference_answer}

            Response from Standard Retrieval:
            {standard_result["response"]}

            Response from Relevant Segment Extraction:
            {rse_result["response"]}

            Compare these two responses against the reference answer. Which one is:
            1. More accurate and comprehensive
            2. Better at addressing the user's query
            3. Less likely to include irrelevant information

            Explain your reasoning for each point.
        """

        logger.debug("Evaluating responses against reference answer...")

        evaluation = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[
                {"role": "system", "content": "You are an objective evaluator of RAG system responses."},
                {"role": "user", "content": evaluation_prompt}
            ]
        )

        logger.debug("\n=== EVALUATION RESULTS ===")
        logger.debug(evaluation.choices[0].message.content)

    return {
        "rse_result": rse_result,
        "standard_result": standard_result
    }

with open('data/val.json') as f:
    data = json.load(f)

query = data[0]['question']

reference_answer = data[0]['ideal_answer']

pdf_path = f"{GENERATED_DIR}/AI_Information.pdf"

results = evaluate_methods(pdf_path, query, reference_answer)

logger.info("\n\n[DONE]", bright=True)