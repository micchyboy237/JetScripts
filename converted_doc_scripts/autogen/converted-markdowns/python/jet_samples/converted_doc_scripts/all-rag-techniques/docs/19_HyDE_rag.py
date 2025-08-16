from jet.logger import CustomLogger
from openai import Ollama
import fitz
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

"""
# Hypothetical Document Embedding (HyDE) for RAG

In this notebook, I implement HyDE (Hypothetical Document Embedding) - an innovative retrieval technique that transforms user queries into hypothetical answer documents before performing retrieval. This approach bridges the semantic gap between short queries and lengthy documents.

Traditional RAG systems embed the user's short query directly, but this often fails to capture the semantic richness needed for optimal retrieval. HyDE solves this by:

- Generating a hypothetical document that answers the query
- Embedding this expanded document instead of the original query
- Retrieving documents similar to this hypothetical document
- Creating more contextually relevant answers

## Setting Up the Environment
We begin by importing necessary libraries.
"""
logger.info("# Hypothetical Document Embedding (HyDE) for RAG")


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

def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Split text into overlapping chunks.

    Args:
        text (str): Input text to chunk
        chunk_size (int): Size of each chunk in characters
        overlap (int): Overlap between chunks in characters

    Returns:
        List[Dict]: List of chunks with metadata
    """
    chunks = []  # Initialize an empty list to store the chunks

    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]  # Extract the chunk of text
        if chunk_text:  # Ensure we don't add empty chunks
            chunks.append({
                "text": chunk_text,  # Add the chunk text
                "metadata": {
                    "start_pos": i,  # Start position of the chunk in the original text
                    "end_pos": i + len(chunk_text)  # End position of the chunk in the original text
                }
            })

    logger.debug(f"Created {len(chunks)} text chunks")  # Print the number of chunks created
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
## Document Processing Pipeline
"""
logger.info("## Document Processing Pipeline")

def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    Process a document for RAG.

    Args:
        pdf_path (str): Path to the PDF file
        chunk_size (int): Size of each chunk in characters
        chunk_overlap (int): Overlap between chunks in characters

    Returns:
        SimpleVectorStore: Vector store containing document chunks
    """
    pages = extract_text_from_pdf(pdf_path)

    all_chunks = []
    for page in pages:
        page_chunks = chunk_text(page["text"], chunk_size, chunk_overlap)

        for chunk in page_chunks:
            chunk["metadata"].update(page["metadata"])

        all_chunks.extend(page_chunks)

    logger.debug("Creating embeddings for chunks...")
    chunk_texts = [chunk["text"] for chunk in all_chunks]
    chunk_embeddings = create_embeddings(chunk_texts)

    vector_store = SimpleVectorStore()
    for i, chunk in enumerate(all_chunks):
        vector_store.add_item(
            text=chunk["text"],
            embedding=chunk_embeddings[i],
            metadata=chunk["metadata"]
        )

    logger.debug(f"Vector store created with {len(all_chunks)} chunks")
    return vector_store

"""
## Hypothetical Document Generation
"""
logger.info("## Hypothetical Document Generation")

def generate_hypothetical_document(query, desired_length=1000):
    """
    Generate a hypothetical document that answers the query.

    Args:
        query (str): User query
        desired_length (int): Target length of the hypothetical document

    Returns:
        str: Generated hypothetical document
    """
    system_prompt = f"""You are an expert document creator.
    Given a question, generate a detailed document that would directly answer this question.
    The document should be approximately {desired_length} characters long and provide an in-depth,
    informative answer to the question. Write as if this document is from an authoritative source
    on the subject. Include specific details, facts, and explanations.
    Do not mention that this is a hypothetical document - just write the content directly."""

    user_prompt = f"Question: {query}\n\nGenerate a document that fully answers this question:"

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # Specify the model to use
        messages=[
            {"role": "system", "content": system_prompt},  # System message to guide the assistant
            {"role": "user", "content": user_prompt}  # User message with the query
        ],
        temperature=0.1  # Set the temperature for response generation
    )

    return response.choices[0].message.content

"""
## Complete HyDE RAG Implementation
"""
logger.info("## Complete HyDE RAG Implementation")

def hyde_rag(query, vector_store, k=5, should_generate_response=True):
    """
    Perform RAG using Hypothetical Document Embedding.

    Args:
        query (str): User query
        vector_store (SimpleVectorStore): Vector store with document chunks
        k (int): Number of chunks to retrieve
        generate_response (bool): Whether to generate a final response

    Returns:
        Dict: Results including hypothetical document and retrieved chunks
    """
    logger.debug(f"\n=== Processing query with HyDE: {query} ===\n")

    logger.debug("Generating hypothetical document...")
    hypothetical_doc = generate_hypothetical_document(query)
    logger.debug(f"Generated hypothetical document of {len(hypothetical_doc)} characters")

    logger.debug("Creating embedding for hypothetical document...")
    hypothetical_embedding = create_embeddings([hypothetical_doc])[0]

    logger.debug(f"Retrieving {k} most similar chunks...")
    retrieved_chunks = vector_store.similarity_search(hypothetical_embedding, k=k)

    results = {
        "query": query,
        "hypothetical_document": hypothetical_doc,
        "retrieved_chunks": retrieved_chunks
    }

    if should_generate_response:
        logger.debug("Generating final response...")
        response = generate_response(query, retrieved_chunks)
        results["response"] = response

    return results

"""
## Standard (Direct) RAG Implementation for Comparison
"""
logger.info("## Standard (Direct) RAG Implementation for Comparison")

def standard_rag(query, vector_store, k=5, should_generate_response=True):
    """
    Perform standard RAG using direct query embedding.

    Args:
        query (str): User query
        vector_store (SimpleVectorStore): Vector store with document chunks
        k (int): Number of chunks to retrieve
        generate_response (bool): Whether to generate a final response

    Returns:
        Dict: Results including retrieved chunks
    """
    logger.debug(f"\n=== Processing query with Standard RAG: {query} ===\n")

    logger.debug("Creating embedding for query...")
    query_embedding = create_embeddings([query])[0]

    logger.debug(f"Retrieving {k} most similar chunks...")
    retrieved_chunks = vector_store.similarity_search(query_embedding, k=k)

    results = {
        "query": query,
        "retrieved_chunks": retrieved_chunks
    }

    if should_generate_response:
        logger.debug("Generating final response...")
        response = generate_response(query, retrieved_chunks)
        results["response"] = response

    return results

"""
## Response Generation
"""
logger.info("## Response Generation")

def generate_response(query, relevant_chunks):
    """
    Generate a final response based on the query and relevant chunks.

    Args:
        query (str): User query
        relevant_chunks (List[Dict]): Retrieved relevant chunks

    Returns:
        str: Generated response
    """
    context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer the question based on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.5,
        max_tokens=500
    )

    return response.choices[0].message.content

"""
## Evaluation Functions
"""
logger.info("## Evaluation Functions")

def compare_approaches(query, vector_store, reference_answer=None):
    """
    Compare HyDE and standard RAG approaches for a query.

    Args:
        query (str): User query
        vector_store (SimpleVectorStore): Vector store with document chunks
        reference_answer (str, optional): Reference answer for evaluation

    Returns:
        Dict: Comparison results
    """
    hyde_result = hyde_rag(query, vector_store)
    hyde_response = hyde_result["response"]

    standard_result = standard_rag(query, vector_store)
    standard_response = standard_result["response"]

    comparison = compare_responses(query, hyde_response, standard_response, reference_answer)

    return {
        "query": query,
        "hyde_response": hyde_response,
        "hyde_hypothetical_doc": hyde_result["hypothetical_document"],
        "standard_response": standard_response,
        "reference_answer": reference_answer,
        "comparison": comparison
    }

"""

"""

def compare_responses(query, hyde_response, standard_response, reference=None):
    """
    Compare responses from HyDE and standard RAG.

    Args:
        query (str): User query
        hyde_response (str): Response from HyDE RAG
        standard_response (str): Response from standard RAG
        reference (str, optional): Reference answer

    Returns:
        str: Comparison analysis
    """
    system_prompt = """You are an expert evaluator of information retrieval systems.
Compare the two responses to the same query, one generated using HyDE (Hypothetical Document Embedding)
and the other using standard RAG with direct query embedding.

Evaluate them based on:
1. Accuracy: Which response provides more factually correct information?
2. Relevance: Which response better addresses the query?
3. Completeness: Which response provides more thorough coverage of the topic?
4. Clarity: Which response is better organized and easier to understand?

Be specific about the strengths and weaknesses of each approach."""

    user_prompt = f"""Query: {query}

Response from HyDE RAG:
{hyde_response}

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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content

"""

"""

def run_evaluation(pdf_path, test_queries, reference_answers=None, chunk_size=1000, chunk_overlap=200):
    """
    Run a complete evaluation with multiple test queries.

    Args:
        pdf_path (str): Path to the PDF document
        test_queries (List[str]): List of test queries
        reference_answers (List[str], optional): Reference answers for queries
        chunk_size (int): Size of each chunk in characters
        chunk_overlap (int): Overlap between chunks in characters

    Returns:
        Dict: Evaluation results
    """
    vector_store = process_document(pdf_path, chunk_size, chunk_overlap)

    results = []

    for i, query in enumerate(test_queries):
        logger.debug(f"\n\n===== Evaluating Query {i+1}/{len(test_queries)} =====")
        logger.debug(f"Query: {query}")

        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]

        result = compare_approaches(query, vector_store, reference)
        results.append(result)

    overall_analysis = generate_overall_analysis(results)

    return {
        "results": results,
        "overall_analysis": overall_analysis
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
Based on multiple test queries, provide an overall analysis comparing HyDE RAG (using hypothetical document embedding)
with standard RAG (using direct query embedding).

Focus on:
1. When HyDE performs better and why
2. When standard RAG performs better and why
3. The types of queries that benefit most from HyDE
4. The overall strengths and weaknesses of each approach
5. Recommendations for when to use each approach"""

    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        evaluations_summary += f"Comparison summary: {result['comparison'][:200]}...\n\n"

    user_prompt = f"""Based on the following evaluations comparing HyDE vs standard RAG across {len(results)} queries,
provide an overall analysis of these two approaches:

{evaluations_summary}

Please provide a comprehensive analysis of the relative strengths and weaknesses of HyDE compared to standard RAG,
focusing on when and why one approach outperforms the other."""

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
## Visualization Functions
"""
logger.info("## Visualization Functions")

def visualize_results(query, hyde_result, standard_result):
    """
    Visualize the results of HyDE and standard RAG approaches.

    Args:
        query (str): User query
        hyde_result (Dict): Results from HyDE RAG
        standard_result (Dict): Results from standard RAG
    """
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    axs[0].text(0.5, 0.5, f"Query:\n\n{query}",
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, wrap=True)
    axs[0].axis('off')  # Hide the axis for the query plot

    hypothetical_doc = hyde_result["hypothetical_document"]
    shortened_doc = hypothetical_doc[:500] + "..." if len(hypothetical_doc) > 500 else hypothetical_doc
    axs[1].text(0.5, 0.5, f"Hypothetical Document:\n\n{shortened_doc}",
                horizontalalignment='center', verticalalignment='center',
                fontsize=10, wrap=True)
    axs[1].axis('off')  # Hide the axis for the hypothetical document plot

    hyde_chunks = [chunk["text"][:100] + "..." for chunk in hyde_result["retrieved_chunks"]]
    std_chunks = [chunk["text"][:100] + "..." for chunk in standard_result["retrieved_chunks"]]

    comparison_text = "Retrieved by HyDE:\n\n"
    for i, chunk in enumerate(hyde_chunks):
        comparison_text += f"{i+1}. {chunk}\n\n"

    comparison_text += "\nRetrieved by Standard RAG:\n\n"
    for i, chunk in enumerate(std_chunks):
        comparison_text += f"{i+1}. {chunk}\n\n"

    axs[2].text(0.5, 0.5, comparison_text,
                horizontalalignment='center', verticalalignment='center',
                fontsize=8, wrap=True)
    axs[2].axis('off')  # Hide the axis for the comparison plot

    plt.tight_layout()
    plt.show()

"""
## Evaluation of Hypothetical Document Embedding (HyDE) vs. Standard RAG
"""
logger.info("## Evaluation of Hypothetical Document Embedding (HyDE) vs. Standard RAG")

pdf_path = f"{GENERATED_DIR}/AI_Information.pdf"

vector_store = process_document(pdf_path)

query = "What are the main ethical considerations in artificial intelligence development?"

hyde_result = hyde_rag(query, vector_store)
logger.debug("\n=== HyDE Response ===")
logger.debug(hyde_result["response"])

standard_result = standard_rag(query, vector_store)
logger.debug("\n=== Standard RAG Response ===")
logger.debug(standard_result["response"])

visualize_results(query, hyde_result, standard_result)

test_queries = [
    "How does neural network architecture impact AI performance?"
]

reference_answers = [
    "Neural network architecture significantly impacts AI performance through factors like depth (number of layers), width (neurons per layer), connectivity patterns, and activation functions. Different architectures like CNNs, RNNs, and Transformers are optimized for specific tasks such as image recognition, sequence processing, and natural language understanding respectively.",
]

evaluation_results = run_evaluation(
    pdf_path=pdf_path,
    test_queries=test_queries,
    reference_answers=reference_answers
)

logger.debug("\n=== OVERALL ANALYSIS ===")
logger.debug(evaluation_results["overall_analysis"])

logger.info("\n\n[DONE]", bright=True)