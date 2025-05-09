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

```python
import os
import numpy as np
import json
import fitz
from openai import OpenAI
import re
import pickle
```

## Setting Up the OpenAI API Client
We initialize the OpenAI client to generate embeddings and responses.

```python
# Initialize the OpenAI client with the base URL and API key
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")  # Retrieve the API key from environment variables
)
```

## Document Processing Functions

```python
def extract_text_from_pdf(pdf_path):
    """
    Extract text content from a PDF file with page separation.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        List[Dict]: List of pages with text content and metadata
    """
    print(f"Extracting text from {pdf_path}...")  # Print the path of the PDF being processed
    pdf = fitz.open(pdf_path)  # Open the PDF file using PyMuPDF
    pages = []  # Initialize an empty list to store the pages with text content

    # Iterate over each page in the PDF
    for page_num in range(len(pdf)):
        page = pdf[page_num]  # Get the current page
        text = page.get_text()  # Extract text from the current page

        # Skip pages with very little text (less than 50 characters)
        if len(text.strip()) > 50:
            # Append the page text and metadata to the list
            pages.append({
                "text": text,
                "metadata": {
                    "source": pdf_path,  # Source file path
                    "page": page_num + 1  # Page number (1-based index)
                }
            })

    print(f"Extracted {len(pages)} pages with content")  # Print the number of pages extracted
    return pages  # Return the list of pages with text content and metadata
```

```python
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

    # Iterate over the text with the specified chunk size and overlap
    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]  # Extract the chunk of text

        # Skip very small chunks (less than 50 characters)
        if chunk_text and len(chunk_text.strip()) > 50:
            # Create a copy of metadata and add chunk-specific info
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": len(chunks),  # Index of the chunk
                "start_char": i,  # Start character index of the chunk
                "end_char": i + len(chunk_text),  # End character index of the chunk
                "is_summary": False  # Flag indicating this is not a summary
            })

            # Append the chunk with its metadata to the list
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })

    return chunks  # Return the list of chunks with metadata
```

## Simple Vector Store Implementation

```python
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

        # Convert query embedding to numpy array
        query_vector = np.array(query_embedding)

        # Calculate similarities using cosine similarity
        similarities = []
        for i, vector in enumerate(self.vectors):
            # Skip if doesn't pass the filter
            if filter_func and not filter_func(self.metadata[i]):
                continue

            # Calculate cosine similarity
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))  # Append index and similarity score

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top k results
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],  # Add the text content
                "metadata": self.metadata[idx],  # Add the metadata
                "similarity": float(score)  # Add the similarity score
            })

        return results  # Return the list of top k results
```

## Creating Embeddings

```python
def create_embeddings(texts, model="BAAI/bge-en-icl"):
    """
    Create embeddings for the given texts.

    Args:
        texts (List[str]): Input texts
        model (str): Embedding model name

    Returns:
        List[List[float]]: Embedding vectors
    """
    # Handle empty input
    if not texts:
        return []

    # Process in batches if needed (OpenAI API limits)
    batch_size = 100
    all_embeddings = []

    # Iterate over the input texts in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]  # Get the current batch of texts

        # Create embeddings for the current batch
        response = client.embeddings.create(
            model=model,
            input=batch
        )

        # Extract embeddings from the response
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)  # Add the batch embeddings to the list

    return all_embeddings  # Return all embeddings
```

## Summarization Function

```python
def generate_page_summary(page_text):
    """
    Generate a concise summary of a page.

    Args:
        page_text (str): Text content of the page

    Returns:
        str: Generated summary
    """
    # Define the system prompt to instruct the summarization model
    system_prompt = """You are an expert summarization system.
    Create a detailed summary of the provided text.
    Focus on capturing the main topics, key information, and important facts.
    Your summary should be comprehensive enough to understand what the page contains
    but more concise than the original."""

    # Truncate input text if it exceeds the maximum token limit
    max_tokens = 6000
    truncated_text = page_text[:max_tokens] if len(page_text) > max_tokens else page_text

    # Make a request to the OpenAI API to generate the summary
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # Specify the model to use
        messages=[
            {"role": "system", "content": system_prompt},  # System message to guide the assistant
            {"role": "user", "content": f"Please summarize this text:\n\n{truncated_text}"}  # User message with the text to summarize
        ],
        temperature=0.3  # Set the temperature for response generation
    )

    # Return the generated summary content
    return response.choices[0].message.content
```

## Hierarchical Document Processing

```python
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
    # Extract pages from PDF
    pages = extract_text_from_pdf(pdf_path)

    # Create summaries for each page
    print("Generating page summaries...")
    summaries = []
    for i, page in enumerate(pages):
        print(f"Summarizing page {i+1}/{len(pages)}...")
        summary_text = generate_page_summary(page["text"])

        # Create summary metadata
        summary_metadata = page["metadata"].copy()
        summary_metadata.update({"is_summary": True})

        # Append the summary text and metadata to the summaries list
        summaries.append({
            "text": summary_text,
            "metadata": summary_metadata
        })

    # Create detailed chunks for each page
    detailed_chunks = []
    for page in pages:
        # Chunk the text of the page
        page_chunks = chunk_text(
            page["text"],
            page["metadata"],
            chunk_size,
            chunk_overlap
        )
        # Extend the detailed_chunks list with the chunks from the current page
        detailed_chunks.extend(page_chunks)

    print(f"Created {len(detailed_chunks)} detailed chunks")

    # Create embeddings for summaries
    print("Creating embeddings for summaries...")
    summary_texts = [summary["text"] for summary in summaries]
    summary_embeddings = create_embeddings(summary_texts)

    # Create embeddings for detailed chunks
    print("Creating embeddings for detailed chunks...")
    chunk_texts = [chunk["text"] for chunk in detailed_chunks]
    chunk_embeddings = create_embeddings(chunk_texts)

    # Create vector stores
    summary_store = SimpleVectorStore()
    detailed_store = SimpleVectorStore()

    # Add summaries to summary store
    for i, summary in enumerate(summaries):
        summary_store.add_item(
            text=summary["text"],
            embedding=summary_embeddings[i],
            metadata=summary["metadata"]
        )

    # Add chunks to detailed store
    for i, chunk in enumerate(detailed_chunks):
        detailed_store.add_item(
            text=chunk["text"],
            embedding=chunk_embeddings[i],
            metadata=chunk["metadata"]
        )

    print(f"Created vector stores with {len(summaries)} summaries and {len(detailed_chunks)} chunks")
    return summary_store, detailed_store
```

## Hierarchical Retrieval

```python
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
    print(f"Performing hierarchical retrieval for query: {query}")

    # Create query embedding
    query_embedding = create_embeddings(query)

    # First, retrieve relevant summaries
    summary_results = summary_store.similarity_search(
        query_embedding,
        k=k_summaries
    )

    print(f"Retrieved {len(summary_results)} relevant summaries")

    # Collect pages from relevant summaries
    relevant_pages = [result["metadata"]["page"] for result in summary_results]

    # Create a filter function to only keep chunks from relevant pages
    def page_filter(metadata):
        return metadata["page"] in relevant_pages

    # Then, retrieve detailed chunks from only those relevant pages
    detailed_results = detailed_store.similarity_search(
        query_embedding,
        k=k_chunks * len(relevant_pages),
        filter_func=page_filter
    )

    print(f"Retrieved {len(detailed_results)} detailed chunks from relevant pages")

    # For each result, add which summary/page it came from
    for result in detailed_results:
        page = result["metadata"]["page"]
        matching_summaries = [s for s in summary_results if s["metadata"]["page"] == page]
        if matching_summaries:
            result["summary"] = matching_summaries[0]["text"]

    return detailed_results
```

## Response Generation with Context

```python
def generate_response(query, retrieved_chunks):
    """
    Generate a response based on the query and retrieved chunks.

    Args:
        query (str): User query
        retrieved_chunks (List[Dict]): Retrieved chunks from hierarchical search

    Returns:
        str: Generated response
    """
    # Extract text from chunks and prepare context parts
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks):
        page_num = chunk["metadata"]["page"]  # Get the page number from metadata
        context_parts.append(f"[Page {page_num}]: {chunk['text']}")  # Format the chunk text with page number

    # Combine all context parts into a single context string
    context = "\n\n".join(context_parts)

    # Define the system message to guide the AI assistant
    system_message = """You are a helpful AI assistant answering questions based on the provided context.
Use the information from the context to answer the user's question accurately.
If the context doesn't contain relevant information, acknowledge that.
Include page numbers when referencing specific information."""

    # Generate the response using the OpenAI API
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # Specify the model to use
        messages=[
            {"role": "system", "content": system_message},  # System message to guide the assistant
            {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {query}"}  # User message with context and query
        ],
        temperature=0.2  # Set the temperature for response generation
    )

    # Return the generated response content
    return response.choices[0].message.content
```

## Complete RAG Pipeline with Hierarchical Retrieval

```python
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
    # Create store filenames for caching
    summary_store_file = f"{os.path.basename(pdf_path)}_summary_store.pkl"
    detailed_store_file = f"{os.path.basename(pdf_path)}_detailed_store.pkl"

    # Process document and create stores if needed
    if regenerate or not os.path.exists(summary_store_file) or not os.path.exists(detailed_store_file):
        print("Processing document and creating vector stores...")
        # Process the document to create hierarchical indices and vector stores
        summary_store, detailed_store = process_document_hierarchically(
            pdf_path, chunk_size, chunk_overlap
        )

        # Save the summary store to a file for future use
        with open(summary_store_file, 'wb') as f:
            pickle.dump(summary_store, f)

        # Save the detailed store to a file for future use
        with open(detailed_store_file, 'wb') as f:
            pickle.dump(detailed_store, f)
    else:
        # Load existing summary store from file
        print("Loading existing vector stores...")
        with open(summary_store_file, 'rb') as f:
            summary_store = pickle.load(f)

        # Load existing detailed store from file
        with open(detailed_store_file, 'rb') as f:
            detailed_store = pickle.load(f)

    # Retrieve relevant chunks hierarchically using the query
    retrieved_chunks = retrieve_hierarchically(
        query, summary_store, detailed_store, k_summaries, k_chunks
    )

    # Generate a response based on the retrieved chunks
    response = generate_response(query, retrieved_chunks)

    # Return results including the query, response, retrieved chunks, and counts of summaries and detailed chunks
    return {
        "query": query,
        "response": response,
        "retrieved_chunks": retrieved_chunks,
        "summary_count": len(summary_store.texts),
        "detailed_count": len(detailed_store.texts)
    }
```

## Standard (Non-Hierarchical) RAG for Comparison

```python
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
    # Extract pages from the PDF document
    pages = extract_text_from_pdf(pdf_path)

    # Create chunks directly from all pages
    chunks = []
    for page in pages:
        # Chunk the text of the page
        page_chunks = chunk_text(
            page["text"],
            page["metadata"],
            chunk_size,
            chunk_overlap
        )
        # Extend the chunks list with the chunks from the current page
        chunks.extend(page_chunks)

    print(f"Created {len(chunks)} chunks for standard RAG")

    # Create a vector store to hold the chunks
    store = SimpleVectorStore()

    # Create embeddings for the chunks
    print("Creating embeddings for chunks...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings(texts)

    # Add chunks to the vector store
    for i, chunk in enumerate(chunks):
        store.add_item(
            text=chunk["text"],
            embedding=embeddings[i],
            metadata=chunk["metadata"]
        )

    # Create an embedding for the query
    query_embedding = create_embeddings(query)

    # Retrieve the most relevant chunks based on the query embedding
    retrieved_chunks = store.similarity_search(query_embedding, k=k)
    print(f"Retrieved {len(retrieved_chunks)} chunks with standard RAG")

    # Generate a response based on the retrieved chunks
    response = generate_response(query, retrieved_chunks)

    # Return the results including the query, response, and retrieved chunks
    return {
        "query": query,
        "response": response,
        "retrieved_chunks": retrieved_chunks
    }
```

## Evaluation Functions

```python
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
    print(f"\n=== Comparing RAG approaches for query: {query} ===")

    # Run hierarchical RAG
    print("\nRunning hierarchical RAG...")
    hierarchical_result = hierarchical_rag(query, pdf_path)
    hier_response = hierarchical_result["response"]

    # Run standard RAG
    print("\nRunning standard RAG...")
    standard_result = standard_rag(query, pdf_path)
    std_response = standard_result["response"]

    # Compare results from hierarchical and standard RAG
    comparison = compare_responses(query, hier_response, std_response, reference_answer)

    # Return a dictionary with the comparison results
    return {
        "query": query,  # The original query
        "hierarchical_response": hier_response,  # Response from hierarchical RAG
        "standard_response": std_response,  # Response from standard RAG
        "reference_answer": reference_answer,  # Reference answer for evaluation
        "comparison": comparison,  # Comparison analysis
        "hierarchical_chunks_count": len(hierarchical_result["retrieved_chunks"]),  # Number of chunks retrieved by hierarchical RAG
        "standard_chunks_count": len(standard_result["retrieved_chunks"])  # Number of chunks retrieved by standard RAG
    }
```

```python
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
    # Define the system prompt to instruct the model on how to evaluate the responses
    system_prompt = """You are an expert evaluator of information retrieval systems.
Compare the two responses to the same query, one generated using hierarchical retrieval
and the other using standard retrieval.

Evaluate them based on:
1. Accuracy: Which response provides more factually correct information?
2. Comprehensiveness: Which response better covers all aspects of the query?
3. Coherence: Which response has better logical flow and organization?
4. Page References: Does either response make better use of page references?

Be specific in your analysis of the strengths and weaknesses of each approach."""

    # Create the user prompt with the query and both responses
    user_prompt = f"""Query: {query}

Response from Hierarchical RAG:
{hierarchical_response}

Response from Standard RAG:
{standard_response}"""

    # If a reference answer is provided, include it in the user prompt
    if reference:
        user_prompt += f"""

Reference Answer:
{reference}"""

    # Add the final instruction to the user prompt
    user_prompt += """

Please provide a detailed comparison of these two responses, highlighting which approach performed better and why."""

    # Make a request to the OpenAI API to generate the comparison analysis
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},  # System message to guide the assistant
            {"role": "user", "content": user_prompt}  # User message with the query and responses
        ],
        temperature=0  # Set the temperature for response generation
    )

    # Return the generated comparison analysis
    return response.choices[0].message.content
```

```python
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

    # Iterate over each query in the test queries
    for i, query in enumerate(test_queries):
        print(f"Query: {query}")  # Print the current query

        # Get reference answer if available
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]  # Retrieve the reference answer for the current query

        # Compare hierarchical and standard RAG approaches
        result = compare_approaches(query, pdf_path, reference)
        results.append(result)  # Append the result to the results list

    # Generate overall analysis of the evaluation results
    overall_analysis = generate_overall_analysis(results)

    return {
        "results": results,  # Return the individual results
        "overall_analysis": overall_analysis  # Return the overall analysis
    }
```

```python
def generate_overall_analysis(results):
    """
    Generate an overall analysis of the evaluation results.

    Args:
        results (List[Dict]): Results from individual query evaluations

    Returns:
        str: Overall analysis
    """
    # Define the system prompt to instruct the model on how to evaluate the results
    system_prompt = """You are an expert at evaluating information retrieval systems.
Based on multiple test queries, provide an overall analysis comparing hierarchical RAG
with standard RAG.

Focus on:
1. When hierarchical retrieval performs better and why
2. When standard retrieval performs better and why
3. The overall strengths and weaknesses of each approach
4. Recommendations for when to use each approach"""

    # Create a summary of the evaluations
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        evaluations_summary += f"Hierarchical chunks: {result['hierarchical_chunks_count']}, Standard chunks: {result['standard_chunks_count']}\n"
        evaluations_summary += f"Comparison summary: {result['comparison'][:200]}...\n\n"

    # Define the user prompt with the evaluations summary
    user_prompt = f"""Based on the following evaluations comparing hierarchical vs standard RAG across {len(results)} queries,
provide an overall analysis of these two approaches:

{evaluations_summary}

Please provide a comprehensive analysis of the relative strengths and weaknesses of hierarchical RAG
compared to standard RAG, with specific focus on retrieval quality and response generation."""

    # Make a request to the OpenAI API to generate the overall analysis
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},  # System message to guide the assistant
            {"role": "user", "content": user_prompt}  # User message with the evaluations summary
        ],
        temperature=0  # Set the temperature for response generation
    )

    # Return the generated overall analysis
    return response.choices[0].message.content
```

## Evaluation of Hierarchical and Standard RAG Approaches

```python
# Path to the PDF document containing AI information
pdf_path = "data/AI_Information.pdf"

# Example query about AI for testing the hierarchical RAG approach
query = "What are the key applications of transformer models in natural language processing?"
result = hierarchical_rag(query, pdf_path)

print("\n=== Response ===")
print(result["response"])

# Test query for formal evaluation (using only one query as requested)
test_queries = [
    "How do transformers handle sequential data compared to RNNs?"
]

# Reference answer for the test query to enable comparison
reference_answers = [
    "Transformers handle sequential data differently from RNNs by using self-attention mechanisms instead of recurrent connections. This allows transformers to process all tokens in parallel rather than sequentially, capturing long-range dependencies more efficiently and enabling better parallelization during training. Unlike RNNs, transformers don't suffer from vanishing gradient problems with long sequences."
]

# Run the evaluation comparing hierarchical and standard RAG approaches
evaluation_results = run_evaluation(
    pdf_path=pdf_path,
    test_queries=test_queries,
    reference_answers=reference_answers
)

# Print the overall analysis of the comparison
print("\n=== OVERALL ANALYSIS ===")
print(evaluation_results["overall_analysis"])
```

```output
Processing document and creating vector stores...
Extracting text from data/AI_Information.pdf...
Extracted 15 pages with content
Generating page summaries...
Summarizing page 1/15...
Summarizing page 2/15...
Summarizing page 3/15...
Summarizing page 4/15...
Summarizing page 5/15...
Summarizing page 6/15...
Summarizing page 7/15...
Summarizing page 8/15...
Summarizing page 9/15...
Summarizing page 10/15...
Summarizing page 11/15...
Summarizing page 12/15...
Summarizing page 13/15...
Summarizing page 14/15...
Summarizing page 15/15...
Created 47 detailed chunks
Creating embeddings for summaries...
Creating embeddings for detailed chunks...
Created vector stores with 15 summaries and 47 chunks
Performing hierarchical retrieval for query: What are the key applications of transformer models in natural language processing?
```

```output
C:\Users\faree\AppData\Local\Temp\ipykernel_9608\2918097221.py:62: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
  "similarity": float(score)  # Add the similarity score
```

```output
Retrieved 3 relevant summaries
Retrieved 10 detailed chunks from relevant pages

=== Response ===
I couldn't find any information about transformer models in the provided context. The context appears to focus on various applications of Artificial Intelligence (AI) and Machine Learning (ML), including computer vision, deep learning, reinforcement learning, and more. However, transformer models are not mentioned.

If you're looking for information on transformer models, I'd be happy to try and help you find it. Alternatively, if you have any other questions based on the provided context, I'd be happy to try and assist you.
Query: How do transformers handle sequential data compared to RNNs?

=== Comparing RAG approaches for query: How do transformers handle sequential data compared to RNNs? ===

Running hierarchical RAG...
Loading existing vector stores...
Performing hierarchical retrieval for query: How do transformers handle sequential data compared to RNNs?
Retrieved 3 relevant summaries
Retrieved 10 detailed chunks from relevant pages

Running standard RAG...
Extracting text from data/AI_Information.pdf...
Extracted 15 pages with content
Created 47 chunks for standard RAG
Creating embeddings for chunks...
Retrieved 15 chunks with standard RAG

=== OVERALL ANALYSIS ===
Based on the provided evaluation, I will provide a comprehensive analysis of the relative strengths and weaknesses of hierarchical RAG compared to standard RAG.

**Overview of Hierarchical RAG and Standard RAG**

Hierarchical RAG (Retrieval Algorithm for Generating) is an extension of the standard RAG approach, which involves dividing the input text into smaller chunks or sub-sequences to facilitate more efficient and effective retrieval. The hierarchical approach further divides these chunks into smaller sub-chunks, allowing for more granular and detailed retrieval.

Standard RAG, on the other hand, uses a single chunk size to retrieve relevant information from the input text.

**Strengths of Hierarchical RAG**

1. **Improved Retrieval Quality**: Hierarchical RAG's ability to divide the input text into smaller sub-chunks allows for more precise retrieval, as it can capture subtle nuances and relationships between words and phrases that may be missed by standard RAG.
2. **Enhanced Response Generation**: By considering multiple levels of granularity, hierarchical RAG can generate more accurate and informative responses, as it can take into account the context and relationships between different parts of the input text.
3. **Better Handling of Complex Input Text**: Hierarchical RAG is particularly well-suited for handling complex input text, such as long documents or texts with multiple layers of abstraction.

**Weaknesses of Hierarchical RAG**

1. **Increased Computational Complexity**: The hierarchical approach requires more computational resources and processing power, as it needs to handle multiple levels of granularity.
2. **Higher Risk of Overfitting**: The increased number of parameters and complexity of the hierarchical model can lead to overfitting, particularly if the training data is limited or biased.

**Strengths of Standard RAG**

1. **Simpler and Faster**: Standard RAG is a simpler and faster approach, as it only requires a single chunk size and less computational resources.
2. **Less Risk of Overfitting**: The standard model has fewer parameters and is less prone to overfitting, making it a more robust and reliable choice.

**Weaknesses of Standard RAG**

1. **Limited Retrieval Quality**: Standard RAG's single chunk size can lead to limited retrieval quality, as it may not capture the full range of nuances and relationships between words and phrases.
2. **Less Effective for Complex Input Text**: Standard RAG is less effective for handling complex input text, as it may struggle to capture the context and relationships between different parts of the text.

**When to Use Each Approach**

1. **Use Hierarchical RAG**:
	* When dealing with complex input text, such as long documents or texts with multiple layers of abstraction.
	* When high retrieval quality and response generation are critical, such as in applications requiring accurate and informative responses.
	* When computational resources are not a concern, and the benefits of hierarchical retrieval outweigh the costs.
2. **Use Standard RAG**:
	* When dealing with simple input text, such as short documents or texts with a clear and concise structure.
	* When computational resources are limited, and speed is a priority.
	* When the goal is to quickly retrieve relevant information, rather than generating accurate and informative responses.

In conclusion, hierarchical RAG offers improved retrieval quality and response generation, but at the cost of increased computational complexity and risk of overfitting. Standard RAG, on the other hand, is simpler and faster, but may have limited retrieval quality and be less effective for complex input text. The choice of approach depends on the specific requirements and constraints of the application.
```
