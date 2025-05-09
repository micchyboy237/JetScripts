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

```python
import os
import numpy as np
from rank_bm25 import BM25Okapi
import fitz
from openai import OpenAI
import re
import json
import time
from sklearn.metrics.pairwise import cosine_similarity
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
    Extract text content from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        str: Extracted text content
    """
    print(f"Extracting text from {pdf_path}...")  # Print the path of the PDF being processed
    pdf_document = fitz.open(pdf_path)  # Open the PDF file using PyMuPDF
    text = ""  # Initialize an empty string to store the extracted text

    # Iterate through each page in the PDF
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]  # Get the page object
        text += page.get_text()  # Extract text from the page and append to the text string

    return text  # Return the extracted text content
```

```python
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

    # Iterate over the text with the specified chunk size and overlap
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

    print(f"Created {len(chunks)} text chunks")  # Print the number of created chunks
    return chunks  # Return the list of chunks
```

```python
def clean_text(text):
    """
    Clean text by removing extra whitespace and special characters.

    Args:
        text (str): Input text

    Returns:
        str: Cleaned text
    """
    # Replace multiple whitespace characters (including newlines and tabs) with a single space
    text = re.sub(r'\s+', ' ', text)

    # Fix common OCR issues by replacing tab and newline characters with a space
    text = text.replace('\\t', ' ')
    text = text.replace('\\n', ' ')

    # Remove any leading or trailing whitespace and ensure single spaces between words
    text = ' '.join(text.split())

    return text
```

## Creating Our Vector Store

```python
def create_embeddings(texts, model="BAAI/bge-en-icl"):
    """
    Create embeddings for the given texts.

    Args:
        texts (str or List[str]): Input text(s)
        model (str): Embedding model name

    Returns:
        List[List[float]]: Embedding vectors
    """
    # Handle both string and list inputs
    input_texts = texts if isinstance(texts, list) else [texts]

    # Process in batches if needed (OpenAI API limits)
    batch_size = 100
    all_embeddings = []

    # Iterate over the input texts in batches
    for i in range(0, len(input_texts), batch_size):
        batch = input_texts[i:i + batch_size]  # Get the current batch of texts

        # Create embeddings for the current batch
        response = client.embeddings.create(
            model=model,
            input=batch
        )

        # Extract embeddings from the response
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)  # Add the batch embeddings to the list

    # If input was a string, return just the first embedding
    if isinstance(texts, str):
        return all_embeddings[0]

    # Otherwise return all embeddings
    return all_embeddings
```

```python
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

        # Convert query embedding to numpy array
        query_vector = np.array(query_embedding)

        # Calculate similarities using cosine similarity
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = cosine_similarity([query_vector], [vector])[0][0]  # Compute cosine similarity
            similarities.append((i, similarity))  # Append index and similarity score

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top k results with scores
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
```

## BM25 Implementation

```python
def create_bm25_index(chunks):
    """
    Create a BM25 index from the given chunks.

    Args:
        chunks (List[Dict]): List of text chunks

    Returns:
        BM25Okapi: A BM25 index
    """
    # Extract text from each chunk
    texts = [chunk["text"] for chunk in chunks]

    # Tokenize each document by splitting on whitespace
    tokenized_docs = [text.split() for text in texts]

    # Create the BM25 index using the tokenized documents
    bm25 = BM25Okapi(tokenized_docs)

    # Print the number of documents in the BM25 index
    print(f"Created BM25 index with {len(texts)} documents")

    return bm25
```

```python
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
    # Tokenize the query by splitting it into individual words
    query_tokens = query.split()

    # Get BM25 scores for the query tokens against the indexed documents
    scores = bm25.get_scores(query_tokens)

    # Initialize an empty list to store results with their scores
    results = []

    # Iterate over the scores and corresponding chunks
    for i, score in enumerate(scores):
        # Create a copy of the metadata to avoid modifying the original
        metadata = chunks[i].get("metadata", {}).copy()
        # Add index to metadata
        metadata["index"] = i

        results.append({
            "text": chunks[i]["text"],
            "metadata": metadata,  # Add metadata with index
            "bm25_score": float(score)
        })

    # Sort the results by BM25 score in descending order
    results.sort(key=lambda x: x["bm25_score"], reverse=True)

    # Return the top k results
    return results[:k]
```

## Fusion Retrieval Function

```python
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
    print(f"Performing fusion retrieval for query: {query}")

    # Define small epsilon to avoid division by zero
    epsilon = 1e-8

    # Get vector search results
    query_embedding = create_embeddings(query)  # Create embedding for the query
    vector_results = vector_store.similarity_search_with_scores(query_embedding, k=len(chunks))  # Perform vector search

    # Get BM25 search results
    bm25_results = bm25_search(bm25_index, chunks, query, k=len(chunks))  # Perform BM25 search

    # Create dictionaries to map document index to score
    vector_scores_dict = {result["metadata"]["index"]: result["similarity"] for result in vector_results}
    bm25_scores_dict = {result["metadata"]["index"]: result["bm25_score"] for result in bm25_results}

    # Ensure all documents have scores for both methods
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

    # Extract scores as arrays
    vector_scores = np.array([doc["vector_score"] for doc in combined_results])
    bm25_scores = np.array([doc["bm25_score"] for doc in combined_results])

    # Normalize scores
    norm_vector_scores = (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + epsilon)
    norm_bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + epsilon)

    # Compute combined scores
    combined_scores = alpha * norm_vector_scores + (1 - alpha) * norm_bm25_scores

    # Add combined scores to results
    for i, score in enumerate(combined_scores):
        combined_results[i]["combined_score"] = float(score)

    # Sort by combined score (descending)
    combined_results.sort(key=lambda x: x["combined_score"], reverse=True)

    # Return top k results
    top_results = combined_results[:k]

    print(f"Retrieved {len(top_results)} documents with fusion retrieval")
    return top_results
```

## Document Processing Pipeline

```python
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
    # Extract text from the PDF file
    text = extract_text_from_pdf(pdf_path)

    # Clean the extracted text to remove extra whitespace and special characters
    cleaned_text = clean_text(text)

    # Split the cleaned text into overlapping chunks
    chunks = chunk_text(cleaned_text, chunk_size, chunk_overlap)

    # Extract the text content from each chunk for embedding creation
    chunk_texts = [chunk["text"] for chunk in chunks]
    print("Creating embeddings for chunks...")

    # Create embeddings for the chunk texts
    embeddings = create_embeddings(chunk_texts)

    # Initialize the vector store
    vector_store = SimpleVectorStore()

    # Add the chunks and their embeddings to the vector store
    vector_store.add_items(chunks, embeddings)
    print(f"Added {len(chunks)} items to vector store")

    # Create a BM25 index from the chunks
    bm25_index = create_bm25_index(chunks)

    # Return the chunks, vector store, and BM25 index
    return chunks, vector_store, bm25_index
```

## Response Generation

```python
def generate_response(query, context):
    """
    Generate a response based on the query and context.

    Args:
        query (str): User query
        context (str): Context from retrieved documents

    Returns:
        str: Generated response
    """
    # Define the system prompt to guide the AI assistant
    system_prompt = """You are a helpful AI assistant. Answer the user's question based on the provided context.
    If the context doesn't contain relevant information to answer the question fully, acknowledge this limitation."""

    # Format the user prompt with the context and query
    user_prompt = f"""Context:
    {context}

    Question: {query}

    Please answer the question based on the provided context."""

    # Generate the response using the OpenAI API
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # Specify the model to use
        messages=[
            {"role": "system", "content": system_prompt},  # System message to guide the assistant
            {"role": "user", "content": user_prompt}  # User message with context and query
        ],
        temperature=0.1  # Set the temperature for response generation
    )

    # Return the generated response
    return response.choices[0].message.content
```

## Main Retrieval Function

```python
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
    # Retrieve documents using fusion retrieval method
    retrieved_docs = fusion_retrieval(query, chunks, vector_store, bm25_index, k=k, alpha=alpha)

    # Format the context from the retrieved documents by joining their text with separators
    context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_docs])

    # Generate a response based on the query and the formatted context
    response = generate_response(query, context)

    # Return the query, retrieved documents, and the generated response
    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "response": response
    }
```

## Comparing Retrieval Methods

```python
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
    # Create query embedding
    query_embedding = create_embeddings(query)

    # Retrieve documents using vector-based similarity search
    retrieved_docs = vector_store.similarity_search_with_scores(query_embedding, k=k)

    # Format the context from the retrieved documents by joining their text with separators
    context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_docs])

    # Generate a response based on the query and the formatted context
    response = generate_response(query, context)

    # Return the query, retrieved documents, and the generated response
    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "response": response
    }
```

```python
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
    # Retrieve documents using BM25 search
    retrieved_docs = bm25_search(bm25_index, chunks, query, k=k)

    # Format the context from the retrieved documents by joining their text with separators
    context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_docs])

    # Generate a response based on the query and the formatted context
    response = generate_response(query, context)

    # Return the query, retrieved documents, and the generated response
    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "response": response
    }
```

## Evaluation Functions

```python
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
    print(f"\n=== Comparing retrieval methods for query: {query} ===\n")

    # Run vector-only RAG
    print("\nRunning vector-only RAG...")
    vector_result = vector_only_rag(query, vector_store, k)

    # Run BM25-only RAG
    print("\nRunning BM25-only RAG...")
    bm25_result = bm25_only_rag(query, chunks, bm25_index, k)

    # Run fusion RAG
    print("\nRunning fusion RAG...")
    fusion_result = answer_with_fusion_rag(query, chunks, vector_store, bm25_index, k, alpha)

    # Compare responses from different retrieval methods
    print("\nComparing responses...")
    comparison = evaluate_responses(
        query,
        vector_result["response"],
        bm25_result["response"],
        fusion_result["response"],
        reference_answer
    )

    # Return the comparison results
    return {
        "query": query,
        "vector_result": vector_result,
        "bm25_result": bm25_result,
        "fusion_result": fusion_result,
        "comparison": comparison
    }
```

```python
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
    # System prompt for the evaluator to guide the evaluation process
    system_prompt = """You are an expert evaluator of RAG systems. Compare responses from three different retrieval approaches:
    1. Vector-based retrieval: Uses semantic similarity for document retrieval
    2. BM25 keyword retrieval: Uses keyword matching for document retrieval
    3. Fusion retrieval: Combines both vector and keyword approaches

    Evaluate the responses based on:
    - Relevance to the query
    - Factual correctness
    - Comprehensiveness
    - Clarity and coherence"""

    # User prompt containing the query and responses
    user_prompt = f"""Query: {query}

    Vector-based response:
    {vector_response}

    BM25 keyword response:
    {bm25_response}

    Fusion response:
    {fusion_response}
    """

    # Add reference answer to the prompt if provided
    if reference_answer:
        user_prompt += f"""
            Reference answer:
            {reference_answer}
        """

    # Add instructions for detailed comparison to the user prompt
    user_prompt += """
    Please provide a detailed comparison of these three responses. Which approach performed best for this query and why?
    Be specific about the strengths and weaknesses of each approach for this particular query.
    """

    # Generate the evaluation using meta-llama/Llama-3.2-3B-Instruct
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # Specify the model to use
        messages=[
            {"role": "system", "content": system_prompt},  # System message to guide the evaluator
            {"role": "user", "content": user_prompt}  # User message with query and responses
        ],
        temperature=0  # Set the temperature for response generation
    )

    # Return the generated evaluation content
    return response.choices[0].message.content
```

## Complete Evaluation Pipeline

```python
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
    print("=== EVALUATING FUSION RETRIEVAL ===\n")

    # Process the document to extract text, create chunks, and build vector and BM25 indices
    chunks, vector_store, bm25_index = process_document(pdf_path)

    # Initialize a list to store results for each query
    results = []

    # Iterate over each test query
    for i, query in enumerate(test_queries):
        print(f"\n\n=== Evaluating Query {i+1}/{len(test_queries)} ===")
        print(f"Query: {query}")

        # Get the reference answer if available
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]

        # Compare retrieval methods for the current query
        comparison = compare_retrieval_methods(
            query,
            chunks,
            vector_store,
            bm25_index,
            k=k,
            alpha=alpha,
            reference_answer=reference
        )

        # Append the comparison results to the results list
        results.append(comparison)

        # Print the responses from different retrieval methods
        print("\n=== Vector-based Response ===")
        print(comparison["vector_result"]["response"])

        print("\n=== BM25 Response ===")
        print(comparison["bm25_result"]["response"])

        print("\n=== Fusion Response ===")
        print(comparison["fusion_result"]["response"])

        print("\n=== Comparison ===")
        print(comparison["comparison"])

    # Generate an overall analysis of the fusion retrieval performance
    overall_analysis = generate_overall_analysis(results)

    # Return the results and overall analysis
    return {
        "results": results,
        "overall_analysis": overall_analysis
    }
```

```python
def generate_overall_analysis(results):
    """
    Generate an overall analysis of fusion retrieval.

    Args:
        results (List[Dict]): Results from evaluating queries

    Returns:
        str: Overall analysis
    """
    # System prompt to guide the evaluation process
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

    # Create a summary of evaluations for each query
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        evaluations_summary += f"Comparison Summary: {result['comparison'][:200]}...\n\n"

    # User prompt containing the evaluations summary
    user_prompt = f"""Based on the following evaluations of different retrieval methods across {len(results)} queries,
    provide an overall analysis comparing these three approaches:

    {evaluations_summary}

    Please provide a comprehensive analysis of vector-based, BM25, and fusion retrieval approaches,
    highlighting when and why fusion retrieval provides advantages over the individual methods."""

    # Generate the overall analysis using meta-llama/Llama-3.2-3B-Instruct
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # Return the generated analysis content
    return response.choices[0].message.content
```

## Evaluating Fusion Retrieval

```python
# Path to PDF document
# Path to PDF document containing AI information for knowledge retrieval testing
pdf_path = "data/AI_Information.pdf"

# Define a single AI-related test query
test_queries = [
    "What are the main applications of transformer models in natural language processing?"  # AI-specific query
]

# Optional reference answer
reference_answers = [
    "Transformer models have revolutionized natural language processing with applications including machine translation, text summarization, question answering, sentiment analysis, and text generation. They excel at capturing long-range dependencies in text and have become the foundation for models like BERT, GPT, and T5.",
]

# Set parameters
k = 5  # Number of documents to retrieve
alpha = 0.5  # Weight for vector scores (0.5 means equal weight between vector and BM25)

# Run evaluation
evaluation_results = evaluate_fusion_retrieval(
    pdf_path=pdf_path,
    test_queries=test_queries,
    reference_answers=reference_answers,
    k=k,
    alpha=alpha
)

# Print overall analysis
print("\n\n=== OVERALL ANALYSIS ===\n")
print(evaluation_results["overall_analysis"])
```

```output
=== EVALUATING FUSION RETRIEVAL ===

Extracting text from data/AI_Information.pdf...
Created 42 text chunks
Creating embeddings for chunks...
Added 42 items to vector store
Created BM25 index with 42 documents


=== Evaluating Query 1/1 ===
Query: What are the main applications of transformer models in natural language processing?

=== Comparing retrieval methods for query: What are the main applications of transformer models in natural language processing? ===


Running vector-only RAG...

Running BM25-only RAG...

Running fusion RAG...
Performing fusion retrieval for query: What are the main applications of transformer models in natural language processing?
Retrieved 5 documents with fusion retrieval

Comparing responses...

=== Vector-based Response ===
The provided context does not mention transformer models specifically. However, it does mention Natural Language Processing (NLP) as a branch of AI that focuses on enabling computers to understand, interpret, and generate human language. NLP techniques are used in chatbots, machine translation, text summarization, and sentiment analysis.

Transformer models are a type of neural network architecture that is particularly effective for NLP tasks, such as machine translation, text generation, and text classification. They are not explicitly mentioned in the provided context.

If you're looking for information on transformer models, I can provide general information on this topic. However, please note that the context provided does not specifically address transformer models.

=== BM25 Response ===
The provided context does not mention transformer models or their applications in natural language processing. The context covers various topics such as deep learning, convolutional neural networks, recurrent neural networks, natural language processing, and machine learning, but it does not specifically discuss transformer models.

If you're looking for information on transformer models, I can provide general information on this topic. Transformer models are a type of neural network architecture that have gained popularity in natural language processing tasks such as machine translation, text generation, and language understanding. They are particularly effective in handling long-range dependencies in sequential data and have been widely adopted in many NLP applications. However, this information is not present in the provided context.

=== Fusion Response ===
The provided context does not explicitly mention the main applications of transformer models in natural language processing. However, it does mention that Generative Adversarial Networks (GANs) and transformers are examples of generative AI models that can create original content, including images, text, and music.

Based on general knowledge, transformer models are widely used in natural language processing (NLP) for tasks such as:

1. Machine translation
2. Text generation
3. Sentiment analysis
4. Text classification
5. Language modeling

These models have achieved state-of-the-art results in many NLP tasks and have become a popular choice for many applications.

If you're looking for more specific information on the applications of transformer models in NLP, I can try to provide more general information or point you in the direction of more resources.

=== Comparison ===
**Comparison of Vector-based, BM25 Keyword, and Fusion Retrieval Approaches**

For the given query, "What are the main applications of transformer models in natural language processing?", we can evaluate the responses based on relevance, factual correctness, comprehensiveness, and clarity/coherence.

**Relevance:**

* Vector-based response: 6/10 (The response is relevant to the query, but it does not directly answer the question. It provides general information about NLP and mentions transformer models, but does not explicitly state their main applications.)
* BM25 keyword response: 5/10 (The response is not directly relevant to the query, as it does not mention transformer models or their applications in NLP.)
* Fusion response: 9/10 (The response directly answers the question and provides a comprehensive list of transformer models' main applications in NLP.)

**Factual Correctness:**

* Vector-based response: 8/10 (The response is generally correct, but it does not explicitly mention the main applications of transformer models in NLP.)
* BM25 keyword response: 8/10 (The response is generally correct, but it does not mention transformer models or their applications in NLP.)
* Fusion response: 9/10 (The response is factually correct and provides a comprehensive list of transformer models' main applications in NLP.)

**Comprehensiveness:**

* Vector-based response: 6/10 (The response provides general information about NLP, but does not explicitly state the main applications of transformer models.)
* BM25 keyword response: 4/10 (The response does not provide any information about transformer models or their applications in NLP.)
* Fusion response: 9/10 (The response provides a comprehensive list of transformer models' main applications in NLP.)

**Clarity and Coherence:**

* Vector-based response: 7/10 (The response is clear, but it does not explicitly state the main applications of transformer models.)
* BM25 keyword response: 6/10 (The response is clear, but it does not mention transformer models or their applications in NLP.)
* Fusion response: 9/10 (The response is clear, concise, and well-organized, making it easy to understand the main applications of transformer models in NLP.)

**Overall Performance:**

* Vector-based response: 6.5/10
* BM25 keyword response: 5.5/10
* Fusion response: 8.5/10

Based on the evaluation, the Fusion retrieval approach performed best for this query. The Fusion response provided a comprehensive list of transformer models' main applications in NLP, was factually correct, and was clear and concise. The Vector-based response was relevant but did not explicitly state the main applications of transformer models, while the BM25 keyword response was not directly relevant to the query.


=== OVERALL ANALYSIS ===

**Overall Analysis: Vector-based, BM25, and Fusion Retrieval Approaches**

In this analysis, we will evaluate the performance of three retrieval approaches: Vector-based, BM25 Keyword, and Fusion Retrieval. We will examine the strengths and weaknesses of each approach, their performance on specific query types, and how fusion retrieval balances the trade-offs.

**Query 1: What are the main applications of transformer models in natural language processing?**

For this query, we can evaluate the performance of the three approaches as follows:

1. **Vector-based Retrieval (Semantic Similarity)**: This approach is suitable for queries that require understanding the semantic meaning of the query and the documents. In this case, the query is asking about the main applications of transformer models, which implies a need for semantic understanding. The vector-based approach is likely to perform well, as it can capture the nuances of the query and the documents.

Performance: 8/10

2. **BM25 Keyword Retrieval (Keyword Matching)**: This approach is suitable for queries that require exact keyword matching. In this case, the query is asking about the main applications of transformer models, which implies a need for exact keyword matching. However, the query is also asking about the main applications, which may require a more nuanced understanding of the documents.

Performance: 6/10

3. **Fusion Retrieval (Combination of Both)**: This approach combines the strengths of both vector-based and BM25 keyword retrieval. By using a combination of both approaches, fusion retrieval can capture both the semantic meaning of the query and the exact keyword matching.

Performance: 9/10

**Overall Strengths and Weaknesses of Each Approach**

1. **Vector-based Retrieval (Semantic Similarity)**:
	* Strengths: Can capture nuances of the query and documents, suitable for queries that require semantic understanding.
	* Weaknesses: May not perform well for queries that require exact keyword matching.
2. **BM25 Keyword Retrieval (Keyword Matching)**:
	* Strengths: Can perform well for queries that require exact keyword matching.
	* Weaknesses: May not capture nuances of the query and documents, suitable for queries that require semantic understanding.
3. **Fusion Retrieval (Combination of Both)**:
	* Strengths: Can capture both the semantic meaning of the query and the exact keyword matching, suitable for a wide range of queries.
	* Weaknesses: May require more computational resources and complex implementation.

**How Fusion Retrieval Balances the Trade-Offs**

Fusion retrieval balances the trade-offs between vector-based and BM25 keyword retrieval by combining the strengths of both approaches. By using a combination of both, fusion retrieval can capture both the semantic meaning of the query and the exact keyword matching, resulting in a more comprehensive search result.

**Recommendations for When to Use Each Approach**

1. **Vector-based Retrieval (Semantic Similarity)**: Use for queries that require semantic understanding, such as questions that ask about the meaning or context of a term.
2. **BM25 Keyword Retrieval (Keyword Matching)**: Use for queries that require exact keyword matching, such as searches for specific terms or phrases.
3. **Fusion Retrieval (Combination of Both)**: Use for queries that require a balance between semantic understanding and exact keyword matching, such as searches for terms or phrases with nuanced meanings.

In conclusion, fusion retrieval provides advantages over individual methods by combining the strengths of both vector-based and BM25 keyword retrieval. By using a combination of both approaches, fusion retrieval can capture both the semantic meaning of the query and the exact keyword matching, resulting in a more comprehensive search result.
```
