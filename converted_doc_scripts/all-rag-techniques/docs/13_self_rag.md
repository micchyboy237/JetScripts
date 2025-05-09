# Self-RAG: A Dynamic Approach to RAG

In this notebook, I implement Self-RAG, an advanced RAG system that dynamically decides when and how to use retrieved information. Unlike traditional RAG approaches, Self-RAG introduces reflection points throughout the retrieval and generation process, resulting in higher quality and more reliable responses.

## Key Components of Self-RAG

1. **Retrieval Decision**: Determines if retrieval is even necessary for a given query
2. **Document Retrieval**: Fetches potentially relevant documents when needed
3. **Relevance Evaluation**: Assesses how relevant each retrieved document is
4. **Response Generation**: Creates responses based on relevant contexts
5. **Support Assessment**: Evaluates if responses are properly grounded in the context
6. **Utility Evaluation**: Rates the overall usefulness of generated responses

## Setting Up the Environment
We begin by importing necessary libraries.

```python
import os
import numpy as np
import json
import fitz
from openai import OpenAI
import re
```

## Extracting Text from a PDF File
To implement RAG, we first need a source of textual data. In this case, we extract text from a PDF file using the PyMuPDF library.

```python
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file and prints the first `num_chars` characters.

    Args:
    pdf_path (str): Path to the PDF file.

    Returns:
    str: Extracted text from the PDF.
    """
    # Open the PDF file
    mypdf = fitz.open(pdf_path)
    all_text = ""  # Initialize an empty string to store the extracted text

    # Iterate through each page in the PDF
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]  # Get the page
        text = page.get_text("text")  # Extract text from the page
        all_text += text  # Append the extracted text to the all_text string

    return all_text  # Return the extracted text
```

## Chunking the Extracted Text
Once we have the extracted text, we divide it into smaller, overlapping chunks to improve retrieval accuracy.

```python
def chunk_text(text, n, overlap):
    """
    Chunks the given text into segments of n characters with overlap.

    Args:
    text (str): The text to be chunked.
    n (int): The number of characters in each chunk.
    overlap (int): The number of overlapping characters between chunks.

    Returns:
    List[str]: A list of text chunks.
    """
    chunks = []  # Initialize an empty list to store the chunks

    # Loop through the text with a step size of (n - overlap)
    for i in range(0, len(text), n - overlap):
        # Append a chunk of text from index i to i + n to the chunks list
        chunks.append(text[i:i + n])

    return chunks  # Return the list of text chunks
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

## Simple Vector Store Implementation
We'll create a basic vector store to manage document chunks and their embeddings.

```python
class SimpleVectorStore:
    """
    A simple vector store implementation using NumPy.
    """
    def __init__(self):
        """
        Initialize the vector store.
        """
        self.vectors = []  # List to store embedding vectors
        self.texts = []  # List to store original texts
        self.metadata = []  # List to store metadata for each text

    def add_item(self, text, embedding, metadata=None):
        """
        Add an item to the vector store.

        Args:
        text (str): The original text.
        embedding (List[float]): The embedding vector.
        metadata (dict, optional): Additional metadata.
        """
        self.vectors.append(np.array(embedding))  # Convert embedding to numpy array and add to vectors list
        self.texts.append(text)  # Add the original text to texts list
        self.metadata.append(metadata or {})  # Add metadata to metadata list, default to empty dict if None

    def similarity_search(self, query_embedding, k=5, filter_func=None):
        """
        Find the most similar items to a query embedding.

        Args:
        query_embedding (List[float]): Query embedding vector.
        k (int): Number of results to return.
        filter_func (callable, optional): Function to filter results.

        Returns:
        List[Dict]: Top k most similar items with their texts and metadata.
        """
        if not self.vectors:
            return []  # Return empty list if no vectors are stored

        # Convert query embedding to numpy array
        query_vector = np.array(query_embedding)

        # Calculate similarities using cosine similarity
        similarities = []
        for i, vector in enumerate(self.vectors):
            # Apply filter if provided
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
                "text": self.texts[idx],  # Add the text
                "metadata": self.metadata[idx],  # Add the metadata
                "similarity": score  # Add the similarity score
            })

        return results  # Return the list of top k results
```

## Creating Embeddings

```python
def create_embeddings(text, model="BAAI/bge-en-icl"):
    """
    Creates embeddings for the given text.

    Args:
    text (str or List[str]): The input text(s) for which embeddings are to be created.
    model (str): The model to be used for creating embeddings.

    Returns:
    List[float] or List[List[float]]: The embedding vector(s).
    """
    # Handle both string and list inputs by converting string input to a list
    input_text = text if isinstance(text, list) else [text]

    # Create embeddings for the input text using the specified model
    response = client.embeddings.create(
        model=model,
        input=input_text
    )

    # If the input was a single string, return just the first embedding
    if isinstance(text, str):
        return response.data[0].embedding

    # Otherwise, return all embeddings for the list of texts
    return [item.embedding for item in response.data]
```

## Document Processing Pipeline

```python
def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    Process a document for Self-RAG.

    Args:
        pdf_path (str): Path to the PDF file.
        chunk_size (int): Size of each chunk in characters.
        chunk_overlap (int): Overlap between chunks in characters.

    Returns:
        SimpleVectorStore: A vector store containing document chunks and their embeddings.
    """
    # Extract text from the PDF file
    print("Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_path)

    # Chunk the extracted text
    print("Chunking text...")
    chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} text chunks")

    # Create embeddings for each chunk
    print("Creating embeddings for chunks...")
    chunk_embeddings = create_embeddings(chunks)

    # Initialize the vector store
    store = SimpleVectorStore()

    # Add each chunk and its embedding to the vector store
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={"index": i, "source": pdf_path}
        )

    print(f"Added {len(chunks)} chunks to the vector store")
    return store
```

## Self-RAG Components
### 1. Retrieval Decision

```python
def determine_if_retrieval_needed(query):
    """
    Determines if retrieval is necessary for the given query.

    Args:
        query (str): User query

    Returns:
        bool: True if retrieval is needed, False otherwise
    """
    # System prompt to instruct the AI on how to determine if retrieval is necessary
    system_prompt = """You are an AI assistant that determines if retrieval is necessary to answer a query.
    For factual questions, specific information requests, or questions about events, people, or concepts, answer "Yes".
    For opinions, hypothetical scenarios, or simple queries with common knowledge, answer "No".
    Answer with ONLY "Yes" or "No"."""

    # User prompt containing the query
    user_prompt = f"Query: {query}\n\nIs retrieval necessary to answer this query accurately?"

    # Generate response from the model
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # Extract the answer from the model's response and convert to lowercase
    answer = response.choices[0].message.content.strip().lower()

    # Return True if the answer contains "yes", otherwise return False
    return "yes" in answer
```

### 2. Relevance Evaluation

```python
def evaluate_relevance(query, context):
    """
    Evaluates the relevance of a context to the query.

    Args:
        query (str): User query
        context (str): Context text

    Returns:
        str: 'relevant' or 'irrelevant'
    """
    # System prompt to instruct the AI on how to determine document relevance
    system_prompt = """You are an AI assistant that determines if a document is relevant to a query.
    Consider whether the document contains information that would be helpful in answering the query.
    Answer with ONLY "Relevant" or "Irrelevant"."""

    # Truncate context if it is too long to avoid exceeding token limits
    max_context_length = 2000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "... [truncated]"

    # User prompt containing the query and the document content
    user_prompt = f"""Query: {query}
    Document content:
    {context}

    Is this document relevant to the query? Answer with ONLY "Relevant" or "Irrelevant".
    """

    # Generate response from the model
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # Extract the answer from the model's response and convert to lowercase
    answer = response.choices[0].message.content.strip().lower()

    return answer  # Return the relevance evaluation
```

### 3. Support Assessment

```python
def assess_support(response, context):
    """
    Assesses how well a response is supported by the context.

    Args:
        response (str): Generated response
        context (str): Context text

    Returns:
        str: 'fully supported', 'partially supported', or 'no support'
    """
    # System prompt to instruct the AI on how to evaluate support
    system_prompt = """You are an AI assistant that determines if a response is supported by the given context.
    Evaluate if the facts, claims, and information in the response are backed by the context.
    Answer with ONLY one of these three options:
    - "Fully supported": All information in the response is directly supported by the context.
    - "Partially supported": Some information in the response is supported by the context, but some is not.
    - "No support": The response contains significant information not found in or contradicting the context.
    """

    # Truncate context if it is too long to avoid exceeding token limits
    max_context_length = 2000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "... [truncated]"

    # User prompt containing the context and the response to be evaluated
    user_prompt = f"""Context:
    {context}

    Response:
    {response}

    How well is this response supported by the context? Answer with ONLY "Fully supported", "Partially supported", or "No support".
    """

    # Generate response from the model
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # Extract the answer from the model's response and convert to lowercase
    answer = response.choices[0].message.content.strip().lower()

    return answer  # Return the support assessment
```

### 4. Utility Evaluation

```python
def rate_utility(query, response):
    """
    Rates the utility of a response for the query.

    Args:
        query (str): User query
        response (str): Generated response

    Returns:
        int: Utility rating from 1 to 5
    """
    # System prompt to instruct the AI on how to rate the utility of the response
    system_prompt = """You are an AI assistant that rates the utility of a response to a query.
    Consider how well the response answers the query, its completeness, correctness, and helpfulness.
    Rate the utility on a scale from 1 to 5, where:
    - 1: Not useful at all
    - 2: Slightly useful
    - 3: Moderately useful
    - 4: Very useful
    - 5: Exceptionally useful
    Answer with ONLY a single number from 1 to 5."""

    # User prompt containing the query and the response to be rated
    user_prompt = f"""Query: {query}
    Response:
    {response}

    Rate the utility of this response on a scale from 1 to 5:"""

    # Generate the utility rating using the OpenAI client
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # Extract the rating from the model's response
    rating = response.choices[0].message.content.strip()

    # Extract just the number from the rating
    rating_match = re.search(r'[1-5]', rating)
    if rating_match:
        return int(rating_match.group())  # Return the extracted rating as an integer

    return 3  # Default to middle rating if parsing fails
```

## Response Generation

```python
def generate_response(query, context=None):
    """
    Generates a response based on the query and optional context.

    Args:
        query (str): User query
        context (str, optional): Context text

    Returns:
        str: Generated response
    """
    # System prompt to instruct the AI on how to generate a helpful response
    system_prompt = """You are a helpful AI assistant. Provide a clear, accurate, and informative response to the query."""

    # Create the user prompt based on whether context is provided
    if context:
        user_prompt = f"""Context:
        {context}

        Query: {query}

        Please answer the query based on the provided context.
        """
    else:
        user_prompt = f"""Query: {query}

        Please answer the query to the best of your ability."""

    # Generate the response using the OpenAI client
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )

    # Return the generated response text
    return response.choices[0].message.content.strip()
```

## Complete Self-RAG Implementation

```python
def self_rag(query, vector_store, top_k=3):
    """
    Implements the complete Self-RAG pipeline.

    Args:
        query (str): User query
        vector_store (SimpleVectorStore): Vector store containing document chunks
        top_k (int): Number of documents to retrieve initially

    Returns:
        dict: Results including query, response, and metrics from the Self-RAG process
    """
    print(f"\n=== Starting Self-RAG for query: {query} ===\n")

    # Step 1: Determine if retrieval is necessary
    print("Step 1: Determining if retrieval is necessary...")
    retrieval_needed = determine_if_retrieval_needed(query)
    print(f"Retrieval needed: {retrieval_needed}")

    # Initialize metrics to track the Self-RAG process
    metrics = {
        "retrieval_needed": retrieval_needed,
        "documents_retrieved": 0,
        "relevant_documents": 0,
        "response_support_ratings": [],
        "utility_ratings": []
    }

    best_response = None
    best_score = -1

    if retrieval_needed:
        # Step 2: Retrieve documents
        print("\nStep 2: Retrieving relevant documents...")
        query_embedding = create_embeddings(query)
        results = vector_store.similarity_search(query_embedding, k=top_k)
        metrics["documents_retrieved"] = len(results)
        print(f"Retrieved {len(results)} documents")

        # Step 3: Evaluate relevance of each document
        print("\nStep 3: Evaluating document relevance...")
        relevant_contexts = []

        for i, result in enumerate(results):
            context = result["text"]
            relevance = evaluate_relevance(query, context)
            print(f"Document {i+1} relevance: {relevance}")

            if relevance == "relevant":
                relevant_contexts.append(context)

        metrics["relevant_documents"] = len(relevant_contexts)
        print(f"Found {len(relevant_contexts)} relevant documents")

        if relevant_contexts:
            # Step 4: Process each relevant context
            print("\nStep 4: Processing relevant contexts...")
            for i, context in enumerate(relevant_contexts):
                print(f"\nProcessing context {i+1}/{len(relevant_contexts)}...")

                # Generate response based on the context
                print("Generating response...")
                response = generate_response(query, context)

                # Assess how well the response is supported by the context
                print("Assessing support...")
                support_rating = assess_support(response, context)
                print(f"Support rating: {support_rating}")
                metrics["response_support_ratings"].append(support_rating)

                # Rate the utility of the response
                print("Rating utility...")
                utility_rating = rate_utility(query, response)
                print(f"Utility rating: {utility_rating}/5")
                metrics["utility_ratings"].append(utility_rating)

                # Calculate overall score (higher for better support and utility)
                support_score = {
                    "fully supported": 3,
                    "partially supported": 1,
                    "no support": 0
                }.get(support_rating, 0)

                overall_score = support_score * 5 + utility_rating
                print(f"Overall score: {overall_score}")

                # Keep track of the best response
                if overall_score > best_score:
                    best_response = response
                    best_score = overall_score
                    print("New best response found!")

        # If no relevant contexts were found or all responses scored poorly
        if not relevant_contexts or best_score <= 0:
            print("\nNo suitable context found or poor responses, generating without retrieval...")
            best_response = generate_response(query)
    else:
        # No retrieval needed, generate directly
        print("\nNo retrieval needed, generating response directly...")
        best_response = generate_response(query)

    # Final metrics
    metrics["best_score"] = best_score
    metrics["used_retrieval"] = retrieval_needed and best_score > 0

    print("\n=== Self-RAG Completed ===")

    return {
        "query": query,
        "response": best_response,
        "metrics": metrics
    }
```

## Running the Complete Self-RAG System

```python
def run_self_rag_example():
    """
    Demonstrates the complete Self-RAG system with examples.
    """
    # Process document
    pdf_path = "data/AI_Information.pdf"  # Path to the PDF document
    print(f"Processing document: {pdf_path}")
    vector_store = process_document(pdf_path)  # Process the document and create a vector store

    # Example 1: Query likely needing retrieval
    query1 = "What are the main ethical concerns in AI development?"
    print("\n" + "="*80)
    print(f"EXAMPLE 1: {query1}")
    result1 = self_rag(query1, vector_store)  # Run Self-RAG for the first query
    print("\nFinal response:")
    print(result1["response"])  # Print the final response for the first query
    print("\nMetrics:")
    print(json.dumps(result1["metrics"], indent=2))  # Print the metrics for the first query

    # Example 2: Query likely not needing retrieval
    query2 = "Can you write a short poem about artificial intelligence?"
    print("\n" + "="*80)
    print(f"EXAMPLE 2: {query2}")
    result2 = self_rag(query2, vector_store)  # Run Self-RAG for the second query
    print("\nFinal response:")
    print(result2["response"])  # Print the final response for the second query
    print("\nMetrics:")
    print(json.dumps(result2["metrics"], indent=2))  # Print the metrics for the second query

    # Example 3: Query with some relevance to document but requiring additional knowledge
    query3 = "How might AI impact economic growth in developing countries?"
    print("\n" + "="*80)
    print(f"EXAMPLE 3: {query3}")
    result3 = self_rag(query3, vector_store)  # Run Self-RAG for the third query
    print("\nFinal response:")
    print(result3["response"])  # Print the final response for the third query
    print("\nMetrics:")
    print(json.dumps(result3["metrics"], indent=2))  # Print the metrics for the third query

    return {
        "example1": result1,
        "example2": result2,
        "example3": result3
    }
```

## Evaluating Self-RAG Against Traditional RAG

```python
def traditional_rag(query, vector_store, top_k=3):
    """
    Implements a traditional RAG approach for comparison.

    Args:
        query (str): User query
        vector_store (SimpleVectorStore): Vector store containing document chunks
        top_k (int): Number of documents to retrieve

    Returns:
        str: Generated response
    """
    print(f"\n=== Running traditional RAG for query: {query} ===\n")

    # Retrieve documents
    print("Retrieving documents...")
    query_embedding = create_embeddings(query)  # Create embeddings for the query
    results = vector_store.similarity_search(query_embedding, k=top_k)  # Search for similar documents
    print(f"Retrieved {len(results)} documents")

    # Combine contexts from retrieved documents
    contexts = [result["text"] for result in results]  # Extract text from results
    combined_context = "\n\n".join(contexts)  # Combine texts into a single context

    # Generate response using the combined context
    print("Generating response...")
    response = generate_response(query, combined_context)  # Generate response based on the combined context

    return response
```

```python
def evaluate_rag_approaches(pdf_path, test_queries, reference_answers=None):
    """
    Compare Self-RAG with traditional RAG.

    Args:
        pdf_path (str): Path to the document
        test_queries (List[str]): List of test queries
        reference_answers (List[str], optional): Reference answers for evaluation

    Returns:
        dict: Evaluation results
    """
    print("=== Evaluating RAG Approaches ===")

    # Process document to create a vector store
    vector_store = process_document(pdf_path)

    results = []

    for i, query in enumerate(test_queries):
        print(f"\nProcessing query {i+1}: {query}")

        # Run Self-RAG
        self_rag_result = self_rag(query, vector_store)  # Get response from Self-RAG
        self_rag_response = self_rag_result["response"]

        # Run traditional RAG
        trad_rag_response = traditional_rag(query, vector_store)  # Get response from traditional RAG

        # Compare results if reference answer is available
        reference = reference_answers[i] if reference_answers and i < len(reference_answers) else None
        comparison = compare_responses(query, self_rag_response, trad_rag_response, reference)  # Compare responses

        results.append({
            "query": query,
            "self_rag_response": self_rag_response,
            "traditional_rag_response": trad_rag_response,
            "reference_answer": reference,
            "comparison": comparison,
            "self_rag_metrics": self_rag_result["metrics"]
        })

    # Generate overall analysis
    overall_analysis = generate_overall_analysis(results)

    return {
        "results": results,
        "overall_analysis": overall_analysis
    }
```

```python
def compare_responses(query, self_rag_response, trad_rag_response, reference=None):
    """
    Compare responses from Self-RAG and traditional RAG.

    Args:
        query (str): User query
        self_rag_response (str): Response from Self-RAG
        trad_rag_response (str): Response from traditional RAG
        reference (str, optional): Reference answer

    Returns:
        str: Comparison analysis
    """
    system_prompt = """You are an expert evaluator of RAG systems. Your task is to compare responses from two different RAG approaches:
1. Self-RAG: A dynamic approach that decides if retrieval is needed and evaluates information relevance and response quality
2. Traditional RAG: Always retrieves documents and uses them to generate a response

Compare the responses based on:
- Relevance to the query
- Factual correctness
- Completeness and informativeness
- Conciseness and focus"""

    user_prompt = f"""Query: {query}

Response from Self-RAG:
{self_rag_response}

Response from Traditional RAG:
{trad_rag_response}
"""

    if reference:
        user_prompt += f"""
Reference Answer (for factual checking):
{reference}
"""

    user_prompt += """
Compare these responses and explain which one is better and why.
Focus on accuracy, relevance, completeness, and quality.
"""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # Using a stronger model for evaluation
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content
```

```python
def generate_overall_analysis(results):
    """
    Generate an overall analysis of Self-RAG vs traditional RAG.

    Args:
        results (List[Dict]): Results from evaluate_rag_approaches

    Returns:
        str: Overall analysis
    """
    system_prompt = """You are an expert evaluator of RAG systems. Your task is to provide an overall analysis comparing
    Self-RAG and Traditional RAG based on multiple test queries.

    Focus your analysis on:
    1. When Self-RAG performs better and why
    2. When Traditional RAG performs better and why
    3. The impact of dynamic retrieval decisions in Self-RAG
    4. The value of relevance and support evaluation in Self-RAG
    5. Overall recommendations on which approach to use for different types of queries"""

    # Prepare a summary of the individual comparisons
    comparisons_summary = ""
    for i, result in enumerate(results):
        comparisons_summary += f"Query {i+1}: {result['query']}\n"
        comparisons_summary += f"Self-RAG metrics: Retrieval needed: {result['self_rag_metrics']['retrieval_needed']}, "
        comparisons_summary += f"Relevant docs: {result['self_rag_metrics']['relevant_documents']}/{result['self_rag_metrics']['documents_retrieved']}\n"
        comparisons_summary += f"Comparison summary: {result['comparison'][:200]}...\n\n"

        user_prompt = f"""Based on the following comparison results from {len(results)} test queries, please provide an overall analysis of
    Self-RAG versus Traditional RAG:

    {comparisons_summary}

    Please provide your comprehensive analysis.
    """

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content
```

## Evaluating the Self-RAG System

The final step is to evaluate the Self-RAG system against traditional RAG approaches. We'll compare the quality of responses generated by both systems and analyze the performance of Self-RAG in different scenarios.

```python
# Path to the AI information document
pdf_path = "data/AI_Information.pdf"

# Define test queries covering different query types to test Self-RAG's adaptive retrieval
test_queries = [
    "What are the main ethical concerns in AI development?",        # Document-focused query
    # "How does explainable AI improve trust in AI systems?",         # Document-focused query
    # "Write a poem about artificial intelligence",                   # Creative query, doesn't need retrieval
    # "Will superintelligent AI lead to human obsolescence?"          # Speculative query, partial retrieval needed
]

# Reference answers for more objective evaluation
reference_answers = [
    "The main ethical concerns in AI development include bias and fairness, privacy, transparency, accountability, safety, and the potential for misuse or harmful applications.",
    # "Explainable AI improves trust by making AI decision-making processes transparent and understandable to users, helping them verify fairness, identify potential biases, and better understand AI limitations.",
    # "A quality poem about artificial intelligence should creatively explore themes of AI's capabilities, limitations, relationship with humanity, potential futures, or philosophical questions about consciousness and intelligence.",
    # "Views on superintelligent AI's impact on human relevance vary widely. Some experts warn of potential risks if AI surpasses human capabilities across domains, possibly leading to economic displacement or loss of human agency. Others argue humans will remain relevant through complementary skills, emotional intelligence, and by defining AI's purpose. Most experts agree that thoughtful governance and human-centered design are essential regardless of the outcome."
]

# Run the evaluation comparing Self-RAG with traditional RAG approaches
evaluation_results = evaluate_rag_approaches(
    pdf_path=pdf_path,                  # Source document containing AI information
    test_queries=test_queries,          # List of AI-related test queries
    reference_answers=reference_answers  # Ground truth answers for evaluation
)

# Print the overall comparative analysis
print("\n=== OVERALL ANALYSIS ===\n")
print(evaluation_results["overall_analysis"])
```

```output
=== Evaluating RAG Approaches ===
Extracting text from PDF...
Chunking text...
Created 42 text chunks
Creating embeddings for chunks...
Added 42 chunks to the vector store

Processing query 1: What are the main ethical concerns in AI development?

=== Starting Self-RAG for query: What are the main ethical concerns in AI development? ===

Step 1: Determining if retrieval is necessary...
Retrieval needed: True

Step 2: Retrieving relevant documents...
Retrieved 3 documents

Step 3: Evaluating document relevance...
Document 1 relevance: relevant
Document 2 relevance: relevant
Document 3 relevance: relevant
Found 3 relevant documents

Step 4: Processing relevant contexts...

Processing context 1/3...
Generating response...
Assessing support...
Support rating: fully supported
Rating utility...
Utility rating: 4/5
Overall score: 19
New best response found!

Processing context 2/3...
Generating response...
Assessing support...
Support rating: partially supported
Rating utility...
Utility rating: 4/5
Overall score: 9

Processing context 3/3...
Generating response...
Assessing support...
Support rating: fully supported
Rating utility...
Utility rating: 5/5
Overall score: 20
New best response found!

=== Self-RAG Completed ===

=== Running traditional RAG for query: What are the main ethical concerns in AI development? ===

Retrieving documents...
Retrieved 3 documents
Generating response...

=== OVERALL ANALYSIS ===

**Overall Analysis: Self-RAG vs Traditional RAG**

Based on the comparison results from the test query "What are the main ethical concerns in AI development?", I will provide a comprehensive analysis of the strengths and weaknesses of both Self-RAG and Traditional RAG systems.

**When Self-RAG performs better:**

1. **Dynamic retrieval decisions**: Self-RAG's ability to dynamically adjust its retrieval decisions based on the query context and user feedback can lead to better results in complex queries with multiple relevant documents. In the case of Query 1, Self-RAG's retrieval needed was True, indicating that it was able to identify the most relevant documents for the query. This suggests that Self-RAG's dynamic retrieval decisions were effective in this scenario.
2. **Relevance and support evaluation**: Self-RAG's evaluation of relevance and support can lead to more accurate and informative responses. In this case, Self-RAG's relevant docs were 3/3, indicating that it was able to identify the most relevant documents for the query. This suggests that Self-RAG's evaluation of relevance and support was effective in this scenario.

**When Traditional RAG performs better:**

1. **Simple queries**: Traditional RAG may perform better in simple queries with a single relevant document. In this case, the query "What are the main ethical concerns in AI development?" may have been too complex for Traditional RAG to handle effectively.
2. **Pre-defined ranking**: Traditional RAG's pre-defined ranking may be more effective in scenarios where the ranking of documents is not critical. In this case, the query "What are the main ethical concerns in AI development?" may not have required a highly ranked response.

**The impact of dynamic retrieval decisions in Self-RAG:**

Self-RAG's dynamic retrieval decisions can lead to better results in complex queries with multiple relevant documents. However, this may also lead to over-retrieval or under-retrieval of documents, depending on the query context and user feedback. To mitigate this, Self-RAG's dynamic retrieval decisions should be carefully tuned to ensure that the most relevant documents are retrieved.

**The value of relevance and support evaluation in Self-RAG:**

Self-RAG's evaluation of relevance and support is critical in ensuring that the retrieved documents are accurate and informative. By evaluating the relevance and support of each document, Self-RAG can provide more accurate and informative responses. However, this evaluation should be carefully tuned to ensure that the most relevant documents are retrieved.

**Overall recommendations:**

1. **Use Self-RAG for complex queries**: Self-RAG's dynamic retrieval decisions and evaluation of relevance and support make it a better choice for complex queries with multiple relevant documents.
2. **Use Traditional RAG for simple queries**: Traditional RAG's pre-defined ranking and simplicity make it a better choice for simple queries with a single relevant document.
3. **Tune Self-RAG's dynamic retrieval decisions**: Self-RAG's dynamic retrieval decisions should be carefully tuned to ensure that the most relevant documents are retrieved.
4. **Evaluate relevance and support in Self-RAG**: Self-RAG's evaluation of relevance and support is critical in ensuring that the retrieved documents are accurate and informative.

In conclusion, Self-RAG and Traditional RAG have different strengths and weaknesses, and the choice of which system to use depends on the type of query and the desired outcome. By understanding the strengths and weaknesses of each system, we can make informed decisions about which system to use in different scenarios.
```
