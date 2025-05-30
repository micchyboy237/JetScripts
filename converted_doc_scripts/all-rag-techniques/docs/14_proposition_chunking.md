# Proposition Chunking for Enhanced RAG

In this notebook, I implement proposition chunking - an advanced technique to break down documents into atomic, factual statements for more accurate retrieval. Unlike traditional chunking that simply divides text by character count, proposition chunking preserves the semantic integrity of individual facts.

Proposition chunking delivers more precise retrieval by:

1. Breaking content into atomic, self-contained facts
2. Creating smaller, more granular units for retrieval
3. Enabling more precise matching between queries and relevant content
4. Filtering out low-quality or incomplete propositions

Let's build a complete implementation without relying on LangChain or FAISS.

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
def chunk_text(text, chunk_size=800, overlap=100):
    """
    Split text into overlapping chunks.

    Args:
        text (str): Input text to chunk
        chunk_size (int): Size of each chunk in characters
        overlap (int): Overlap between chunks in characters

    Returns:
        List[Dict]: List of chunk dictionaries with text and metadata
    """
    chunks = []  # Initialize an empty list to store the chunks

    # Iterate over the text with the specified chunk size and overlap
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]  # Extract a chunk of the specified size
        if chunk:  # Ensure we don't add empty chunks
            chunks.append({
                "text": chunk,  # The chunk text
                "chunk_id": len(chunks) + 1,  # Unique ID for the chunk
                "start_char": i,  # Starting character index of the chunk
                "end_char": i + len(chunk)  # Ending character index of the chunk
            })

    print(f"Created {len(chunks)} text chunks")  # Print the number of created chunks
    return chunks  # Return the list of chunks
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
        # Initialize lists to store vectors, texts, and metadata
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add_item(self, text, embedding, metadata=None):
        """
        Add an item to the vector store.

        Args:
            text (str): The text content
            embedding (List[float]): The embedding vector
            metadata (Dict, optional): Additional metadata
        """
        # Append the embedding, text, and metadata to their respective lists
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})

    def add_items(self, texts, embeddings, metadata_list=None):
        """
        Add multiple items to the vector store.

        Args:
            texts (List[str]): List of text contents
            embeddings (List[List[float]]): List of embedding vectors
            metadata_list (List[Dict], optional): List of metadata dictionaries
        """
        # If no metadata list is provided, create an empty dictionary for each text
        if metadata_list is None:
            metadata_list = [{} for _ in range(len(texts))]

        # Add each text, embedding, and metadata to the store
        for text, embedding, metadata in zip(texts, embeddings, metadata_list):
            self.add_item(text, embedding, metadata)

    def similarity_search(self, query_embedding, k=5):
        """
        Find the most similar items to a query embedding.

        Args:
            query_embedding (List[float]): Query embedding vector
            k (int): Number of results to return

        Returns:
            List[Dict]: Top k most similar items
        """
        # Return an empty list if there are no vectors in the store
        if not self.vectors:
            return []

        # Convert query embedding to a numpy array
        query_vector = np.array(query_embedding)

        # Calculate similarities using cosine similarity
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))

        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Collect the top k results
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": float(score)  # Convert to float for JSON serialization
            })

        return results
```

## Creating Embeddings

```python
def create_embeddings(texts, model="BAAI/bge-en-icl"):
    """
    Create embeddings for the given texts.

    Args:
        texts (str or List[str]): Input text(s)
        model (str): Embedding model name

    Returns:
        List[List[float]]: Embedding vector(s)
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

    # If input was a single string, return just the first embedding
    if isinstance(texts, str):
        return all_embeddings[0]

    # Otherwise, return all embeddings
    return all_embeddings
```

## Proposition Generation

```python
def generate_propositions(chunk):
    """
    Generate atomic, self-contained propositions from a text chunk.

    Args:
        chunk (Dict): Text chunk with content and metadata

    Returns:
        List[str]: List of generated propositions
    """
    # System prompt to instruct the AI on how to generate propositions
    system_prompt = """Please break down the following text into simple, self-contained propositions.
    Ensure that each proposition meets the following criteria:

    1. Express a Single Fact: Each proposition should state one specific fact or claim.
    2. Be Understandable Without Context: The proposition should be self-contained, meaning it can be understood without needing additional context.
    3. Use Full Names, Not Pronouns: Avoid pronouns or ambiguous references; use full entity names.
    4. Include Relevant Dates/Qualifiers: If applicable, include necessary dates, times, and qualifiers to make the fact precise.
    5. Contain One Subject-Predicate Relationship: Focus on a single subject and its corresponding action or attribute, without conjunctions or multiple clauses.

    Output ONLY the list of propositions without any additional text or explanations."""

    # User prompt containing the text chunk to be converted into propositions
    user_prompt = f"Text to convert into propositions:\n\n{chunk['text']}"

    # Generate response from the model
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # Using a stronger model for accurate proposition generation
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # Extract propositions from the response
    raw_propositions = response.choices[0].message.content.strip().split('\n')

    # Clean up propositions (remove numbering, bullets, etc.)
    clean_propositions = []
    for prop in raw_propositions:
        # Remove numbering (1., 2., etc.) and bullet points
        cleaned = re.sub(r'^\s*(\d+\.|\-|\*)\s*', '', prop).strip()
        if cleaned and len(cleaned) > 10:  # Simple filter for empty or very short propositions
            clean_propositions.append(cleaned)

    return clean_propositions
```

## Quality Checking for Propositions

```python
def evaluate_proposition(proposition, original_text):
    """
    Evaluate a proposition's quality based on accuracy, clarity, completeness, and conciseness.

    Args:
        proposition (str): The proposition to evaluate
        original_text (str): The original text for comparison

    Returns:
        Dict: Scores for each evaluation dimension
    """
    # System prompt to instruct the AI on how to evaluate the proposition
    system_prompt = """You are an expert at evaluating the quality of propositions extracted from text.
    Rate the given proposition on the following criteria (scale 1-10):

    - Accuracy: How well the proposition reflects information in the original text
    - Clarity: How easy it is to understand the proposition without additional context
    - Completeness: Whether the proposition includes necessary details (dates, qualifiers, etc.)
    - Conciseness: Whether the proposition is concise without losing important information

    The response must be in valid JSON format with numerical scores for each criterion:
    {"accuracy": X, "clarity": X, "completeness": X, "conciseness": X}
    """

    # User prompt containing the proposition and the original text
    user_prompt = f"""Proposition: {proposition}

    Original Text: {original_text}

    Please provide your evaluation scores in JSON format."""

    # Generate response from the model
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        # response_format={"type": "json_object"},
        temperature=0
    )

    # Parse the JSON response
    try:
        scores = json.loads(response.choices[0].message.content.strip())
        return scores
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return {
            "accuracy": 5,
            "clarity": 5,
            "completeness": 5,
            "conciseness": 5
        }
```

## Complete Proposition Processing Pipeline

```python
def process_document_into_propositions(pdf_path, chunk_size=800, chunk_overlap=100,
                                      quality_thresholds=None):
    """
    Process a document into quality-checked propositions.

    Args:
        pdf_path (str): Path to the PDF file
        chunk_size (int): Size of each chunk in characters
        chunk_overlap (int): Overlap between chunks in characters
        quality_thresholds (Dict): Threshold scores for proposition quality

    Returns:
        Tuple[List[Dict], List[Dict]]: Original chunks and proposition chunks
    """
    # Set default quality thresholds if not provided
    if quality_thresholds is None:
        quality_thresholds = {
            "accuracy": 7,
            "clarity": 7,
            "completeness": 7,
            "conciseness": 7
        }

    # Extract text from the PDF file
    text = extract_text_from_pdf(pdf_path)

    # Create chunks from the extracted text
    chunks = chunk_text(text, chunk_size, chunk_overlap)

    # Initialize a list to store all propositions
    all_propositions = []

    print("Generating propositions from chunks...")
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")

        # Generate propositions for the current chunk
        chunk_propositions = generate_propositions(chunk)
        print(f"Generated {len(chunk_propositions)} propositions")

        # Process each generated proposition
        for prop in chunk_propositions:
            proposition_data = {
                "text": prop,
                "source_chunk_id": chunk["chunk_id"],
                "source_text": chunk["text"]
            }
            all_propositions.append(proposition_data)

    # Evaluate the quality of the generated propositions
    print("\nEvaluating proposition quality...")
    quality_propositions = []

    for i, prop in enumerate(all_propositions):
        if i % 10 == 0:  # Status update every 10 propositions
            print(f"Evaluating proposition {i+1}/{len(all_propositions)}...")

        # Evaluate the quality of the current proposition
        scores = evaluate_proposition(prop["text"], prop["source_text"])
        prop["quality_scores"] = scores

        # Check if the proposition passes the quality thresholds
        passes_quality = True
        for metric, threshold in quality_thresholds.items():
            if scores.get(metric, 0) < threshold:
                passes_quality = False
                break

        if passes_quality:
            quality_propositions.append(prop)
        else:
            print(f"Proposition failed quality check: {prop['text'][:50]}...")

    print(f"\nRetained {len(quality_propositions)}/{len(all_propositions)} propositions after quality filtering")

    return chunks, quality_propositions
```

## Building Vector Stores for Both Approaches

```python
def build_vector_stores(chunks, propositions):
    """
    Build vector stores for both chunk-based and proposition-based approaches.

    Args:
        chunks (List[Dict]): Original document chunks
        propositions (List[Dict]): Quality-filtered propositions

    Returns:
        Tuple[SimpleVectorStore, SimpleVectorStore]: Chunk and proposition vector stores
    """
    # Create vector store for chunks
    chunk_store = SimpleVectorStore()

    # Extract chunk texts and create embeddings
    chunk_texts = [chunk["text"] for chunk in chunks]
    print(f"Creating embeddings for {len(chunk_texts)} chunks...")
    chunk_embeddings = create_embeddings(chunk_texts)

    # Add chunks to vector store with metadata
    chunk_metadata = [{"chunk_id": chunk["chunk_id"], "type": "chunk"} for chunk in chunks]
    chunk_store.add_items(chunk_texts, chunk_embeddings, chunk_metadata)

    # Create vector store for propositions
    prop_store = SimpleVectorStore()

    # Extract proposition texts and create embeddings
    prop_texts = [prop["text"] for prop in propositions]
    print(f"Creating embeddings for {len(prop_texts)} propositions...")
    prop_embeddings = create_embeddings(prop_texts)

    # Add propositions to vector store with metadata
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
```

## Query and Retrieval Functions

```python
def retrieve_from_store(query, vector_store, k=5):
    """
    Retrieve relevant items from a vector store based on query.

    Args:
        query (str): User query
        vector_store (SimpleVectorStore): Vector store to search
        k (int): Number of results to retrieve

    Returns:
        List[Dict]: Retrieved items with scores and metadata
    """
    # Create query embedding
    query_embedding = create_embeddings(query)

    # Search vector store for the top k most similar items
    results = vector_store.similarity_search(query_embedding, k=k)

    return results
```

```python
def compare_retrieval_approaches(query, chunk_store, prop_store, k=5):
    """
    Compare chunk-based and proposition-based retrieval for a query.

    Args:
        query (str): User query
        chunk_store (SimpleVectorStore): Chunk-based vector store
        prop_store (SimpleVectorStore): Proposition-based vector store
        k (int): Number of results to retrieve from each store

    Returns:
        Dict: Comparison results
    """
    print(f"\n=== Query: {query} ===")

    # Retrieve results from the proposition-based vector store
    print("\nRetrieving with proposition-based approach...")
    prop_results = retrieve_from_store(query, prop_store, k)

    # Retrieve results from the chunk-based vector store
    print("Retrieving with chunk-based approach...")
    chunk_results = retrieve_from_store(query, chunk_store, k)

    # Display proposition-based results
    print("\n=== Proposition-Based Results ===")
    for i, result in enumerate(prop_results):
        print(f"{i+1}) {result['text']} (Score: {result['similarity']:.4f})")

    # Display chunk-based results
    print("\n=== Chunk-Based Results ===")
    for i, result in enumerate(chunk_results):
        # Truncate text to keep the output manageable
        truncated_text = result['text'][:150] + "..." if len(result['text']) > 150 else result['text']
        print(f"{i+1}) {truncated_text} (Score: {result['similarity']:.4f})")

    # Return the comparison results
    return {
        "query": query,
        "proposition_results": prop_results,
        "chunk_results": chunk_results
    }
```

## Response Generation and Evaluation

```python
def generate_response(query, results, result_type="proposition"):
    """
    Generate a response based on retrieved results.

    Args:
        query (str): User query
        results (List[Dict]): Retrieved items
        result_type (str): Type of results ('proposition' or 'chunk')

    Returns:
        str: Generated response
    """
    # Combine retrieved texts into a single context string
    context = "\n\n".join([result["text"] for result in results])

    # System prompt to instruct the AI on how to generate the response
    system_prompt = f"""You are an AI assistant answering questions based on retrieved information.
Your answer should be based on the following {result_type}s that were retrieved from a knowledge base.
If the retrieved information doesn't answer the question, acknowledge this limitation."""

    # User prompt containing the query and the retrieved context
    user_prompt = f"""Query: {query}

Retrieved {result_type}s:
{context}

Please answer the query based on the retrieved information."""

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
    return response.choices[0].message.content
```

```python
def evaluate_responses(query, prop_response, chunk_response, reference_answer=None):
    """
    Evaluate and compare responses from both approaches.

    Args:
        query (str): User query
        prop_response (str): Response from proposition-based approach
        chunk_response (str): Response from chunk-based approach
        reference_answer (str, optional): Reference answer for comparison

    Returns:
        str: Evaluation analysis
    """
    # System prompt to instruct the AI on how to evaluate the responses
    system_prompt = """You are an expert evaluator of information retrieval systems.
    Compare the two responses to the same query, one generated from proposition-based retrieval
    and the other from chunk-based retrieval.

    Evaluate them based on:
    1. Accuracy: Which response provides more factually correct information?
    2. Relevance: Which response better addresses the specific query?
    3. Conciseness: Which response is more concise while maintaining completeness?
    4. Clarity: Which response is easier to understand?

    Be specific about the strengths and weaknesses of each approach."""

    # User prompt containing the query and the responses to be compared
    user_prompt = f"""Query: {query}

    Response from Proposition-Based Retrieval:
    {prop_response}

    Response from Chunk-Based Retrieval:
    {chunk_response}"""

    # If a reference answer is provided, include it in the user prompt for factual checking
    if reference_answer:
        user_prompt += f"""

    Reference Answer (for factual checking):
    {reference_answer}"""

    # Add the final instruction to the user prompt
    user_prompt += """
    Please provide a detailed comparison of these two responses, highlighting which approach performed better and why."""

    # Generate the evaluation analysis using the OpenAI client
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # Return the generated evaluation analysis
    return response.choices[0].message.content
```

## Complete End-to-End Evaluation Pipeline

```python
def run_proposition_chunking_evaluation(pdf_path, test_queries, reference_answers=None):
    """
    Run a complete evaluation of proposition chunking vs standard chunking.

    Args:
        pdf_path (str): Path to the PDF file
        test_queries (List[str]): List of test queries
        reference_answers (List[str], optional): Reference answers for queries

    Returns:
        Dict: Evaluation results
    """
    print("=== Starting Proposition Chunking Evaluation ===\n")

    # Process document into propositions and chunks
    chunks, propositions = process_document_into_propositions(pdf_path)

    # Build vector stores for chunks and propositions
    chunk_store, prop_store = build_vector_stores(chunks, propositions)

    # Initialize a list to store results for each query
    results = []

    # Run tests for each query
    for i, query in enumerate(test_queries):
        print(f"\n\n=== Testing Query {i+1}/{len(test_queries)} ===")
        print(f"Query: {query}")

        # Get retrieval results from both chunk-based and proposition-based approaches
        retrieval_results = compare_retrieval_approaches(query, chunk_store, prop_store)

        # Generate responses based on the retrieved proposition-based results
        print("\nGenerating response from proposition-based results...")
        prop_response = generate_response(
            query,
            retrieval_results["proposition_results"],
            "proposition"
        )

        # Generate responses based on the retrieved chunk-based results
        print("Generating response from chunk-based results...")
        chunk_response = generate_response(
            query,
            retrieval_results["chunk_results"],
            "chunk"
        )

        # Get reference answer if available
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]

        # Evaluate the generated responses
        print("\nEvaluating responses...")
        evaluation = evaluate_responses(query, prop_response, chunk_response, reference)

        # Compile results for the current query
        query_result = {
            "query": query,
            "proposition_results": retrieval_results["proposition_results"],
            "chunk_results": retrieval_results["chunk_results"],
            "proposition_response": prop_response,
            "chunk_response": chunk_response,
            "reference_answer": reference,
            "evaluation": evaluation
        }

        # Append the results to the overall results list
        results.append(query_result)

        # Print the responses and evaluation for the current query
        print("\n=== Proposition-Based Response ===")
        print(prop_response)

        print("\n=== Chunk-Based Response ===")
        print(chunk_response)

        print("\n=== Evaluation ===")
        print(evaluation)

    # Generate overall analysis of the evaluation
    print("\n\n=== Generating Overall Analysis ===")
    overall_analysis = generate_overall_analysis(results)
    print("\n" + overall_analysis)

    # Return the evaluation results, overall analysis, and counts of propositions and chunks
    return {
        "results": results,
        "overall_analysis": overall_analysis,
        "proposition_count": len(propositions),
        "chunk_count": len(chunks)
    }
```

```python
def generate_overall_analysis(results):
    """
    Generate an overall analysis of proposition vs chunk approaches.

    Args:
        results (List[Dict]): Results from each test query

    Returns:
        str: Overall analysis
    """
    # System prompt to instruct the AI on how to generate the overall analysis
    system_prompt = """You are an expert at evaluating information retrieval systems.
    Based on multiple test queries, provide an overall analysis comparing proposition-based retrieval
    to chunk-based retrieval for RAG (Retrieval-Augmented Generation) systems.

    Focus on:
    1. When proposition-based retrieval performs better
    2. When chunk-based retrieval performs better
    3. The overall strengths and weaknesses of each approach
    4. Recommendations for when to use each approach"""

    # Create a summary of evaluations for each query
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        evaluations_summary += f"Evaluation Summary: {result['evaluation'][:200]}...\n\n"

    # User prompt containing the summary of evaluations
    user_prompt = f"""Based on the following evaluations of proposition-based vs chunk-based retrieval across {len(results)} queries,
    provide an overall analysis comparing these two approaches:

    {evaluations_summary}

    Please provide a comprehensive analysis on the relative strengths and weaknesses of proposition-based
    and chunk-based retrieval for RAG systems."""

    # Generate the overall analysis using the OpenAI client
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # Return the generated analysis text
    return response.choices[0].message.content
```

## Evaluation of Proposition Chunking

```python
# Path to the AI information document that will be processed
pdf_path = "data/AI_Information.pdf"

# Define test queries covering different aspects of AI to evaluate proposition chunking
test_queries = [
    "What are the main ethical concerns in AI development?",
    # "How does explainable AI improve trust in AI systems?",
    # "What are the key challenges in developing fair AI systems?",
    # "What role does human oversight play in AI safety?"
]

# Reference answers for more thorough evaluation and comparison of results
# These provide a ground truth to measure the quality of generated responses
reference_answers = [
    "The main ethical concerns in AI development include bias and fairness, privacy, transparency, accountability, safety, and the potential for misuse or harmful applications.",
    # "Explainable AI improves trust by making AI decision-making processes transparent and understandable to users, helping them verify fairness, identify potential biases, and better understand AI limitations.",
    # "Key challenges in developing fair AI systems include addressing data bias, ensuring diverse representation in training data, creating transparent algorithms, defining fairness across different contexts, and balancing competing fairness criteria.",
    # "Human oversight plays a critical role in AI safety by monitoring system behavior, verifying outputs, intervening when necessary, setting ethical boundaries, and ensuring AI systems remain aligned with human values and intentions throughout their operation."
]

# Run the evaluation
evaluation_results = run_proposition_chunking_evaluation(
    pdf_path=pdf_path,
    test_queries=test_queries,
    reference_answers=reference_answers
)

# Print the overall analysis
print("\n\n=== Overall Analysis ===")
print(evaluation_results["overall_analysis"])
```
