# Corrective RAG (CRAG) Implementation

In this notebook, I implement Corrective RAG - an advanced approach that dynamically evaluates retrieved information and corrects the retrieval process when necessary, using web search as a fallback.

CRAG improves on traditional RAG by:

- Evaluating retrieved content before using it
- Dynamically switching between knowledge sources based on relevance
- Correcting the retrieval with web search when local knowledge is insufficient
- Combining information from multiple sources when appropriate

## Setting Up the Environment

We begin by importing necessary libraries.

```python
import os
import numpy as np
import json
import fitz  # PyMuPDF
from openai import OpenAI
import requests
from typing import List, Dict, Tuple, Any
import re
from urllib.parse import quote_plus
import time
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
    print(f"Extracting text from {pdf_path}...")

    # Open the PDF file
    pdf = fitz.open(pdf_path)
    text = ""

    # Iterate through each page in the PDF
    for page_num in range(len(pdf)):
        page = pdf[page_num]
        # Extract text from the current page and append it to the text variable
        text += page.get_text()

    return text
```

```python
def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Split text into overlapping chunks for efficient retrieval and processing.

    This function divides a large text into smaller, manageable chunks with
    specified overlap between consecutive chunks. Chunking is critical for RAG
    systems as it allows for more precise retrieval of relevant information.

    Args:
        text (str): Input text to be chunked
        chunk_size (int): Maximum size of each chunk in characters
        overlap (int): Number of overlapping characters between consecutive chunks
                       to maintain context across chunk boundaries

    Returns:
        List[Dict]: List of text chunks, each containing:
                   - text: The chunk content
                   - metadata: Dictionary with positional information and source type
    """
    chunks = []

    # Iterate through the text with a sliding window approach
    # Moving by (chunk_size - overlap) ensures proper overlap between chunks
    for i in range(0, len(text), chunk_size - overlap):
        # Extract the current chunk, limited by chunk_size
        chunk_text = text[i:i + chunk_size]

        # Only add non-empty chunks
        if chunk_text:
            chunks.append({
                "text": chunk_text,  # The actual text content
                "metadata": {
                    "start_pos": i,  # Starting position in the original text
                    "end_pos": i + len(chunk_text),  # Ending position
                    "source_type": "document"  # Indicates the source of this text
                }
            })

    print(f"Created {len(chunks)} text chunks")
    return chunks
```

## Simple Vector Store Implementation

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

    def add_items(self, items, embeddings):
        """
        Add multiple items to the vector store.

        Args:
            items (List[Dict]): List of items with text and metadata
            embeddings (List[List[float]]): List of embedding vectors
        """
        # Iterate over items and embeddings and add them to the store
        for i, (item, embedding) in enumerate(zip(items, embeddings)):
            self.add_item(
                text=item["text"],
                embedding=embedding,
                metadata=item.get("metadata", {})
            )

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

        # Convert query embedding to numpy array
        query_vector = np.array(query_embedding)

        # Calculate similarities using cosine similarity
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top k results
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": float(score)
            })

        return results
```

## Creating Embeddings

```python
def create_embeddings(texts, model="text-embedding-3-small"):
    """
    Create vector embeddings for text inputs using OpenAI's embedding models.

    Embeddings are dense vector representations of text that capture semantic meaning,
    allowing for similarity comparisons. In RAG systems, embeddings are essential
    for matching queries with relevant document chunks.

    Args:
        texts (str or List[str]): Input text(s) to be embedded. Can be a single string
                                  or a list of strings.
        model (str): The embedding model name to use. Defaults to "text-embedding-3-small".

    Returns:
        List[List[float]]: If input is a list, returns a list of embedding vectors.
                          If input is a single string, returns a single embedding vector.
    """
    # Handle both single string and list inputs by converting single strings to a list
    input_texts = texts if isinstance(texts, list) else [texts]

    # Process in batches to avoid API rate limits and payload size restrictions
    # OpenAI API typically has limits on request size and rate
    batch_size = 100
    all_embeddings = []

    # Process each batch of texts
    for i in range(0, len(input_texts), batch_size):
        # Extract the current batch of texts
        batch = input_texts[i:i + batch_size]

        # Make API call to generate embeddings for the current batch
        response = client.embeddings.create(
            model=model,
            input=batch
        )

        # Extract the embedding vectors from the response
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    # If the original input was a single string, return just the first embedding
    if isinstance(texts, str):
        return all_embeddings[0]

    # Otherwise return the full list of embeddings
    return all_embeddings
```

## Document Processing Pipeline

```python
def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    Process a document into a vector store.

    Args:
        pdf_path (str): Path to the PDF file
        chunk_size (int): Size of each chunk in characters
        chunk_overlap (int): Overlap between chunks in characters

    Returns:
        SimpleVectorStore: Vector store containing document chunks
    """
    # Extract text from the PDF file
    text = extract_text_from_pdf(pdf_path)

    # Split the extracted text into chunks with specified size and overlap
    chunks = chunk_text(text, chunk_size, chunk_overlap)

    # Create embeddings for each chunk of text
    print("Creating embeddings for chunks...")
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = create_embeddings(chunk_texts)

    # Initialize a new vector store
    vector_store = SimpleVectorStore()

    # Add the chunks and their embeddings to the vector store
    vector_store.add_items(chunks, chunk_embeddings)

    print(f"Vector store created with {len(chunks)} chunks")
    return vector_store
```

## Relevance Evaluation Function

```python
def evaluate_document_relevance(query, document):
    """
    Evaluate the relevance of a document to a query.

    Args:
        query (str): User query
        document (str): Document text

    Returns:
        float: Relevance score (0-1)
    """
    # Define the system prompt to instruct the model on how to evaluate relevance
    system_prompt = """
    You are an expert at evaluating document relevance.
    Rate how relevant the given document is to the query on a scale from 0 to 1.
    0 means completely irrelevant, 1 means perfectly relevant.
    Provide ONLY the score as a float between 0 and 1.
    """

    # Define the user prompt with the query and document
    user_prompt = f"Query: {query}\n\nDocument: {document}"

    try:
        # Make a request to the OpenAI API to evaluate the relevance
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Specify the model to use
            messages=[
                {"role": "system", "content": system_prompt},  # System message to guide the assistant
                {"role": "user", "content": user_prompt}  # User message with the query and document
            ],
            temperature=0,  # Set the temperature for response generation
            max_tokens=5  # Very short response needed
        )

        # Extract the score from the response
        score_text = response.choices[0].message.content.strip()
        # Use regex to find the float value in the response
        score_match = re.search(r'(\d+(\.\d+)?)', score_text)
        if score_match:
            return float(score_match.group(1))  # Return the extracted score as a float
        return 0.5  # Default to middle value if parsing fails

    except Exception as e:
        # Print the error message and return a default value on error
        print(f"Error evaluating document relevance: {e}")
        return 0.5  # Default to middle value on error
```

## Web Search Function

```python
def duck_duck_go_search(query, num_results=3):
    """
    Perform a web search using DuckDuckGo.

    Args:
        query (str): Search query
        num_results (int): Number of results to return

    Returns:
        Tuple[str, List[Dict]]: Combined search results text and source metadata
    """
    # Encode the query for URL
    encoded_query = quote_plus(query)

    # DuckDuckGo search API endpoint (unofficial)
    url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json"

    try:
        # Perform the web search request
        response = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        data = response.json()

        # Initialize variables to store results text and sources
        results_text = ""
        sources = []

        # Add abstract if available
        if data.get("AbstractText"):
            results_text += f"{data['AbstractText']}\n\n"
            sources.append({
                "title": data.get("AbstractSource", "Wikipedia"),
                "url": data.get("AbstractURL", "")
            })

        # Add related topics
        for topic in data.get("RelatedTopics", [])[:num_results]:
            if "Text" in topic and "FirstURL" in topic:
                results_text += f"{topic['Text']}\n\n"
                sources.append({
                    "title": topic.get("Text", "").split(" - ")[0],
                    "url": topic.get("FirstURL", "")
                })

        return results_text, sources

    except Exception as e:
        # Print error message if the main search fails
        print(f"Error performing web search: {e}")

        # Fallback to a backup search API
        try:
            backup_url = f"https://serpapi.com/search.json?q={encoded_query}&engine=duckduckgo"
            response = requests.get(backup_url)
            data = response.json()

            # Initialize variables to store results text and sources
            results_text = ""
            sources = []

            # Extract results from the backup API
            for result in data.get("organic_results", [])[:num_results]:
                results_text += f"{result.get('title', '')}: {result.get('snippet', '')}\n\n"
                sources.append({
                    "title": result.get("title", ""),
                    "url": result.get("link", "")
                })

            return results_text, sources
        except Exception as backup_error:
            # Print error message if the backup search also fails
            print(f"Backup search also failed: {backup_error}")
            return "Failed to retrieve search results.", []
```

```python
def rewrite_search_query(query):
    """
    Rewrite a query to be more suitable for web search.

    Args:
        query (str): Original query

    Returns:
        str: Rewritten query
    """
    # Define the system prompt to instruct the model on how to rewrite the query
    system_prompt = """
    You are an expert at creating effective search queries.
    Rewrite the given query to make it more suitable for a web search engine.
    Focus on keywords and facts, remove unnecessary words, and make it concise.
    """

    try:
        # Make a request to the OpenAI API to rewrite the query
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Specify the model to use
            messages=[
                {"role": "system", "content": system_prompt},  # System message to guide the assistant
                {"role": "user", "content": f"Original query: {query}\n\nRewritten query:"}  # User message with the original query
            ],
            temperature=0.3,  # Set the temperature for response generation
            max_tokens=50  # Limit the response length
        )

        # Return the rewritten query from the response
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Print the error message and return the original query on error
        print(f"Error rewriting search query: {e}")
        return query  # Return original query on error
```

```python
def perform_web_search(query):
    """
    Perform web search with query rewriting.

    Args:
        query (str): Original user query

    Returns:
        Tuple[str, List[Dict]]: Search results text and source metadata
    """
    # Rewrite the query to improve search results
    rewritten_query = rewrite_search_query(query)
    print(f"Rewritten search query: {rewritten_query}")

    # Perform the web search using the rewritten query
    results_text, sources = duck_duck_go_search(rewritten_query)

    # Return the search results text and source metadata
    return results_text, sources
```

## Knowledge Refinement Function

```python
def refine_knowledge(text):
    """
    Extract and refine key information from text.

    Args:
        text (str): Input text to refine

    Returns:
        str: Refined key points from the text
    """
    # Define the system prompt to instruct the model on how to extract key information
    system_prompt = """
    Extract the key information from the following text as a set of clear, concise bullet points.
    Focus on the most relevant facts and important details.
    Format your response as a bulleted list with each point on a new line starting with "• ".
    """

    try:
        # Make a request to the OpenAI API to refine the text
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Specify the model to use
            messages=[
                {"role": "system", "content": system_prompt},  # System message to guide the assistant
                {"role": "user", "content": f"Text to refine:\n\n{text}"}  # User message with the text to refine
            ],
            temperature=0.3  # Set the temperature for response generation
        )

        # Return the refined key points from the response
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Print the error message and return the original text on error
        print(f"Error refining knowledge: {e}")
        return text  # Return original text on error
```

## Core CRAG Process

```python
def crag_process(query, vector_store, k=3):
    """
    Run the Corrective RAG process.

    Args:
        query (str): User query
        vector_store (SimpleVectorStore): Vector store with document chunks
        k (int): Number of initial documents to retrieve

    Returns:
        Dict: Process results including response and debug info
    """
    print(f"\n=== Processing query with CRAG: {query} ===\n")

    # Step 1: Create query embedding and retrieve documents
    print("Retrieving initial documents...")
    query_embedding = create_embeddings(query)
    retrieved_docs = vector_store.similarity_search(query_embedding, k=k)

    # Step 2: Evaluate document relevance
    print("Evaluating document relevance...")
    relevance_scores = []
    for doc in retrieved_docs:
        score = evaluate_document_relevance(query, doc["text"])
        relevance_scores.append(score)
        doc["relevance"] = score
        print(f"Document scored {score:.2f} relevance")

    # Step 3: Determine action based on best relevance score
    max_score = max(relevance_scores) if relevance_scores else 0
    best_doc_idx = relevance_scores.index(max_score) if relevance_scores else -1

    # Track sources for attribution
    sources = []
    final_knowledge = ""

    # Step 4: Execute the appropriate knowledge acquisition strategy
    if max_score > 0.7:
        # Case 1: High relevance - Use document directly
        print(f"High relevance ({max_score:.2f}) - Using document directly")
        best_doc = retrieved_docs[best_doc_idx]["text"]
        final_knowledge = best_doc
        sources.append({
            "title": "Document",
            "url": ""
        })

    elif max_score < 0.3:
        # Case 2: Low relevance - Use web search
        print(f"Low relevance ({max_score:.2f}) - Performing web search")
        web_results, web_sources = perform_web_search(query)
        final_knowledge = refine_knowledge(web_results)
        sources.extend(web_sources)

    else:
        # Case 3: Medium relevance - Combine document with web search
        print(f"Medium relevance ({max_score:.2f}) - Combining document with web search")
        best_doc = retrieved_docs[best_doc_idx]["text"]
        refined_doc = refine_knowledge(best_doc)

        # Get web results
        web_results, web_sources = perform_web_search(query)
        refined_web = refine_knowledge(web_results)

        # Combine knowledge
        final_knowledge = f"From document:\n{refined_doc}\n\nFrom web search:\n{refined_web}"

        # Add sources
        sources.append({
            "title": "Document",
            "url": ""
        })
        sources.extend(web_sources)

    # Step 5: Generate final response
    print("Generating final response...")
    response = generate_response(query, final_knowledge, sources)

    # Return comprehensive results
    return {
        "query": query,
        "response": response,
        "retrieved_docs": retrieved_docs,
        "relevance_scores": relevance_scores,
        "max_relevance": max_score,
        "final_knowledge": final_knowledge,
        "sources": sources
    }
```

## Response Generation

```python
def generate_response(query, knowledge, sources):
    """
    Generate a response based on the query and knowledge.

    Args:
        query (str): User query
        knowledge (str): Knowledge to base the response on
        sources (List[Dict]): List of sources with title and URL

    Returns:
        str: Generated response
    """
    # Format sources for inclusion in prompt
    sources_text = ""
    for source in sources:
        title = source.get("title", "Unknown Source")
        url = source.get("url", "")
        if url:
            sources_text += f"- {title}: {url}\n"
        else:
            sources_text += f"- {title}\n"

    # Define the system prompt to instruct the model on how to generate the response
    system_prompt = """
    You are a helpful AI assistant. Generate a comprehensive, informative response to the query based on the provided knowledge.
    Include all relevant information while keeping your answer clear and concise.
    If the knowledge doesn't fully answer the query, acknowledge this limitation.
    Include source attribution at the end of your response.
    """

    # Define the user prompt with the query, knowledge, and sources
    user_prompt = f"""
    Query: {query}

    Knowledge:
    {knowledge}

    Sources:
    {sources_text}

    Please provide an informative response to the query based on this information.
    Include the sources at the end of your response.
    """

    try:
        # Make a request to the OpenAI API to generate the response
        response = client.chat.completions.create(
            model="gpt-4",  # Using GPT-4 for high-quality responses
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )

        # Return the generated response
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Print the error message and return an error response
        print(f"Error generating response: {e}")
        return f"I apologize, but I encountered an error while generating a response to your query: '{query}'. The error was: {str(e)}"
```

## Evaluation Functions

```python
def evaluate_crag_response(query, response, reference_answer=None):
    """
    Evaluate the quality of a CRAG response.

    Args:
        query (str): User query
        response (str): Generated response
        reference_answer (str, optional): Reference answer for comparison

    Returns:
        Dict: Evaluation metrics
    """
    # System prompt for the evaluation criteria
    system_prompt = """
    You are an expert at evaluating the quality of responses to questions.
    Please evaluate the provided response based on the following criteria:

    1. Relevance (0-10): How directly does the response address the query?
    2. Accuracy (0-10): How factually correct is the information?
    3. Completeness (0-10): How thoroughly does the response answer all aspects of the query?
    4. Clarity (0-10): How clear and easy to understand is the response?
    5. Source Quality (0-10): How well does the response cite relevant sources?

    Return your evaluation as a JSON object with scores for each criterion and a brief explanation for each score.
    Also include an "overall_score" (0-10) and a brief "summary" of your evaluation.
    """

    # User prompt with the query and response to be evaluated
    user_prompt = f"""
    Query: {query}

    Response to evaluate:
    {response}
    """

    # Include reference answer in the prompt if provided
    if reference_answer:
        user_prompt += f"""
    Reference answer (for comparison):
    {reference_answer}
    """

    try:
        # Request evaluation from the GPT-4 model
        evaluation_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # response_format={"type": "json_object"},
            temperature=0
        )

        # Parse the evaluation response
        evaluation = json.loads(evaluation_response.choices[0].message.content)
        return evaluation
    except Exception as e:
        # Handle any errors during the evaluation process
        print(f"Error evaluating response: {e}")
        return {
            "error": str(e),
            "overall_score": 0,
            "summary": "Evaluation failed due to an error."
        }
```

```python
def compare_crag_vs_standard_rag(query, vector_store, reference_answer=None):
    """
    Compare CRAG against standard RAG for a query.

    Args:
        query (str): User query
        vector_store (SimpleVectorStore): Vector store with document chunks
        reference_answer (str, optional): Reference answer for comparison

    Returns:
        Dict: Comparison results
    """
    # Run CRAG process
    print("\n=== Running CRAG ===")
    crag_result = crag_process(query, vector_store)
    crag_response = crag_result["response"]

    # Run standard RAG (directly retrieve and respond)
    print("\n=== Running standard RAG ===")
    query_embedding = create_embeddings(query)
    retrieved_docs = vector_store.similarity_search(query_embedding, k=3)
    combined_text = "\n\n".join([doc["text"] for doc in retrieved_docs])
    standard_sources = [{"title": "Document", "url": ""}]
    standard_response = generate_response(query, combined_text, standard_sources)

    # Evaluate both approaches
    print("\n=== Evaluating CRAG response ===")
    crag_eval = evaluate_crag_response(query, crag_response, reference_answer)

    print("\n=== Evaluating standard RAG response ===")
    standard_eval = evaluate_crag_response(query, standard_response, reference_answer)

    # Compare approaches
    print("\n=== Comparing approaches ===")
    comparison = compare_responses(query, crag_response, standard_response, reference_answer)

    return {
        "query": query,
        "crag_response": crag_response,
        "standard_response": standard_response,
        "reference_answer": reference_answer,
        "crag_evaluation": crag_eval,
        "standard_evaluation": standard_eval,
        "comparison": comparison
    }
```

```python
def compare_responses(query, crag_response, standard_response, reference_answer=None):
    """
    Compare CRAG and standard RAG responses.

    Args:
        query (str): User query
        crag_response (str): CRAG response
        standard_response (str): Standard RAG response
        reference_answer (str, optional): Reference answer

    Returns:
        str: Comparison analysis
    """
    # System prompt for comparing the two approaches
    system_prompt = """
    You are an expert evaluator comparing two response generation approaches:

    1. CRAG (Corrective RAG): A system that evaluates document relevance and dynamically switches to web search when needed.
    2. Standard RAG: A system that directly retrieves documents based on embedding similarity and uses them for response generation.

    Compare the responses from these two systems based on:
    - Accuracy and factual correctness
    - Relevance to the query
    - Completeness of the answer
    - Clarity and organization
    - Source attribution quality

    Explain which approach performed better for this specific query and why.
    """

    # User prompt with the query and responses to be compared
    user_prompt = f"""
    Query: {query}

    CRAG Response:
    {crag_response}

    Standard RAG Response:
    {standard_response}
    """

    # Include reference answer in the prompt if provided
    if reference_answer:
        user_prompt += f"""
    Reference Answer:
    {reference_answer}
    """

    try:
        # Request comparison from the GPT-4 model
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )

        # Return the comparison analysis
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Handle any errors during the comparison process
        print(f"Error comparing responses: {e}")
        return f"Error comparing responses: {str(e)}"
```

## Complete Evaluation Pipeline

```python
def run_crag_evaluation(pdf_path, test_queries, reference_answers=None):
    """
    Run a complete evaluation of CRAG with multiple test queries.

    Args:
        pdf_path (str): Path to the PDF document
        test_queries (List[str]): List of test queries
        reference_answers (List[str], optional): Reference answers for queries

    Returns:
        Dict: Complete evaluation results
    """
    # Process document and create vector store
    vector_store = process_document(pdf_path)

    results = []

    for i, query in enumerate(test_queries):
        print(f"\n\n===== Evaluating Query {i+1}/{len(test_queries)} =====")
        print(f"Query: {query}")

        # Get reference answer if available
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]

        # Run comparison between CRAG and standard RAG
        result = compare_crag_vs_standard_rag(query, vector_store, reference)
        results.append(result)

        # Display comparison results
        print("\n=== Comparison ===")
        print(result["comparison"])

    # Generate overall analysis from individual results
    overall_analysis = generate_overall_analysis(results)

    return {
        "results": results,
        "overall_analysis": overall_analysis
    }
```

```python
def generate_overall_analysis(results):
    """
    Generate an overall analysis of evaluation results.

    Args:
        results (List[Dict]): Results from individual query evaluations

    Returns:
        str: Overall analysis
    """
    # System prompt for the analysis
    system_prompt = """
    You are an expert at evaluating information retrieval and response generation systems.
    Based on multiple test queries, provide an overall analysis comparing CRAG (Corrective RAG)
    with standard RAG.

    Focus on:
    1. When CRAG performs better and why
    2. When standard RAG performs better and why
    3. The overall strengths and weaknesses of each approach
    4. Recommendations for when to use each approach
    """

    # Create summary of evaluations
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        if 'crag_evaluation' in result and 'overall_score' in result['crag_evaluation']:
            crag_score = result['crag_evaluation'].get('overall_score', 'N/A')
            evaluations_summary += f"CRAG score: {crag_score}\n"
        if 'standard_evaluation' in result and 'overall_score' in result['standard_evaluation']:
            std_score = result['standard_evaluation'].get('overall_score', 'N/A')
            evaluations_summary += f"Standard RAG score: {std_score}\n"
        evaluations_summary += f"Comparison summary: {result['comparison'][:200]}...\n\n"

    # User prompt for the analysis
    user_prompt = f"""
    Based on the following evaluations comparing CRAG vs standard RAG across {len(results)} queries,
    provide an overall analysis of these two approaches:

    {evaluations_summary}

    Please provide a comprehensive analysis of the relative strengths and weaknesses of CRAG
    compared to standard RAG, focusing on when and why one approach outperforms the other.
    """

    try:
        # Generate the overall analysis using GPT-4
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating overall analysis: {e}")
        return f"Error generating overall analysis: {str(e)}"
```

## Evaluation of CRAG with Test Queries

```python
# Path to the AI information PDF document
pdf_path = "data/AI_Information.pdf"

# Run comprehensive evaluation with multiple AI-related queries
test_queries = [
    "How does machine learning differ from traditional programming?",
]

# Optional reference answers for better quality evaluation
reference_answers = [
    "Machine learning differs from traditional programming by having computers learn patterns from data rather than following explicit instructions. In traditional programming, developers write specific rules for the computer to follow, while in machine learning",
]

# Run the full evaluation comparing CRAG vs standard RAG
evaluation_results = run_crag_evaluation(pdf_path, test_queries, reference_answers)
print("\n=== Overall Analysis of CRAG vs Standard RAG ===")
print(evaluation_results["overall_analysis"])
```
