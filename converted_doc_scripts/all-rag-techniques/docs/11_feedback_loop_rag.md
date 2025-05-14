# Feedback Loop in RAG

In this notebook, I implement a RAG system with a feedback loop mechanism that continuously improves over time. By collecting and incorporating user feedback, our system learns to provide more relevant and higher-quality responses with each interaction.

Traditional RAG systems are static - they retrieve information based solely on embedding similarity. With a feedback loop, we create a dynamic system that:

- Remembers what worked (and what didn't)
- Adjusts document relevance scores over time
- Incorporates successful Q&A pairs into its knowledge base
- Gets smarter with each user interaction

## Setting Up the Environment
We begin by importing necessary libraries.

```python
import fitz
import os
import numpy as np
import json
from openai import OpenAI
from datetime import datetime
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
    
    This class provides an in-memory storage and retrieval system for 
    embedding vectors and their corresponding text chunks and metadata.
    It supports basic similarity search functionality using cosine similarity.
    """
    def __init__(self):
        """
        Initialize the vector store with empty lists for vectors, texts, and metadata.
        
        The vector store maintains three parallel lists:
        - vectors: NumPy arrays of embedding vectors
        - texts: Original text chunks corresponding to each vector
        - metadata: Optional metadata dictionaries for each item
        """
        self.vectors = []  # List to store embedding vectors
        self.texts = []    # List to store original text chunks
        self.metadata = [] # List to store metadata for each text chunk
    
    def add_item(self, text, embedding, metadata=None):
        """
        Add an item to the vector store.

        Args:
            text (str): The original text chunk to store.
            embedding (List[float]): The embedding vector representing the text.
            metadata (dict, optional): Additional metadata for the text chunk,
                                      such as source, timestamp, or relevance scores.
        """
        self.vectors.append(np.array(embedding))  # Convert and store the embedding
        self.texts.append(text)                   # Store the original text
        self.metadata.append(metadata or {})      # Store metadata (empty dict if None)
    
    def similarity_search(self, query_embedding, k=5, filter_func=None):
        """
        Find the most similar items to a query embedding using cosine similarity.

        Args:
            query_embedding (List[float]): Query embedding vector to compare against stored vectors.
            k (int): Number of most similar results to return.
            filter_func (callable, optional): Function to filter results based on metadata.
                                             Takes metadata dict as input and returns boolean.

        Returns:
            List[Dict]: Top k most similar items, each containing:
                - text: The original text
                - metadata: Associated metadata
                - similarity: Raw cosine similarity score
                - relevance_score: Either metadata-based relevance or calculated similarity
                
        Note: Returns empty list if no vectors are stored or none pass the filter.
        """
        if not self.vectors:
            return []  # Return empty list if vector store is empty
        
        # Convert query embedding to numpy array for vector operations
        query_vector = np.array(query_embedding)
        
        # Calculate cosine similarity between query and each stored vector
        similarities = []
        for i, vector in enumerate(self.vectors):
            # Skip items that don't pass the filter criteria
            if filter_func and not filter_func(self.metadata[i]):
                continue
                
            # Calculate cosine similarity: dot product / (norm1 * norm2)
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))  # Store index and similarity score
        
        # Sort results by similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Construct result dictionaries for the top k matches
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": score,
                # Use pre-existing relevance score from metadata if available, otherwise use similarity
                "relevance_score": self.metadata[idx].get("relevance_score", score)
            })
        
        return results
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
    # Convert single string to list for uniform processing
    input_text = text if isinstance(text, list) else [text]
    
    # Call the OpenAI API to generate embeddings for all input texts
    response = client.embeddings.create(
        model=model,
        input=input_text
    )
    
    # For single string input, return just the first embedding vector
    if isinstance(text, str):
        return response.data[0].embedding
    
    # For list input, return a list of all embedding vectors
    return [item.embedding for item in response.data]
```

## Feedback System Functions
Now we'll implement the core feedback system components.

```python
def get_user_feedback(query, response, relevance, quality, comments=""):
    """
    Format user feedback in a dictionary.
    
    Args:
        query (str): User's query
        response (str): System's response
        relevance (int): Relevance score (1-5)
        quality (int): Quality score (1-5)
        comments (str): Optional feedback comments
        
    Returns:
        Dict: Formatted feedback
    """
    return {
        "query": query,
        "response": response,
        "relevance": int(relevance),
        "quality": int(quality),
        "comments": comments,
        "timestamp": datetime.now().isoformat()
    }
```

```python
def store_feedback(feedback, feedback_file="feedback_data.json"):
    """
    Store feedback in a JSON file.
    
    Args:
        feedback (Dict): Feedback data
        feedback_file (str): Path to feedback file
    """
    with open(feedback_file, "a") as f:
        json.dump(feedback, f)
        f.write("\n")
```

```python
def load_feedback_data(feedback_file="feedback_data.json"):
    """
    Load feedback data from file.
    
    Args:
        feedback_file (str): Path to feedback file
        
    Returns:
        List[Dict]: List of feedback entries
    """
    feedback_data = []
    try:
        with open(feedback_file, "r") as f:
            for line in f:
                if line.strip():
                    feedback_data.append(json.loads(line.strip()))
    except FileNotFoundError:
        print("No feedback data file found. Starting with empty feedback.")
    
    return feedback_data
```

## Document Processing with Feedback Awareness

```python
def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    Process a document for RAG (Retrieval Augmented Generation) with feedback loop.
    This function handles the complete document processing pipeline:
    1. Text extraction from PDF
    2. Text chunking with overlap
    3. Embedding creation for chunks
    4. Storage in vector database with metadata

    Args:
    pdf_path (str): Path to the PDF file to process.
    chunk_size (int): Size of each text chunk in characters.
    chunk_overlap (int): Number of overlapping characters between consecutive chunks.

    Returns:
    Tuple[List[str], SimpleVectorStore]: A tuple containing:
        - List of document chunks
        - Populated vector store with embeddings and metadata
    """
    # Step 1: Extract raw text content from the PDF document
    print("Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_path)
    
    # Step 2: Split text into manageable, overlapping chunks for better context preservation
    print("Chunking text...")
    chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} text chunks")
    
    # Step 3: Generate vector embeddings for each text chunk
    print("Creating embeddings for chunks...")
    chunk_embeddings = create_embeddings(chunks)
    
    # Step 4: Initialize the vector database to store chunks and their embeddings
    store = SimpleVectorStore()
    
    # Step 5: Add each chunk with its embedding to the vector store
    # Include metadata for feedback-based improvements
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={
                "index": i,                # Position in original document
                "source": pdf_path,        # Source document path
                "relevance_score": 1.0,    # Initial relevance score (will be updated with feedback)
                "feedback_count": 0        # Counter for feedback received on this chunk
            }
        )
    
    print(f"Added {len(chunks)} chunks to the vector store")
    return chunks, store
```

## Relevance Adjustment Based on Feedback

```python
def assess_feedback_relevance(query, doc_text, feedback):
    """
    Use LLM to assess if a past feedback entry is relevant to the current query and document.
    
    This function helps determine which past feedback should influence the current retrieval
    by sending the current query, past query+feedback, and document content to an LLM
    for relevance assessment.
    
    Args:
        query (str): Current user query that needs information retrieval
        doc_text (str): Text content of the document being evaluated
        feedback (Dict): Previous feedback data containing 'query' and 'response' keys
        
    Returns:
        bool: True if the feedback is deemed relevant to current query/document, False otherwise
    """
    # Define system prompt instructing the LLM to make binary relevance judgments only
    system_prompt = """You are an AI system that determines if a past feedback is relevant to a current query and document.
    Answer with ONLY 'yes' or 'no'. Your job is strictly to determine relevance, not to provide explanations."""

    # Construct user prompt with current query, past feedback data, and truncated document content
    user_prompt = f"""
    Current query: {query}
    Past query that received feedback: {feedback['query']}
    Document content: {doc_text[:500]}... [truncated]
    Past response that received feedback: {feedback['response'][:500]}... [truncated]

    Is this past feedback relevant to the current query and document? (yes/no)
    """

    # Call the LLM API with zero temperature for deterministic output
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0  # Use temperature=0 for consistent, deterministic responses
    )
    
    # Extract and normalize the response to determine relevance
    answer = response.choices[0].message.content.strip().lower()
    return 'yes' in answer  # Return True if the answer contains 'yes'
```

```python
def adjust_relevance_scores(query, results, feedback_data):
    """
    Adjust document relevance scores based on historical feedback to improve retrieval quality.
    
    This function analyzes past user feedback to dynamically adjust the relevance scores of 
    retrieved documents. It identifies feedback that is relevant to the current query context,
    calculates score modifiers based on relevance ratings, and re-ranks the results accordingly.
    
    Args:
        query (str): Current user query
        results (List[Dict]): Retrieved documents with their original similarity scores
        feedback_data (List[Dict]): Historical feedback containing user ratings
        
    Returns:
        List[Dict]: Results with adjusted relevance scores, sorted by the new scores
    """
    # If no feedback data available, return original results unchanged
    if not feedback_data:
        return results
    
    print("Adjusting relevance scores based on feedback history...")
    
    # Process each retrieved document
    for i, result in enumerate(results):
        document_text = result["text"]
        relevant_feedback = []
        
        # Find relevant feedback for this specific document and query combination
        # by querying the LLM to assess relevance of each historical feedback item
        for feedback in feedback_data:
            is_relevant = assess_feedback_relevance(query, document_text, feedback)
            if is_relevant:
                relevant_feedback.append(feedback)
        
        # Apply score adjustments if relevant feedback exists
        if relevant_feedback:
            # Calculate average relevance rating from all applicable feedback entries
            # Feedback relevance is on a 1-5 scale (1=not relevant, 5=highly relevant)
            avg_relevance = sum(f['relevance'] for f in relevant_feedback) / len(relevant_feedback)
            
            # Convert the average relevance to a score modifier in range 0.5-1.5
            # - Scores below 3/5 will reduce the original similarity (modifier < 1.0)
            # - Scores above 3/5 will increase the original similarity (modifier > 1.0)
            modifier = 0.5 + (avg_relevance / 5.0)
            
            # Apply the modifier to the original similarity score
            original_score = result["similarity"]
            adjusted_score = original_score * modifier
            
            # Update the result dictionary with new scores and feedback metadata
            result["original_similarity"] = original_score  # Preserve the original score
            result["similarity"] = adjusted_score           # Update the primary score
            result["relevance_score"] = adjusted_score      # Update the relevance score
            result["feedback_applied"] = True               # Flag that feedback was applied
            result["feedback_count"] = len(relevant_feedback)  # Number of feedback entries used
            
            # Log the adjustment details
            print(f"  Document {i+1}: Adjusted score from {original_score:.4f} to {adjusted_score:.4f} based on {len(relevant_feedback)} feedback(s)")
    
    # Re-sort results by adjusted scores to ensure higher quality matches appear first
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    return results
```

## Fine-tuning Our Index with Feedback

```python
def fine_tune_index(current_store, chunks, feedback_data):
    """
    Enhance vector store with high-quality feedback to improve retrieval quality over time.
    
    This function implements a continuous learning process by:
    1. Identifying high-quality feedback (highly rated Q&A pairs)
    2. Creating new retrieval items from successful interactions
    3. Adding these to the vector store with boosted relevance weights
    
    Args:
        current_store (SimpleVectorStore): Current vector store containing original document chunks
        chunks (List[str]): Original document text chunks 
        feedback_data (List[Dict]): Historical user feedback with relevance and quality ratings
        
    Returns:
        SimpleVectorStore: Enhanced vector store containing both original chunks and feedback-derived content
    """
    print("Fine-tuning index with high-quality feedback...")
    
    # Filter for only high-quality responses (both relevance and quality rated 4 or 5)
    # This ensures we only learn from the most successful interactions
    good_feedback = [f for f in feedback_data if f['relevance'] >= 4 and f['quality'] >= 4]
    
    if not good_feedback:
        print("No high-quality feedback found for fine-tuning.")
        return current_store  # Return original store unchanged if no good feedback exists
    
    # Initialize new store that will contain both original and enhanced content
    new_store = SimpleVectorStore()
    
    # First transfer all original document chunks with their existing metadata
    for i in range(len(current_store.texts)):
        new_store.add_item(
            text=current_store.texts[i],
            embedding=current_store.vectors[i],
            metadata=current_store.metadata[i].copy()  # Use copy to prevent reference issues
        )
    
    # Create and add enhanced content from good feedback
    for feedback in good_feedback:
        # Format a new document that combines the question and its high-quality answer
        # This creates retrievable content that directly addresses user queries
        enhanced_text = f"Question: {feedback['query']}\nAnswer: {feedback['response']}"
        
        # Generate embedding vector for this new synthetic document
        embedding = create_embeddings(enhanced_text)
        
        # Add to vector store with special metadata that identifies its origin and importance
        new_store.add_item(
            text=enhanced_text,
            embedding=embedding,
            metadata={
                "type": "feedback_enhanced",  # Mark as derived from feedback
                "query": feedback["query"],   # Store original query for reference
                "relevance_score": 1.2,       # Boost initial relevance to prioritize these items
                "feedback_count": 1,          # Track feedback incorporation
                "original_feedback": feedback # Preserve complete feedback record
            }
        )
        
        print(f"Added enhanced content from feedback: {feedback['query'][:50]}...")
    
    # Log summary statistics about the enhancement
    print(f"Fine-tuned index now has {len(new_store.texts)} items (original: {len(chunks)})")
    return new_store
```

## Complete RAG Pipeline with Feedback Loop

```python
def generate_response(query, context, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Generate a response based on the query and context.
    
    Args:
        query (str): User query
        context (str): Context text from retrieved documents
        model (str): LLM model to use
        
    Returns:
        str: Generated response
    """
    # Define the system prompt to guide the AI's behavior
    system_prompt = """You are a helpful AI assistant. Answer the user's question based only on the provided context. If you cannot find the answer in the context, state that you don't have enough information."""
    
    # Create the user prompt by combining the context and the query
    user_prompt = f"""
        Context:
        {context}

        Question: {query}

        Please provide a comprehensive answer based only on the context above.
    """
    
    # Call the OpenAI API to generate a response based on the system and user prompts
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0  # Use temperature=0 for consistent, deterministic responses
    )
    
    # Return the generated response content
    return response.choices[0].message.content
```

```python
def rag_with_feedback_loop(query, vector_store, feedback_data, k=5, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Complete RAG pipeline incorporating feedback loop.
    
    Args:
        query (str): User query
        vector_store (SimpleVectorStore): Vector store with document chunks
        feedback_data (List[Dict]): History of feedback
        k (int): Number of documents to retrieve
        model (str): LLM model for response generation
        
    Returns:
        Dict: Results including query, retrieved documents, and response
    """
    print(f"\n=== Processing query with feedback-enhanced RAG ===")
    print(f"Query: {query}")
    
    # Step 1: Create query embedding
    query_embedding = create_embeddings(query)
    
    # Step 2: Perform initial retrieval based on query embedding
    results = vector_store.similarity_search(query_embedding, k=k)
    
    # Step 3: Adjust relevance scores of retrieved documents based on feedback
    adjusted_results = adjust_relevance_scores(query, results, feedback_data)
    
    # Step 4: Extract texts from adjusted results for context building
    retrieved_texts = [result["text"] for result in adjusted_results]
    
    # Step 5: Build context for response generation by concatenating retrieved texts
    context = "\n\n---\n\n".join(retrieved_texts)
    
    # Step 6: Generate response using the context and query
    print("Generating response...")
    response = generate_response(query, context, model)
    
    # Step 7: Compile the final result
    result = {
        "query": query,
        "retrieved_documents": adjusted_results,
        "response": response
    }
    
    print("\n=== Response ===")
    print(response)
    
    return result
```

## Complete Workflow: From Initial Setup to Feedback Collection

```python
def full_rag_workflow(pdf_path, query, feedback_data=None, feedback_file="feedback_data.json", fine_tune=False):
    """
    Execute a complete RAG workflow with feedback integration for continuous improvement.
    
    This function orchestrates the entire Retrieval-Augmented Generation process:
    1. Load historical feedback data
    2. Process and chunk the document
    3. Optionally fine-tune the vector index with prior feedback
    4. Perform retrieval and generation with feedback-adjusted relevance scores
    5. Collect new user feedback for future improvement
    6. Store feedback to enable system learning over time
    
    Args:
        pdf_path (str): Path to the PDF document to be processed
        query (str): User's natural language query
        feedback_data (List[Dict], optional): Pre-loaded feedback data, loads from file if None
        feedback_file (str): Path to the JSON file storing feedback history
        fine_tune (bool): Whether to enhance the index with successful past Q&A pairs
        
    Returns:
        Dict: Results containing the response and retrieval metadata
    """
    # Step 1: Load historical feedback for relevance adjustment if not explicitly provided
    if feedback_data is None:
        feedback_data = load_feedback_data(feedback_file)
        print(f"Loaded {len(feedback_data)} feedback entries from {feedback_file}")
    
    # Step 2: Process document through extraction, chunking and embedding pipeline
    chunks, vector_store = process_document(pdf_path)
    
    # Step 3: Fine-tune the vector index by incorporating high-quality past interactions
    # This creates enhanced retrievable content from successful Q&A pairs
    if fine_tune and feedback_data:
        vector_store = fine_tune_index(vector_store, chunks, feedback_data)
    
    # Step 4: Execute core RAG with feedback-aware retrieval
    # Note: This depends on the rag_with_feedback_loop function which should be defined elsewhere
    result = rag_with_feedback_loop(query, vector_store, feedback_data)
    
    # Step 5: Collect user feedback to improve future performance
    print("\n=== Would you like to provide feedback on this response? ===")
    print("Rate relevance (1-5, with 5 being most relevant):")
    relevance = input()
    
    print("Rate quality (1-5, with 5 being highest quality):")
    quality = input()
    
    print("Any comments? (optional, press Enter to skip)")
    comments = input()
    
    # Step 6: Format feedback into structured data
    feedback = get_user_feedback(
        query=query,
        response=result["response"],
        relevance=int(relevance),
        quality=int(quality),
        comments=comments
    )
    
    # Step 7: Persist feedback to enable continuous system learning
    store_feedback(feedback, feedback_file)
    print("Feedback recorded. Thank you!")
    
    return result
```

## Evaluating Our Feedback Loop

```python
def evaluate_feedback_loop(pdf_path, test_queries, reference_answers=None):
    """
    Evaluate the impact of feedback loop on RAG quality by comparing performance before and after feedback integration.
    
    This function runs a controlled experiment to measure how incorporating feedback affects retrieval and generation:
    1. First round: Run all test queries with no feedback
    2. Generate synthetic feedback based on reference answers (if provided)
    3. Second round: Run the same queries with feedback-enhanced retrieval
    4. Compare results between rounds to quantify feedback impact
    
    Args:
        pdf_path (str): Path to the PDF document used as the knowledge base
        test_queries (List[str]): List of test queries to evaluate system performance
        reference_answers (List[str], optional): Reference/gold standard answers for evaluation
                                                and synthetic feedback generation
        
    Returns:
        Dict: Evaluation results containing:
            - round1_results: Results without feedback
            - round2_results: Results with feedback
            - comparison: Quantitative comparison metrics between rounds
    """
    print("=== Evaluating Feedback Loop Impact ===")
    
    # Create a temporary feedback file for this evaluation session only
    temp_feedback_file = "temp_evaluation_feedback.json"
    
    # Initialize feedback collection (empty at the start)
    feedback_data = []
    
    # ----------------------- FIRST EVALUATION ROUND -----------------------
    # Run all queries without any feedback influence to establish baseline performance
    print("\n=== ROUND 1: NO FEEDBACK ===")
    round1_results = []
    
    for i, query in enumerate(test_queries):
        print(f"\nQuery {i+1}: {query}")
        
        # Process document to create initial vector store
        chunks, vector_store = process_document(pdf_path)
        
        # Execute RAG without feedback influence (empty feedback list)
        result = rag_with_feedback_loop(query, vector_store, [])
        round1_results.append(result)
        
        # Generate synthetic feedback if reference answers are available
        # This simulates user feedback for training the system
        if reference_answers and i < len(reference_answers):
            # Calculate synthetic feedback scores based on similarity to reference answer
            similarity_to_ref = calculate_similarity(result["response"], reference_answers[i])
            # Convert similarity (0-1) to rating scale (1-5)
            relevance = max(1, min(5, int(similarity_to_ref * 5)))
            quality = max(1, min(5, int(similarity_to_ref * 5)))
            
            # Create structured feedback entry
            feedback = get_user_feedback(
                query=query,
                response=result["response"],
                relevance=relevance,
                quality=quality,
                comments=f"Synthetic feedback based on reference similarity: {similarity_to_ref:.2f}"
            )
            
            # Add to in-memory collection and persist to temporary file
            feedback_data.append(feedback)
            store_feedback(feedback, temp_feedback_file)
    
    # ----------------------- SECOND EVALUATION ROUND -----------------------
    # Run the same queries with feedback incorporation to measure improvement
    print("\n=== ROUND 2: WITH FEEDBACK ===")
    round2_results = []
    
    # Process document and enhance with feedback-derived content
    chunks, vector_store = process_document(pdf_path)
    vector_store = fine_tune_index(vector_store, chunks, feedback_data)
    
    for i, query in enumerate(test_queries):
        print(f"\nQuery {i+1}: {query}")
        
        # Execute RAG with feedback influence
        result = rag_with_feedback_loop(query, vector_store, feedback_data)
        round2_results.append(result)
    
    # ----------------------- RESULTS ANALYSIS -----------------------
    # Compare performance metrics between the two rounds
    comparison = compare_results(test_queries, round1_results, round2_results, reference_answers)
    
    # Clean up temporary evaluation artifacts
    if os.path.exists(temp_feedback_file):
        os.remove(temp_feedback_file)
    
    return {
        "round1_results": round1_results,
        "round2_results": round2_results,
        "comparison": comparison
    }
```

## Helper Functions for Evaluation

```python
def calculate_similarity(text1, text2):
    """
    Calculate semantic similarity between two texts using embeddings.
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Generate embeddings for both texts
    embedding1 = create_embeddings(text1)
    embedding2 = create_embeddings(text2)
    
    # Convert embeddings to numpy arrays
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    # Calculate cosine similarity between the two vectors
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    return similarity
```

```python
def compare_results(queries, round1_results, round2_results, reference_answers=None):
    """
    Compare results from two rounds of RAG.
    
    Args:
        queries (List[str]): Test queries
        round1_results (List[Dict]): Results from round 1
        round2_results (List[Dict]): Results from round 2
        reference_answers (List[str], optional): Reference answers
        
    Returns:
        str: Comparison analysis
    """
    print("\n=== COMPARING RESULTS ===")
    
    # System prompt to guide the AI's evaluation behavior
    system_prompt = """You are an expert evaluator of RAG systems. Compare responses from two versions:
        1. Standard RAG: No feedback used
        2. Feedback-enhanced RAG: Uses a feedback loop to improve retrieval

        Analyze which version provides better responses in terms of:
        - Relevance to the query
        - Accuracy of information
        - Completeness
        - Clarity and conciseness
    """

    comparisons = []
    
    # Iterate over each query and its corresponding results from both rounds
    for i, (query, r1, r2) in enumerate(zip(queries, round1_results, round2_results)):
        # Create a prompt for comparing the responses
        comparison_prompt = f"""
        Query: {query}

        Standard RAG Response:
        {r1["response"]}

        Feedback-enhanced RAG Response:
        {r2["response"]}
        """

        # Include reference answer if available
        if reference_answers and i < len(reference_answers):
            comparison_prompt += f"""
            Reference Answer:
            {reference_answers[i]}
            """

        comparison_prompt += """
        Compare these responses and explain which one is better and why.
        Focus specifically on how the feedback loop has (or hasn't) improved the response quality.
        """

        # Call the OpenAI API to generate a comparison analysis
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": comparison_prompt}
            ],
            temperature=0
        )
        
        # Append the comparison analysis to the results
        comparisons.append({
            "query": query,
            "analysis": response.choices[0].message.content
        })
        
        # Print a snippet of the analysis for each query
        print(f"\nQuery {i+1}: {query}")
        print(f"Analysis: {response.choices[0].message.content[:200]}...")
    
    return comparisons
```

## Evaluation of the feedback loop (Custom Validation Queries)

```python
# AI Document Path
pdf_path = "data/AI_Information.pdf"

# Define test queries
test_queries = [
    "What is a neural network and how does it function?",

    #################################################################################
    ### Commented out queries to reduce the number of queries for testing purposes ###
    
    # "Describe the process and applications of reinforcement learning.",
    # "What are the main applications of natural language processing in today's technology?",
    # "Explain the impact of overfitting in machine learning models and how it can be mitigated."
]

# Define reference answers for evaluation
reference_answers = [
    "A neural network is a series of algorithms that attempt to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. It consists of layers of nodes, with each node representing a neuron. Neural networks function by adjusting the weights of connections between nodes based on the error of the output compared to the expected result.",

    ############################################################################################
    #### Commented out reference answers to reduce the number of queries for testing purposes ###

#     "Reinforcement learning is a type of machine learning where an agent learns to make decisions by performing actions in an environment to maximize cumulative reward. It involves exploration, exploitation, and learning from the consequences of actions. Applications include robotics, game playing, and autonomous vehicles.",
#     "The main applications of natural language processing in today's technology include machine translation, sentiment analysis, chatbots, information retrieval, text summarization, and speech recognition. NLP enables machines to understand and generate human language, facilitating human-computer interaction.",
#     "Overfitting in machine learning models occurs when a model learns the training data too well, capturing noise and outliers. This results in poor generalization to new data, as the model performs well on training data but poorly on unseen data. Mitigation techniques include cross-validation, regularization, pruning, and using more training data."
]

# Run the evaluation
evaluation_results = evaluate_feedback_loop(
    pdf_path=pdf_path,
    test_queries=test_queries,
    reference_answers=reference_answers
)
```

```python
########################################
# # Run a full RAG workflow
########################################

# # Run an interactive example
# print("\n\n=== INTERACTIVE EXAMPLE ===")
# print("Enter your query about AI:")
# user_query = input()

# # Load accumulated feedback
# all_feedback = load_feedback_data()

# # Run full workflow
# result = full_rag_workflow(
#     pdf_path=pdf_path,
#     query=user_query,
#     feedback_data=all_feedback,
#     fine_tune=True
# )

########################################
# # Run a full RAG workflow
########################################
```

## Visualizing Feedback Impact

```python
# Extract the comparison data which contains the analysis of feedback impact
comparisons = evaluation_results['comparison']

# Print out the analysis results to visualize feedback impact
print("\n=== FEEDBACK IMPACT ANALYSIS ===\n")
for i, comparison in enumerate(comparisons):
    print(f"Query {i+1}: {comparison['query']}")
    print(f"\nAnalysis of feedback impact:")
    print(comparison['analysis'])
    print("\n" + "-"*50 + "\n")

# Additionally, we can compare some metrics between rounds
round_responses = [evaluation_results[f'round{round_num}_results'] for round_num in range(1, len(evaluation_results) - 1)]
response_lengths = [[len(r["response"]) for r in round] for round in round_responses]

print("\nResponse length comparison (proxy for completeness):")
avg_lengths = [sum(lengths) / len(lengths) for lengths in response_lengths]
for round_num, avg_len in enumerate(avg_lengths, start=1):
    print(f"Round {round_num}: {avg_len:.1f} chars")

if len(avg_lengths) > 1:
    changes = [(avg_lengths[i] - avg_lengths[i-1]) / avg_lengths[i-1] * 100 for i in range(1, len(avg_lengths))]
    for round_num, change in enumerate(changes, start=2):
        print(f"Change from Round {round_num-1} to Round {round_num}: {change:.1f}%")
```
