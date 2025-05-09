# Query Transformations for Enhanced RAG Systems

This notebook implements three query transformation techniques to enhance retrieval performance in RAG systems without relying on specialized libraries like LangChain. By modifying user queries, we can significantly improve the relevance and comprehensiveness of retrieved information.

## Key Transformation Techniques

1. **Query Rewriting**: Makes queries more specific and detailed for better search precision.
2. **Step-back Prompting**: Generates broader queries to retrieve useful contextual information.
3. **Sub-query Decomposition**: Breaks complex queries into simpler components for comprehensive retrieval.

## Setting Up the Environment
We begin by importing necessary libraries.

```python
import fitz
import os
import numpy as np
import json
from openai import OpenAI
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

## Implementing Query Transformation Techniques
### 1. Query Rewriting
This technique makes queries more specific and detailed to improve precision in retrieval.

```python
def rewrite_query(original_query, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Rewrites a query to make it more specific and detailed for better retrieval.
    
    Args:
        original_query (str): The original user query
        model (str): The model to use for query rewriting
        
    Returns:
        str: The rewritten query
    """
    # Define the system prompt to guide the AI assistant's behavior
    system_prompt = "You are an AI assistant specialized in improving search queries. Your task is to rewrite user queries to be more specific, detailed, and likely to retrieve relevant information."
    
    # Define the user prompt with the original query to be rewritten
    user_prompt = f"""
    Rewrite the following query to make it more specific and detailed. Include relevant terms and concepts that might help in retrieving accurate information.
    
    Original query: {original_query}
    
    Rewritten query:
    """
    
    # Generate the rewritten query using the specified model
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,  # Low temperature for deterministic output
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    # Return the rewritten query, stripping any leading/trailing whitespace
    return response.choices[0].message.content.strip()
```

### 2. Step-back Prompting
This technique generates broader queries to retrieve contextual background information.

```python
def generate_step_back_query(original_query, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Generates a more general 'step-back' query to retrieve broader context.
    
    Args:
        original_query (str): The original user query
        model (str): The model to use for step-back query generation
        
    Returns:
        str: The step-back query
    """
    # Define the system prompt to guide the AI assistant's behavior
    system_prompt = "You are an AI assistant specialized in search strategies. Your task is to generate broader, more general versions of specific queries to retrieve relevant background information."
    
    # Define the user prompt with the original query to be generalized
    user_prompt = f"""
    Generate a broader, more general version of the following query that could help retrieve useful background information.
    
    Original query: {original_query}
    
    Step-back query:
    """
    
    # Generate the step-back query using the specified model
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,  # Slightly higher temperature for some variation
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    # Return the step-back query, stripping any leading/trailing whitespace
    return response.choices[0].message.content.strip()
```

### 3. Sub-query Decomposition
This technique breaks down complex queries into simpler components for comprehensive retrieval.

```python
def decompose_query(original_query, num_subqueries=4, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Decomposes a complex query into simpler sub-queries.
    
    Args:
        original_query (str): The original complex query
        num_subqueries (int): Number of sub-queries to generate
        model (str): The model to use for query decomposition
        
    Returns:
        List[str]: A list of simpler sub-queries
    """
    # Define the system prompt to guide the AI assistant's behavior
    system_prompt = "You are an AI assistant specialized in breaking down complex questions. Your task is to decompose complex queries into simpler sub-questions that, when answered together, address the original query."
    
    # Define the user prompt with the original query to be decomposed
    user_prompt = f"""
    Break down the following complex query into {num_subqueries} simpler sub-queries. Each sub-query should focus on a different aspect of the original question.
    
    Original query: {original_query}
    
    Generate {num_subqueries} sub-queries, one per line, in this format:
    1. [First sub-query]
    2. [Second sub-query]
    And so on...
    """
    
    # Generate the sub-queries using the specified model
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,  # Slightly higher temperature for some variation
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    # Process the response to extract sub-queries
    content = response.choices[0].message.content.strip()
    
    # Extract numbered queries using simple parsing
    lines = content.split("\n")
    sub_queries = []
    
    for line in lines:
        if line.strip() and any(line.strip().startswith(f"{i}.") for i in range(1, 10)):
            # Remove the number and leading space
            query = line.strip()
            query = query[query.find(".")+1:].strip()
            sub_queries.append(query)
    
    return sub_queries
```

## Demonstrating Query Transformation Techniques
Let's apply these techniques to an example query.

```python
# Example query
original_query = "What are the impacts of AI on job automation and employment?"

# Apply query transformations
print("Original Query:", original_query)

# Query Rewriting
rewritten_query = rewrite_query(original_query)
print("\n1. Rewritten Query:")
print(rewritten_query)

# Step-back Prompting
step_back_query = generate_step_back_query(original_query)
print("\n2. Step-back Query:")
print(step_back_query)

# Sub-query Decomposition
sub_queries = decompose_query(original_query, num_subqueries=4)
print("\n3. Sub-queries:")
for i, query in enumerate(sub_queries, 1):
    print(f"   {i}. {query}")
```

## Building a Simple Vector Store
To demonstrate how query transformations integrate with retrieval, let's implement a simple vector store.

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
        self.metadata.append(metadata or {})  # Add metadata to metadata list, use empty dict if None
    
    def similarity_search(self, query_embedding, k=5):
        """
        Find the most similar items to a query embedding.

        Args:
        query_embedding (List[float]): Query embedding vector.
        k (int): Number of results to return.

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
            # Compute cosine similarity between query vector and stored vector
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))  # Append index and similarity score
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],  # Add the corresponding text
                "metadata": self.metadata[idx],  # Add the corresponding metadata
                "similarity": score  # Add the similarity score
            })
        
        return results  # Return the list of top k similar items
```

## Creating Embeddings

```python
def create_embeddings(text, model="BAAI/bge-en-icl"):
    """
    Creates embeddings for the given text using the specified OpenAI model.

    Args:
    text (str): The input text for which embeddings are to be created.
    model (str): The model to be used for creating embeddings.

    Returns:
    List[float]: The embedding vector.
    """
    # Handle both string and list inputs by converting string input to a list
    input_text = text if isinstance(text, list) else [text]
    
    # Create embeddings for the input text using the specified model
    response = client.embeddings.create(
        model=model,
        input=input_text
    )
    
    # If input was a string, return just the first embedding
    if isinstance(text, str):
        return response.data[0].embedding
    
    # Otherwise, return all embeddings as a list of vectors
    return [item.embedding for item in response.data]
```

## Implementing RAG with Query Transformations

```python
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

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

```python
def chunk_text(text, n=1000, overlap=200):
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

```python
def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    Process a document for RAG.

    Args:
    pdf_path (str): Path to the PDF file.
    chunk_size (int): Size of each chunk in characters.
    chunk_overlap (int): Overlap between chunks in characters.

    Returns:
    SimpleVectorStore: A vector store containing document chunks and their embeddings.
    """
    print("Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_path)
    
    print("Chunking text...")
    chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} text chunks")
    
    print("Creating embeddings for chunks...")
    # Create embeddings for all chunks at once for efficiency
    chunk_embeddings = create_embeddings(chunks)
    
    # Create vector store
    store = SimpleVectorStore()
    
    # Add chunks to vector store
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={"index": i, "source": pdf_path}
        )
    
    print(f"Added {len(chunks)} chunks to the vector store")
    return store
```

## RAG with Query Transformations

```python
def transformed_search(query, vector_store, transformation_type, top_k=3):
    """
    Search using a transformed query.
    
    Args:
        query (str): Original query
        vector_store (SimpleVectorStore): Vector store to search
        transformation_type (str): Type of transformation ('rewrite', 'step_back', or 'decompose')
        top_k (int): Number of results to return
        
    Returns:
        List[Dict]: Search results
    """
    print(f"Transformation type: {transformation_type}")
    print(f"Original query: {query}")
    
    results = []
    
    if transformation_type == "rewrite":
        # Query rewriting
        transformed_query = rewrite_query(query)
        print(f"Rewritten query: {transformed_query}")
        
        # Create embedding for transformed query
        query_embedding = create_embeddings(transformed_query)
        
        # Search with rewritten query
        results = vector_store.similarity_search(query_embedding, k=top_k)
        
    elif transformation_type == "step_back":
        # Step-back prompting
        transformed_query = generate_step_back_query(query)
        print(f"Step-back query: {transformed_query}")
        
        # Create embedding for transformed query
        query_embedding = create_embeddings(transformed_query)
        
        # Search with step-back query
        results = vector_store.similarity_search(query_embedding, k=top_k)
        
    elif transformation_type == "decompose":
        # Sub-query decomposition
        sub_queries = decompose_query(query)
        print("Decomposed into sub-queries:")
        for i, sub_q in enumerate(sub_queries, 1):
            print(f"{i}. {sub_q}")
        
        # Create embeddings for all sub-queries
        sub_query_embeddings = create_embeddings(sub_queries)
        
        # Search with each sub-query and combine results
        all_results = []
        for i, embedding in enumerate(sub_query_embeddings):
            sub_results = vector_store.similarity_search(embedding, k=2)  # Get fewer results per sub-query
            all_results.extend(sub_results)
        
        # Remove duplicates (keep highest similarity score)
        seen_texts = {}
        for result in all_results:
            text = result["text"]
            if text not in seen_texts or result["similarity"] > seen_texts[text]["similarity"]:
                seen_texts[text] = result
        
        # Sort by similarity and take top_k
        results = sorted(seen_texts.values(), key=lambda x: x["similarity"], reverse=True)[:top_k]
        
    else:
        # Regular search without transformation
        query_embedding = create_embeddings(query)
        results = vector_store.similarity_search(query_embedding, k=top_k)
    
    return results
```

## Generating a Response with Transformed Queries

```python
def generate_response(query, context, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Generates a response based on the query and retrieved context.
    
    Args:
        query (str): User query
        context (str): Retrieved context
        model (str): The model to use for response generation
        
    Returns:
        str: Generated response
    """
    # Define the system prompt to guide the AI assistant's behavior
    system_prompt = "You are a helpful AI assistant. Answer the user's question based only on the provided context. If you cannot find the answer in the context, state that you don't have enough information."
    
    # Define the user prompt with the context and query
    user_prompt = f"""
        Context:
        {context}

        Question: {query}

        Please provide a comprehensive answer based only on the context above.
    """
    
    # Generate the response using the specified model
    response = client.chat.completions.create(
        model=model,
        temperature=0,  # Low temperature for deterministic output
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    # Return the generated response, stripping any leading/trailing whitespace
    return response.choices[0].message.content.strip()
```

## Running the Complete RAG Pipeline with Query Transformations

```python
def rag_with_query_transformation(pdf_path, query, transformation_type=None):
    """
    Run complete RAG pipeline with optional query transformation.
    
    Args:
        pdf_path (str): Path to PDF document
        query (str): User query
        transformation_type (str): Type of transformation (None, 'rewrite', 'step_back', or 'decompose')
        
    Returns:
        Dict: Results including query, transformed query, context, and response
    """
    # Process the document to create a vector store
    vector_store = process_document(pdf_path)
    
    # Apply query transformation and search
    if transformation_type:
        # Perform search with transformed query
        results = transformed_search(query, vector_store, transformation_type)
    else:
        # Perform regular search without transformation
        query_embedding = create_embeddings(query)
        results = vector_store.similarity_search(query_embedding, k=3)
    
    # Combine context from search results
    context = "\n\n".join([f"PASSAGE {i+1}:\n{result['text']}" for i, result in enumerate(results)])
    
    # Generate response based on the query and combined context
    response = generate_response(query, context)
    
    # Return the results including original query, transformation type, context, and response
    return {
        "original_query": query,
        "transformation_type": transformation_type,
        "context": context,
        "response": response
    }
```

## Evaluating Transformation Techniques

```python

def compare_responses(results, reference_answer, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Compare responses from different query transformation techniques.
    
    Args:
        results (Dict): Results from different transformation techniques
        reference_answer (str): Reference answer for comparison
        model (str): Model for evaluation
    """
    # Define the system prompt to guide the AI assistant's behavior
    system_prompt = """You are an expert evaluator of RAG systems. 
    Your task is to compare different responses generated using various query transformation techniques 
    and determine which technique produced the best response compared to the reference answer."""
    
    # Prepare the comparison text with the reference answer and responses from each technique
    comparison_text = f"""Reference Answer: {reference_answer}\n\n"""
    
    for technique, result in results.items():
        comparison_text += f"{technique.capitalize()} Query Response:\n{result['response']}\n\n"
    
    # Define the user prompt with the comparison text
    user_prompt = f"""
    {comparison_text}
    
    Compare the responses generated by different query transformation techniques to the reference answer.
    
    For each technique (original, rewrite, step_back, decompose):
    1. Score the response from 1-10 based on accuracy, completeness, and relevance
    2. Identify strengths and weaknesses
    
    Then rank the techniques from best to worst and explain which technique performed best overall and why.
    """
    
    # Generate the evaluation response using the specified model
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    # Print the evaluation results
    print("\n===== EVALUATION RESULTS =====")
    print(response.choices[0].message.content)
    print("=============================")
```

```python
def evaluate_transformations(pdf_path, query, reference_answer=None):
    """
    Evaluate different transformation techniques for the same query.
    
    Args:
        pdf_path (str): Path to PDF document
        query (str): Query to evaluate
        reference_answer (str): Optional reference answer for comparison
        
    Returns:
        Dict: Evaluation results
    """
    # Define the transformation techniques to evaluate
    transformation_types = [None, "rewrite", "step_back", "decompose"]
    results = {}
    
    # Run RAG with each transformation technique
    for transformation_type in transformation_types:
        type_name = transformation_type if transformation_type else "original"
        print(f"\n===== Running RAG with {type_name} query =====")
        
        # Get the result for the current transformation type
        result = rag_with_query_transformation(pdf_path, query, transformation_type)
        results[type_name] = result
        
        # Print the response for the current transformation type
        print(f"Response with {type_name} query:")
        print(result["response"])
        print("=" * 50)
    
    # Compare results if a reference answer is provided
    if reference_answer:
        compare_responses(results, reference_answer)
    
    return results
```

## Evaluation of Query Transformations

```python
# Load the validation data from a JSON file
with open('data/val.json') as f:
    data = json.load(f)

# Extract the first query from the validation data
query = data[0]['question']

# Extract the reference answer from the validation data
reference_answer = data[0]['ideal_answer']

# pdf_path
pdf_path = "data/AI_Information.pdf"

# Run evaluation
evaluation_results = evaluate_transformations(pdf_path, query, reference_answer)
```
