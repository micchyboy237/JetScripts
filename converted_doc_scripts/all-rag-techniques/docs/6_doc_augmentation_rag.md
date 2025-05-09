# Document Augmentation RAG with Question Generation

This notebook implements an enhanced RAG approach using document augmentation through question generation. By generating relevant questions for each text chunk, we improve the retrieval process, leading to better responses from the language model.

In this implementation, we follow these steps:

1. **Data Ingestion**: Extract text from a PDF file.
2. **Chunking**: Split the text into manageable chunks.
3. **Question Generation**: Generate relevant questions for each chunk.
4. **Embedding Creation**: Create embeddings for both chunks and generated questions.
5. **Vector Store Creation**: Build a simple vector store using NumPy.
6. **Semantic Search**: Retrieve relevant chunks and questions for user queries.
7. **Response Generation**: Generate answers based on retrieved content.
8. **Evaluation**: Assess the quality of the generated responses.

## Setting Up the Environment
We begin by importing necessary libraries.

```python
import fitz
import os
import numpy as np
import json
from openai import OpenAI
import re
from tqdm import tqdm
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

## Generating Questions for Text Chunks
This is the key enhancement over simple RAG. We generate questions that could be answered by each text chunk.

```python
def generate_questions(text_chunk, num_questions=5, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Generates relevant questions that can be answered from the given text chunk.

    Args:
    text_chunk (str): The text chunk to generate questions from.
    num_questions (int): Number of questions to generate.
    model (str): The model to use for question generation.

    Returns:
    List[str]: List of generated questions.
    """
    # Define the system prompt to guide the AI's behavior
    system_prompt = "You are an expert at generating relevant questions from text. Create concise questions that can be answered using only the provided text. Focus on key information and concepts."
    
    # Define the user prompt with the text chunk and the number of questions to generate
    user_prompt = f"""
    Based on the following text, generate {num_questions} different questions that can be answered using only this text:

    {text_chunk}
    
    Format your response as a numbered list of questions only, with no additional text.
    """
    
    # Generate questions using the OpenAI API
    response = client.chat.completions.create(
        model=model,
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    # Extract and clean questions from the response
    questions_text = response.choices[0].message.content.strip()
    questions = []
    
    # Extract questions using regex pattern matching
    for line in questions_text.split('\n'):
        # Remove numbering and clean up whitespace
        cleaned_line = re.sub(r'^\d+\.\s*', '', line.strip())
        if cleaned_line and cleaned_line.endswith('?'):
            questions.append(cleaned_line)
    
    return questions
```

## Creating Embeddings for Text
We generate embeddings for both text chunks and generated questions.

```python
def create_embeddings(text, model="BAAI/bge-en-icl"):
    """
    Creates embeddings for the given text using the specified OpenAI model.

    Args:
    text (str): The input text for which embeddings are to be created.
    model (str): The model to be used for creating embeddings.

    Returns:
    dict: The response from the OpenAI API containing the embeddings.
    """
    # Create embeddings for the input text using the specified model
    response = client.embeddings.create(
        model=model,
        input=text
    )

    return response  # Return the response containing the embeddings
```

## Building a Simple Vector Store
We'll implement a simple vector store using NumPy.

```python
class SimpleVectorStore:
    """
    A simple vector store implementation using NumPy.
    """
    def __init__(self):
        """
        Initialize the vector store.
        """
        self.vectors = []
        self.texts = []
        self.metadata = []
    
    def add_item(self, text, embedding, metadata=None):
        """
        Add an item to the vector store.

        Args:
        text (str): The original text.
        embedding (List[float]): The embedding vector.
        metadata (dict, optional): Additional metadata.
        """
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})
    
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
                "similarity": score
            })
        
        return results
```

## Processing Documents with Question Augmentation
Now we'll put everything together to process documents, generate questions, and build our augmented vector store.

```python
def process_document(pdf_path, chunk_size=1000, chunk_overlap=200, questions_per_chunk=5):
    """
    Process a document with question augmentation.

    Args:
    pdf_path (str): Path to the PDF file.
    chunk_size (int): Size of each text chunk in characters.
    chunk_overlap (int): Overlap between chunks in characters.
    questions_per_chunk (int): Number of questions to generate per chunk.

    Returns:
    Tuple[List[str], SimpleVectorStore]: Text chunks and vector store.
    """
    print("Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_path)
    
    print("Chunking text...")
    text_chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)
    print(f"Created {len(text_chunks)} text chunks")
    
    vector_store = SimpleVectorStore()
    
    print("Processing chunks and generating questions...")
    for i, chunk in enumerate(tqdm(text_chunks, desc="Processing Chunks")):
        # Create embedding for the chunk itself
        chunk_embedding_response = create_embeddings(chunk)
        chunk_embedding = chunk_embedding_response.data[0].embedding
        
        # Add the chunk to the vector store
        vector_store.add_item(
            text=chunk,
            embedding=chunk_embedding,
            metadata={"type": "chunk", "index": i}
        )
        
        # Generate questions for this chunk
        questions = generate_questions(chunk, num_questions=questions_per_chunk)
        
        # Create embeddings for each question and add to vector store
        for j, question in enumerate(questions):
            question_embedding_response = create_embeddings(question)
            question_embedding = question_embedding_response.data[0].embedding
            
            # Add the question to the vector store
            vector_store.add_item(
                text=question,
                embedding=question_embedding,
                metadata={"type": "question", "chunk_index": i, "original_chunk": chunk}
            )
    
    return text_chunks, vector_store
```

## Extracting and Processing the Document

```python
# Define the path to the PDF file
pdf_path = "data/AI_Information.pdf"

# Process the document (extract text, create chunks, generate questions, build vector store)
text_chunks, vector_store = process_document(
    pdf_path, 
    chunk_size=1000, 
    chunk_overlap=200, 
    questions_per_chunk=3
)

print(f"Vector store contains {len(vector_store.texts)} items")
```

## Performing Semantic Search
We implement a semantic search function similar to the simple RAG implementation but adapted to our augmented vector store.

```python
def semantic_search(query, vector_store, k=5):
    """
    Performs semantic search using the query and vector store.

    Args:
    query (str): The search query.
    vector_store (SimpleVectorStore): The vector store to search in.
    k (int): Number of results to return.

    Returns:
    List[Dict]: Top k most relevant items.
    """
    # Create embedding for the query
    query_embedding_response = create_embeddings(query)
    query_embedding = query_embedding_response.data[0].embedding
    
    # Search the vector store
    results = vector_store.similarity_search(query_embedding, k=k)
    
    return results
```

## Running a Query on the Augmented Vector Store

```python
# Load the validation data from a JSON file
with open('data/val.json') as f:
    data = json.load(f)

# Extract the first query from the validation data
query = data[0]['question']

# Perform semantic search to find relevant content
search_results = semantic_search(query, vector_store, k=5)

print("Query:", query)
print("\nSearch Results:")

# Organize results by type
chunk_results = []
question_results = []

for result in search_results:
    if result["metadata"]["type"] == "chunk":
        chunk_results.append(result)
    else:
        question_results.append(result)

# Print chunk results first
print("\nRelevant Document Chunks:")
for i, result in enumerate(chunk_results):
    print(f"Context {i + 1} (similarity: {result['similarity']:.4f}):")
    print(result["text"][:300] + "...")
    print("=====================================")

# Then print question matches
print("\nMatched Questions:")
for i, result in enumerate(question_results):
    print(f"Question {i + 1} (similarity: {result['similarity']:.4f}):")
    print(result["text"])
    chunk_idx = result["metadata"]["chunk_index"]
    print(f"From chunk {chunk_idx}")
    print("=====================================")
```

## Generating Context for Response
Now we prepare the context by combining information from relevant chunks and questions.

```python
def prepare_context(search_results):
    """
    Prepares a unified context from search results for response generation.

    Args:
    search_results (List[Dict]): Results from semantic search.

    Returns:
    str: Combined context string.
    """
    # Extract unique chunks referenced in the results
    chunk_indices = set()
    context_chunks = []
    
    # First add direct chunk matches
    for result in search_results:
        if result["metadata"]["type"] == "chunk":
            chunk_indices.add(result["metadata"]["index"])
            context_chunks.append(f"Chunk {result['metadata']['index']}:\n{result['text']}")
    
    # Then add chunks referenced by questions
    for result in search_results:
        if result["metadata"]["type"] == "question":
            chunk_idx = result["metadata"]["chunk_index"]
            if chunk_idx not in chunk_indices:
                chunk_indices.add(chunk_idx)
                context_chunks.append(f"Chunk {chunk_idx} (referenced by question '{result['text']}'):\n{result['metadata']['original_chunk']}")
    
    # Combine all context chunks
    full_context = "\n\n".join(context_chunks)
    return full_context
```

## Generating a Response Based on Retrieved Chunks

```python
def generate_response(query, context, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Generates a response based on the query and context.

    Args:
    query (str): User's question.
    context (str): Context information retrieved from the vector store.
    model (str): Model to use for response generation.

    Returns:
    str: Generated response.
    """
    system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"
    
    user_prompt = f"""
        Context:
        {context}

        Question: {query}

        Please answer the question based only on the context provided above. Be concise and accurate.
    """
    
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    return response.choices[0].message.content
```

## Generating and Displaying the Response

```python
# Prepare context from search results
context = prepare_context(search_results)

# Generate response
response_text = generate_response(query, context)

print("\nQuery:", query)
print("\nResponse:")
print(response_text)
```

## Evaluating the AI Response
We compare the AI response with the expected answer and assign a score.

```python
def evaluate_response(query, response, reference_answer, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Evaluates the AI response against a reference answer.
    
    Args:
    query (str): The user's question.
    response (str): The AI-generated response.
    reference_answer (str): The reference/ideal answer.
    model (str): Model to use for evaluation.
    
    Returns:
    str: Evaluation feedback.
    """
    # Define the system prompt for the evaluation system
    evaluate_system_prompt = """You are an intelligent evaluation system tasked with assessing AI responses.
            
        Compare the AI assistant's response to the true/reference answer, and evaluate based on:
        1. Factual correctness - Does the response contain accurate information?
        2. Completeness - Does it cover all important aspects from the reference?
        3. Relevance - Does it directly address the question?

        Assign a score from 0 to 1:
        - 1.0: Perfect match in content and meaning
        - 0.8: Very good, with minor omissions/differences
        - 0.6: Good, covers main points but misses some details
        - 0.4: Partial answer with significant omissions
        - 0.2: Minimal relevant information
        - 0.0: Incorrect or irrelevant

        Provide your score with justification.
    """
            
    # Create the evaluation prompt
    evaluation_prompt = f"""
        User Query: {query}

        AI Response:
        {response}

        Reference Answer:
        {reference_answer}

        Please evaluate the AI response against the reference answer.
    """
    
    # Generate evaluation
    eval_response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": evaluate_system_prompt},
            {"role": "user", "content": evaluation_prompt}
        ]
    )
    
    return eval_response.choices[0].message.content
```

## Running the Evaluation

```python
# Get reference answer from validation data
reference_answer = data[0]['ideal_answer']

# Evaluate the response
evaluation = evaluate_response(query, response_text, reference_answer)

print("\nEvaluation:")
print(evaluation)
```

## Extracting and Chunking Text from a PDF File
Now, we load the PDF, extract text, and split it into chunks.

```python
# Define the path to the PDF file
pdf_path = "data/AI_Information.pdf"

# Extract text from the PDF file
extracted_text = extract_text_from_pdf(pdf_path)

# Chunk the extracted text into segments of 1000 characters with an overlap of 200 characters
text_chunks = chunk_text(extracted_text, 1000, 200)

# Print the number of text chunks created
print("Number of text chunks:", len(text_chunks))

# Print the first text chunk
print("\nFirst text chunk:")
print(text_chunks[0])
```

## Creating Embeddings for Text Chunks
Embeddings transform text into numerical vectors, which allow for efficient similarity search.

```python
def create_embeddings(text, model="BAAI/bge-en-icl"):
    """
    Creates embeddings for the given text using the specified OpenAI model.

    Args:
    text (str): The input text for which embeddings are to be created.
    model (str): The model to be used for creating embeddings. Default is "BAAI/bge-en-icl".

    Returns:
    dict: The response from the OpenAI API containing the embeddings.
    """
    # Create embeddings for the input text using the specified model
    response = client.embeddings.create(
        model=model,
        input=text
    )

    return response  # Return the response containing the embeddings

# Create embeddings for the text chunks
response = create_embeddings(text_chunks)
```

## Performing Semantic Search
We implement cosine similarity to find the most relevant text chunks for a user query.

```python
def cosine_similarity(vec1, vec2):
    """
    Calculates the cosine similarity between two vectors.

    Args:
    vec1 (np.ndarray): The first vector.
    vec2 (np.ndarray): The second vector.

    Returns:
    float: The cosine similarity between the two vectors.
    """
    # Compute the dot product of the two vectors and divide by the product of their norms
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
```

```python
def semantic_search(query, text_chunks, embeddings, k=5):
    """
    Performs semantic search on the text chunks using the given query and embeddings.

    Args:
    query (str): The query for the semantic search.
    text_chunks (List[str]): A list of text chunks to search through.
    embeddings (List[dict]): A list of embeddings for the text chunks.
    k (int): The number of top relevant text chunks to return. Default is 5.

    Returns:
    List[str]: A list of the top k most relevant text chunks based on the query.
    """
    # Create an embedding for the query
    query_embedding = create_embeddings(query).data[0].embedding
    similarity_scores = []  # Initialize a list to store similarity scores

    # Calculate similarity scores between the query embedding and each text chunk embedding
    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(np.array(query_embedding), np.array(chunk_embedding.embedding))
        similarity_scores.append((i, similarity_score))  # Append the index and similarity score

    # Sort the similarity scores in descending order
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    # Get the indices of the top k most similar text chunks
    top_indices = [index for index, _ in similarity_scores[:k]]
    # Return the top k most relevant text chunks
    return [text_chunks[index] for index in top_indices]
```

## Running a Query on Extracted Chunks

```python
# Load the validation data from a JSON file
with open('data/val.json') as f:
    data = json.load(f)

# Extract the first query from the validation data
query = data[0]['question']

# Perform semantic search to find the top 2 most relevant text chunks for the query
top_chunks = semantic_search(query, text_chunks, response.data, k=2)

# Print the query
print("Query:", query)

# Print the top 2 most relevant text chunks
for i, chunk in enumerate(top_chunks):
    print(f"Context {i + 1}:\n{chunk}\n=====================================")
```

## Generating a Response Based on Retrieved Chunks

```python
# Define the system prompt for the AI assistant
system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"

def generate_response(system_prompt, user_message, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Generates a response from the AI model based on the system prompt and user message.

    Args:
    system_prompt (str): The system prompt to guide the AI's behavior.
    user_message (str): The user's message or query.
    model (str): The model to be used for generating the response. Default is "meta-llama/Llama-2-7B-chat-hf".

    Returns:
    dict: The response from the AI model.
    """
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return response

# Create the user prompt based on the top chunks
user_prompt = "\n".join([f"Context {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(top_chunks)])
user_prompt = f"{user_prompt}\nQuestion: {query}"

# Generate AI response
ai_response = generate_response(system_prompt, user_prompt)
```

## Evaluating the AI Response
We compare the AI response with the expected answer and assign a score.

```python
# Define the system prompt for the evaluation system
evaluate_system_prompt = "You are an intelligent evaluation system tasked with assessing the AI assistant's responses. If the AI assistant's response is very close to the true response, assign a score of 1. If the response is incorrect or unsatisfactory in relation to the true response, assign a score of 0. If the response is partially aligned with the true response, assign a score of 0.5."

# Create the evaluation prompt by combining the user query, AI response, true response, and evaluation system prompt
evaluation_prompt = f"User Query: {query}\nAI Response:\n{ai_response.choices[0].message.content}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"

# Generate the evaluation response using the evaluation system prompt and evaluation prompt
evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)

# Print the evaluation response
print(evaluation_response.choices[0].message.content)
```
