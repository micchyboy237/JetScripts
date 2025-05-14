# Contextual Chunk Headers (CCH) in Simple RAG

Retrieval-Augmented Generation (RAG) improves the factual accuracy of language models by retrieving relevant external knowledge before generating a response. However, standard chunking often loses important context, making retrieval less effective.

Contextual Chunk Headers (CCH) enhance RAG by prepending high-level context (like document titles or section headers) to each chunk before embedding them. This improves retrieval quality and prevents out-of-context responses.

## Steps in this Notebook:

1. **Data Ingestion**: Load and preprocess the text data.
2. **Chunking with Contextual Headers**: Extract section titles and prepend them to chunks.
3. **Embedding Creation**: Convert context-enhanced chunks into numerical representations.
4. **Semantic Search**: Retrieve relevant chunks based on a user query.
5. **Response Generation**: Use a language model to generate a response from retrieved text.
6. **Evaluation**: Assess response accuracy using a scoring system.

## Setting Up the Environment
We begin by importing necessary libraries.

```python
import os
import numpy as np
import json
from openai import OpenAI
import fitz
from tqdm import tqdm
```

## Extracting Text and Identifying Section Headers
We extract text from a PDF while also identifying section titles (potential headers for chunks).

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

## Setting Up the OpenAI API Client
We initialize the OpenAI client to generate embeddings and responses.

```python
# Initialize the OpenAI client with the base URL and API key
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")  # Retrieve the API key from environment variables
)
```

## Chunking Text with Contextual Headers
To improve retrieval, we generate descriptive headers for each chunk using an LLM model.

```python
def generate_chunk_header(chunk, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Generates a title/header for a given text chunk using an LLM.

    Args:
    chunk (str): The text chunk to summarize as a header.
    model (str): The model to be used for generating the header. Default is "meta-llama/Llama-3.2-3B-Instruct".

    Returns:
    str: Generated header/title.
    """
    # Define the system prompt to guide the AI's behavior
    system_prompt = "Generate a concise and informative title for the given text."
    
    # Generate a response from the AI model based on the system prompt and text chunk
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk}
        ]
    )

    # Return the generated header/title, stripping any leading/trailing whitespace
    return response.choices[0].message.content.strip()
```

```python
def chunk_text_with_headers(text, n, overlap):
    """
    Chunks text into smaller segments and generates headers.

    Args:
    text (str): The full text to be chunked.
    n (int): The chunk size in characters.
    overlap (int): Overlapping characters between chunks.

    Returns:
    List[dict]: A list of dictionaries with 'header' and 'text' keys.
    """
    chunks = []  # Initialize an empty list to store chunks

    # Iterate through the text with the specified chunk size and overlap
    for i in range(0, len(text), n - overlap):
        chunk = text[i:i + n]  # Extract a chunk of text
        header = generate_chunk_header(chunk)  # Generate a header for the chunk using LLM
        chunks.append({"header": header, "text": chunk})  # Append the header and chunk to the list

    return chunks  # Return the list of chunks with headers
```

## Extracting and Chunking Text from a PDF File
Now, we load the PDF, extract text, and split it into chunks.

```python
# Define the PDF file path
pdf_path = "data/AI_Information.pdf"

# Extract text from the PDF file
extracted_text = extract_text_from_pdf(pdf_path)

# Chunk the extracted text with headers
# We use a chunk size of 1000 characters and an overlap of 200 characters
text_chunks = chunk_text_with_headers(extracted_text, 1000, 200)

# Print a sample chunk with its generated header
print("Sample Chunk:")
print("Header:", text_chunks[0]['header'])
print("Content:", text_chunks[0]['text'])
```

## Creating Embeddings for Headers and Text
We create embeddings for both headers and text to improve retrieval accuracy.

```python
def create_embeddings(text, model="BAAI/bge-en-icl"):
    """
    Creates embeddings for the given text.

    Args:
    text (str): The input text to be embedded.
    model (str): The embedding model to be used. Default is "BAAI/bge-en-icl".

    Returns:
    dict: The response containing the embedding for the input text.
    """
    # Create embeddings using the specified model and input text
    response = client.embeddings.create(
        model=model,
        input=text
    )
    # Return the embedding from the response
    return response.data[0].embedding
```

```python
# Generate embeddings for each chunk
embeddings = []  # Initialize an empty list to store embeddings

# Iterate through each text chunk with a progress bar
for chunk in tqdm(text_chunks, desc="Generating embeddings"):
    # Create an embedding for the chunk's text
    text_embedding = create_embeddings(chunk["text"])
    # Create an embedding for the chunk's header
    header_embedding = create_embeddings(chunk["header"])
    # Append the chunk's header, text, and their embeddings to the list
    embeddings.append({"header": chunk["header"], "text": chunk["text"], "embedding": text_embedding, "header_embedding": header_embedding})
```

## Performing Semantic Search
We implement cosine similarity to find the most relevant text chunks for a user query.

```python
def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.

    Args:
    vec1 (np.ndarray): First vector.
    vec2 (np.ndarray): Second vector.

    Returns:
    float: Cosine similarity score.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
```

```python
def semantic_search(query, chunks, k=5):
    """
    Searches for the most relevant chunks based on a query.

    Args:
    query (str): User query.
    chunks (List[dict]): List of text chunks with embeddings.
    k (int): Number of top results.

    Returns:
    List[dict]: Top-k most relevant chunks.
    """
    # Create an embedding for the query
    query_embedding = create_embeddings(query)

    similarities = []  # Initialize a list to store similarity scores
    
    # Iterate through each chunk to calculate similarity scores
    for chunk in chunks:
        # Compute cosine similarity between query embedding and chunk text embedding
        sim_text = cosine_similarity(np.array(query_embedding), np.array(chunk["embedding"]))
        # Compute cosine similarity between query embedding and chunk header embedding
        sim_header = cosine_similarity(np.array(query_embedding), np.array(chunk["header_embedding"]))
        # Calculate the average similarity score
        avg_similarity = (sim_text + sim_header) / 2
        # Append the chunk and its average similarity score to the list
        similarities.append((chunk, avg_similarity))

    # Sort the chunks based on similarity scores in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)
    # Return the top-k most relevant chunks
    return [x[0] for x in similarities[:k]]
```

## Running a Query on Extracted Chunks

```python
# Load validation data
with open('data/val.json') as f:
    data = json.load(f)

query = data[0]['question']

# Retrieve the top 2 most relevant text chunks
top_chunks = semantic_search(query, embeddings, k=2)

# Print the results
print("Query:", query)
for i, chunk in enumerate(top_chunks):
    print(f"Header {i+1}: {chunk['header']}")
    print(f"Content:\n{chunk['text']}\n")
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
user_prompt = "\n".join([f"Header: {chunk['header']}\nContent:\n{chunk['text']}" for chunk in top_chunks])
user_prompt = f"{user_prompt}\nQuestion: {query}"

# Generate AI response
ai_response = generate_response(system_prompt, user_prompt)
```

## Evaluating the AI Response
We compare the AI response with the expected answer and assign a score.

```python
# Define evaluation system prompt
evaluate_system_prompt = """You are an intelligent evaluation system. 
Assess the AI assistant's response based on the provided context. 
- Assign a score of 1 if the response is very close to the true answer. 
- Assign a score of 0.5 if the response is partially correct. 
- Assign a score of 0 if the response is incorrect.
Return only the score (0, 0.5, or 1)."""

# Extract the ground truth answer from validation data
true_answer = data[0]['ideal_answer']

# Construct evaluation prompt
evaluation_prompt = f"""
User Query: {query}
AI Response: {ai_response}
True Answer: {true_answer}
{evaluate_system_prompt}
"""

# Generate evaluation score
evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)

# Print the evaluation score
print("Evaluation Score:", evaluation_response.choices[0].message.content)
```
