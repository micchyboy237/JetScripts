## Context-Enriched Retrieval in RAG
Retrieval-Augmented Generation (RAG) enhances AI responses by retrieving relevant knowledge from external sources. Traditional retrieval methods return isolated text chunks, which can lead to incomplete answers.

To address this, we introduce Context-Enriched Retrieval, which ensures that retrieved information includes neighboring chunks for better coherence.

Steps in This Notebook:
- Data Ingestion: Extract text from a PDF.
- Chunking with Overlapping Context: Split text into overlapping chunks to preserve context.
- Embedding Creation: Convert text chunks into numerical representations.
- Context-Aware Retrieval: Retrieve relevant chunks along with their neighbors for better completeness.
- Response Generation: Use a language model to generate responses based on retrieved context.
- Evaluation: Assess the model's response accuracy.

## Setting Up the Environment
We begin by importing necessary libraries.

```python
import fitz
import os
import numpy as np
import json
from openai import OpenAI
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

```output
Number of text chunks: 42

First text chunk:
Understanding Artificial Intelligence
Chapter 1: Introduction to Artificial Intelligence
Artificial intelligence (AI) refers to the ability of a digital computer or computer-controlled robot
to perform tasks commonly associated with intelligent beings. The term is frequently applied to
the project of developing systems endowed with the intellectual processes characteristic of
humans, such as the ability to reason, discover meaning, generalize, or learn from past
experience. Over the past few decades, advancements in computing power and data availability
have significantly accelerated the development and deployment of AI.
Historical Context
The idea of artificial intelligence has existed for centuries, often depicted in myths and fiction.
However, the formal field of AI research began in the mid-20th century. The Dartmouth Workshop
in 1956 is widely considered the birthplace of AI. Early AI research focused on problem-solving
and symbolic methods. The 1980s saw a rise in exp
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

## Implementing Context-Aware Semantic Search
We modify retrieval to include neighboring chunks for better context.

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
def context_enriched_search(query, text_chunks, embeddings, k=1, context_size=1):
    """
    Retrieves the most relevant chunk along with its neighboring chunks.

    Args:
    query (str): Search query.
    text_chunks (List[str]): List of text chunks.
    embeddings (List[dict]): List of chunk embeddings.
    k (int): Number of relevant chunks to retrieve.
    context_size (int): Number of neighboring chunks to include.

    Returns:
    List[str]: Relevant text chunks with contextual information.
    """
    # Convert the query into an embedding vector
    query_embedding = create_embeddings(query).data[0].embedding
    similarity_scores = []

    # Compute similarity scores between query and each text chunk embedding
    for i, chunk_embedding in enumerate(embeddings):
        # Calculate cosine similarity between the query embedding and current chunk embedding
        similarity_score = cosine_similarity(np.array(query_embedding), np.array(chunk_embedding.embedding))
        # Store the index and similarity score as a tuple
        similarity_scores.append((i, similarity_score))

    # Sort chunks by similarity score in descending order (highest similarity first)
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # Get the index of the most relevant chunk
    top_index = similarity_scores[0][0]

    # Define the range for context inclusion
    # Ensure we don't go below 0 or beyond the length of text_chunks
    start = max(0, top_index - context_size)
    end = min(len(text_chunks), top_index + context_size + 1)

    # Return the relevant chunk along with its neighboring context chunks
    return [text_chunks[i] for i in range(start, end)]
```

## Running a Query with Context Retrieval
We now test the context-enriched retrieval.

```python
# Load the validation dataset from a JSON file
with open('data/val.json') as f:
    data = json.load(f)

# Extract the first question from the dataset to use as our query
query = data[0]['question']

# Retrieve the most relevant chunk and its neighboring chunks for context
# Parameters:
# - query: The question we're searching for
# - text_chunks: Our text chunks extracted from the PDF
# - response.data: The embeddings of our text chunks
# - k=1: Return the top match
# - context_size=1: Include 1 chunk before and after the top match for context
top_chunks = context_enriched_search(query, text_chunks, response.data, k=1, context_size=1)

# Print the query for reference
print("Query:", query)
# Print each retrieved chunk with a heading and separator
for i, chunk in enumerate(top_chunks):
    print(f"Context {i + 1}:\n{chunk}\n=====================================")
```

```output
Query: What is 'Explainable AI' and why is it considered important?
Context 1:
nt aligns with societal values. Education and awareness campaigns inform the public
about AI, its impacts, and its potential.
Chapter 19: AI and Ethics
Principles of Ethical AI
Ethical AI principles guide the development and deployment of AI systems to ensure they are fair,
transparent, accountable, and beneficial to society. Key principles include respect for human
rights, privacy, non-discrimination, and beneficence.


Addressing Bias in AI
AI systems can inherit and amplify biases present in the data they are trained on, leading to unfair
or discriminatory outcomes. Addressing bias requires careful data collection, algorithm design,
and ongoing monitoring and evaluation.
Transparency and Explainability
Transparency and explainability are essential for building trust in AI systems. Explainable AI (XAI)
techniques aim to make AI decisions more understandable, enabling users to assess their
fairness and accuracy.
Privacy and Data Protection
AI systems often rely on la
=====================================
Context 2:
systems. Explainable AI (XAI)
techniques aim to make AI decisions more understandable, enabling users to assess their
fairness and accuracy.
Privacy and Data Protection
AI systems often rely on large amounts of data, raising concerns about privacy and data
protection. Ensuring responsible data handling, implementing privacy-preserving techniques,
and complying with data protection regulations are crucial.
Accountability and Responsibility
Establishing accountability and responsibility for AI systems is essential for addressing potential
harms and ensuring ethical behavior. This includes defining roles and responsibilities for
developers, deployers, and users of AI systems.
Chapter 20: Building Trust in AI
Transparency and Explainability
Transparency and explainability are key to building trust in AI. Making AI systems understandable
and providing insights into their decision-making processes helps users assess their reliability
and fairness.
Robustness and Reliability

=====================================
Context 3:
to building trust in AI. Making AI systems understandable
and providing insights into their decision-making processes helps users assess their reliability
and fairness.
Robustness and Reliability
Ensuring that AI systems are robust and reliable is essential for building trust. This includes
testing and validating AI models, monitoring their performance, and addressing potential
vulnerabilities.
User Control and Agency
Empowering users with control over AI systems and providing them with agency in their
interactions with AI enhances trust. This includes allowing users to customize AI settings,
understand how their data is used, and opt out of AI-driven features.
Ethical Design and Development
Incorporating ethical considerations into the design and development of AI systems is crucial for
building trust. This includes conducting ethical impact assessments, engaging stakeholders, and
adhering to ethical guidelines and standards.
Public Engagement and Education
Engaging th
=====================================
```

## Generating a Response Using Retrieved Context
We now generate a response using LLM.

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

```output
Based on the evaluation criteria, I would assign a score of 0.8 to the AI assistant's response.

The response is very close to the true response, and it correctly conveys the main idea of Explainable AI (XAI) and its importance. The AI assistant's response is also well-structured and easy to understand, which is a positive aspect.

However, there are a few minor differences between the AI assistant's response and the true response. The AI assistant's response is slightly more detailed and provides additional points (1-4) that are not present in the true response. Additionally, the AI assistant's response uses more formal language and phrases, such as "In essence," which is not present in the true response.

Despite these minor differences, the AI assistant's response is still very close to the true response, and it effectively conveys the main idea of XAI and its importance. Therefore, I would assign a score of 0.8.
```
