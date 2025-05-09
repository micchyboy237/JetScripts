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

    mypdf = fitz.open(pdf_path)
    all_text = ""


    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]
        text = page.get_text("text")
        all_text += text

    return all_text
```

## Setting Up the OpenAI API Client
We initialize the OpenAI client to generate embeddings and responses.

```python

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")
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

    system_prompt = "Generate a concise and informative title for the given text."


    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk}
        ]
    )


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
    chunks = []


    for i in range(0, len(text), n - overlap):
        chunk = text[i:i + n]
        header = generate_chunk_header(chunk)
        chunks.append({"header": header, "text": chunk})

    return chunks
```

## Extracting and Chunking Text from a PDF File
Now, we load the PDF, extract text, and split it into chunks.

```python

pdf_path = "data/AI_Information.pdf"


extracted_text = extract_text_from_pdf(pdf_path)



text_chunks = chunk_text_with_headers(extracted_text, 1000, 200)


print("Sample Chunk:")
print("Header:", text_chunks[0]['header'])
print("Content:", text_chunks[0]['text'])
```

```output
Sample Chunk:
Header: "Introduction to Artificial Intelligence: Understanding the Foundations and Evolution"
Content: Understanding Artificial Intelligence
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

    response = client.embeddings.create(
        model=model,
        input=text
    )

    return response.data[0].embedding
```

```python

embeddings = []


for chunk in tqdm(text_chunks, desc="Generating embeddings"):

    text_embedding = create_embeddings(chunk["text"])

    header_embedding = create_embeddings(chunk["header"])

    embeddings.append({"header": chunk["header"], "text": chunk["text"], "embedding": text_embedding, "header_embedding": header_embedding})
```

```output
Generating embeddings: 100%|██████████| 42/42 [02:56<00:00,  4.21s/it]
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

    query_embedding = create_embeddings(query)

    similarities = []


    for chunk in chunks:

        sim_text = cosine_similarity(np.array(query_embedding), np.array(chunk["embedding"]))

        sim_header = cosine_similarity(np.array(query_embedding), np.array(chunk["header_embedding"]))

        avg_similarity = (sim_text + sim_header) / 2

        similarities.append((chunk, avg_similarity))


    similarities.sort(key=lambda x: x[1], reverse=True)

    return [x[0] for x in similarities[:k]]
```

## Running a Query on Extracted Chunks

```python

with open('data/val.json') as f:
    data = json.load(f)

query = data[0]['question']


top_chunks = semantic_search(query, embeddings, k=2)


print("Query:", query)
for i, chunk in enumerate(top_chunks):
    print(f"Header {i+1}: {chunk['header']}")
    print(f"Content:\n{chunk['text']}\n")
```

```output
Query: What is 'Explainable AI' and why is it considered important?
Header 1: "Building Trust in AI: Addressing Transparency, Explainability, and Accountability"
Content:
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


Header 2: "Building Trust in AI: Essential Factors for Reliability and Fairness"
Content:
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

```

## Generating a Response Based on Retrieved Chunks

```python

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


user_prompt = "\n".join([f"Header: {chunk['header']}\nContent:\n{chunk['text']}" for chunk in top_chunks])
user_prompt = f"{user_prompt}\nQuestion: {query}"


ai_response = generate_response(system_prompt, user_prompt)
```

## Evaluating the AI Response
We compare the AI response with the expected answer and assign a score.

```python

evaluate_system_prompt = """You are an intelligent evaluation system.
Assess the AI assistant's response based on the provided context.
- Assign a score of 1 if the response is very close to the true answer.
- Assign a score of 0.5 if the response is partially correct.
- Assign a score of 0 if the response is incorrect.
Return only the score (0, 0.5, or 1)."""


true_answer = data[0]['ideal_answer']


evaluation_prompt = f"""
User Query: {query}
AI Response: {ai_response}
True Answer: {true_answer}
{evaluate_system_prompt}
"""


evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)


print("Evaluation Score:", evaluation_response.choices[0].message.content)
```

```output
Evaluation Score: 0.5
```
