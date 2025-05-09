## Introduction to Semantic Chunking
Text chunking is an essential step in Retrieval-Augmented Generation (RAG), where large text bodies are divided into meaningful segments to improve retrieval accuracy.
Unlike fixed-length chunking, semantic chunking splits text based on the content similarity between sentences.

### Breakpoint Methods:
- **Percentile**: Finds the Xth percentile of all similarity differences and splits chunks where the drop is greater than this value.
- **Standard Deviation**: Splits where similarity drops more than X standard deviations below the mean.
- **Interquartile Range (IQR)**: Uses the interquartile distance (Q3 - Q1) to determine split points.

This notebook implements semantic chunking **using the percentile method** and evaluates its performance on a sample text.

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
    Extracts text from a PDF file.

    Args:
    pdf_path (str): Path to the PDF file.

    Returns:
    str: Extracted text from the PDF.
    """

    mypdf = fitz.open(pdf_path)
    all_text = ""


    for page in mypdf:

        all_text += page.get_text("text") + " "


    return all_text.strip()


pdf_path = "data/AI_Information.pdf"


extracted_text = extract_text_from_pdf(pdf_path)


print(extracted_text[:500])
```

```output
Understanding Artificial Intelligence
Chapter 1: Introduction to Artificial Intelligence
Artificial intelligence (AI) refers to the ability of a digital computer or computer-controlled robot
to perform tasks commonly associated with intelligent beings. The term is frequently applied to
the project of developing systems endowed with the intellectual processes characteristic of
humans, such as the ability to reason, discover meaning, generalize, or learn from past
experience. Over the past f
```

## Setting Up the OpenAI API Client
We initialize the OpenAI client to generate embeddings and responses.

```python

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")
)
```

## Creating Sentence-Level Embeddings
We split text into sentences and generate embeddings.

```python
def get_embedding(text, model="BAAI/bge-en-icl"):
    """
    Creates an embedding for the given text using OpenAI.

    Args:
    text (str): Input text.
    model (str): Embedding model name.

    Returns:
    np.ndarray: The embedding vector.
    """
    response = client.embeddings.create(model=model, input=text)
    return np.array(response.data[0].embedding)


sentences = extracted_text.split(". ")


embeddings = [get_embedding(sentence) for sentence in sentences]

print(f"Generated {len(embeddings)} sentence embeddings.")
```

```output
Generated 257 sentence embeddings.
```

## Calculating Similarity Differences
We compute cosine similarity between consecutive sentences.

```python
def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.

    Args:
    vec1 (np.ndarray): First vector.
    vec2 (np.ndarray): Second vector.

    Returns:
    float: Cosine similarity.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


similarities = [cosine_similarity(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]
```

## Implementing Semantic Chunking
We implement three different methods for finding breakpoints.

```python
def compute_breakpoints(similarities, method="percentile", threshold=90):
    """
    Computes chunking breakpoints based on similarity drops.

    Args:
    similarities (List[float]): List of similarity scores between sentences.
    method (str): 'percentile', 'standard_deviation', or 'interquartile'.
    threshold (float): Threshold value (percentile for 'percentile', std devs for 'standard_deviation').

    Returns:
    List[int]: Indices where chunk splits should occur.
    """

    if method == "percentile":

        threshold_value = np.percentile(similarities, threshold)
    elif method == "standard_deviation":

        mean = np.mean(similarities)
        std_dev = np.std(similarities)

        threshold_value = mean - (threshold * std_dev)
    elif method == "interquartile":

        q1, q3 = np.percentile(similarities, [25, 75])

        threshold_value = q1 - 1.5 * (q3 - q1)
    else:

        raise ValueError("Invalid method. Choose 'percentile', 'standard_deviation', or 'interquartile'.")


    return [i for i, sim in enumerate(similarities) if sim < threshold_value]


breakpoints = compute_breakpoints(similarities, method="percentile", threshold=90)
```

## Splitting Text into Semantic Chunks
We split the text based on computed breakpoints.

```python
def split_into_chunks(sentences, breakpoints):
    """
    Splits sentences into semantic chunks.

    Args:
    sentences (List[str]): List of sentences.
    breakpoints (List[int]): Indices where chunking should occur.

    Returns:
    List[str]: List of text chunks.
    """
    chunks = []
    start = 0


    for bp in breakpoints:

        chunks.append(". ".join(sentences[start:bp + 1]) + ".")
        start = bp + 1


    chunks.append(". ".join(sentences[start:]))
    return chunks


text_chunks = split_into_chunks(sentences, breakpoints)


print(f"Number of semantic chunks: {len(text_chunks)}")


print("\nFirst text chunk:")
print(text_chunks[0])
```

```output
Number of semantic chunks: 231

First text chunk:
Understanding Artificial Intelligence
Chapter 1: Introduction to Artificial Intelligence
Artificial intelligence (AI) refers to the ability of a digital computer or computer-controlled robot
to perform tasks commonly associated with intelligent beings.
```

## Creating Embeddings for Semantic Chunks
We create embeddings for each chunk for later retrieval.

```python
def create_embeddings(text_chunks):
    """
    Creates embeddings for each text chunk.

    Args:
    text_chunks (List[str]): List of text chunks.

    Returns:
    List[np.ndarray]: List of embedding vectors.
    """

    return [get_embedding(chunk) for chunk in text_chunks]


chunk_embeddings = create_embeddings(text_chunks)
```

## Performing Semantic Search
We implement cosine similarity to retrieve the most relevant chunks.

```python
def semantic_search(query, text_chunks, chunk_embeddings, k=5):
    """
    Finds the most relevant text chunks for a query.

    Args:
    query (str): Search query.
    text_chunks (List[str]): List of text chunks.
    chunk_embeddings (List[np.ndarray]): List of chunk embeddings.
    k (int): Number of top results to return.

    Returns:
    List[str]: Top-k relevant chunks.
    """

    query_embedding = get_embedding(query)


    similarities = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]


    top_indices = np.argsort(similarities)[-k:][::-1]


    return [text_chunks[i] for i in top_indices]
```

```python

with open('data/val.json') as f:
    data = json.load(f)


query = data[0]['question']


top_chunks = semantic_search(query, text_chunks, chunk_embeddings, k=2)


print(f"Query: {query}")


for i, chunk in enumerate(top_chunks):
    print(f"Context {i+1}:\n{chunk}\n{'='*40}")
```

```output
Query: What is 'Explainable AI' and why is it considered important?
Context 1:

Explainable AI (XAI)
Explainable AI (XAI) aims to make AI systems more transparent and understandable. Research in
XAI focuses on developing methods for explaining AI decisions, enhancing trust, and improving
accountability.
========================================
Context 2:

Transparency and Explainability
Transparency and explainability are essential for building trust in AI systems. Explainable AI (XAI)
techniques aim to make AI decisions more understandable, enabling users to assess their
fairness and accuracy.
========================================
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


user_prompt = "\n".join([f"Context {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(top_chunks)])
user_prompt = f"{user_prompt}\nQuestion: {query}"


ai_response = generate_response(system_prompt, user_prompt)
```

## Evaluating the AI Response
We compare the AI response with the expected answer and assign a score.

```python

evaluate_system_prompt = "You are an intelligent evaluation system tasked with assessing the AI assistant's responses. If the AI assistant's response is very close to the true response, assign a score of 1. If the response is incorrect or unsatisfactory in relation to the true response, assign a score of 0. If the response is partially aligned with the true response, assign a score of 0.5."


evaluation_prompt = f"User Query: {query}\nAI Response:\n{ai_response.choices[0].message.content}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"


evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)


print(evaluation_response.choices[0].message.content)
```

```output
Based on the evaluation criteria, I would assign a score of 0.5 to the AI assistant's response.

The response is partially aligned with the true response, as it correctly identifies the main goal of Explainable AI (XAI) as making AI systems more transparent and understandable. However, it lacks some key details and nuances present in the true response. For example, the true response mentions the importance of assessing fairness and accuracy, which is not explicitly mentioned in the AI assistant's response. Additionally, the true response uses more precise language, such as "providing insights into how they make decisions," which is not present in the AI assistant's response.
```
