# Introduction to Simple RAG

Retrieval-Augmented Generation (RAG) is a hybrid approach that combines information retrieval with generative models. It enhances the performance of language models by incorporating external knowledge, which improves accuracy and factual correctness.

In a Simple RAG setup, we follow these steps:

1. **Data Ingestion**: Load and preprocess the text data.
2. **Chunking**: Break the data into smaller chunks to improve retrieval performance.
3. **Embedding Creation**: Convert the text chunks into numerical representations using an embedding model.
4. **Semantic Search**: Retrieve relevant chunks based on a user query.
5. **Response Generation**: Use a language model to generate a response based on retrieved text.

This notebook implements a Simple RAG approach, evaluates the model’s response, and explores various improvements.

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

    mypdf = fitz.open(pdf_path)
    all_text = ""


    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]
        text = page.get_text("text")
        all_text += text

    return all_text
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
    chunks = []


    for i in range(0, len(text), n - overlap):

        chunks.append(text[i:i + n])

    return chunks
```

## Setting Up the OpenAI API Client
We initialize the OpenAI client to generate embeddings and responses.

```python

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")
)
```

## Extracting and Chunking Text from a PDF File
Now, we load the PDF, extract text, and split it into chunks.

```python

pdf_path = "data/AI_Information.pdf"


extracted_text = extract_text_from_pdf(pdf_path)


text_chunks = chunk_text(extracted_text, 1000, 200)


print("Number of text chunks:", len(text_chunks))


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

    response = client.embeddings.create(
        model=model,
        input=text
    )

    return response


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

    query_embedding = create_embeddings(query).data[0].embedding
    similarity_scores = []


    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(np.array(query_embedding), np.array(chunk_embedding.embedding))
        similarity_scores.append((i, similarity_score))


    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    top_indices = [index for index, _ in similarity_scores[:k]]

    return [text_chunks[index] for index in top_indices]
```

## Running a Query on Extracted Chunks

```python

with open('data/val.json') as f:
    data = json.load(f)


query = data[0]['question']


top_chunks = semantic_search(query, text_chunks, response.data, k=2)


print("Query:", query)


for i, chunk in enumerate(top_chunks):
    print(f"Context {i + 1}:\n{chunk}\n=====================================")
```

```output
Query: What is 'Explainable AI' and why is it considered important?
Context 1:
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
Context 2:
 incidents.
Environmental Monitoring
AI-powered environmental monitoring systems track air and water quality, detect pollution, and
support environmental protection efforts. These systems provide real-time data, identify
pollution sources, and inform environmental policies.
Chapter 15: The Future of AI Research
Advancements in Deep Learning
Continued advancements in deep learning are expected to drive further breakthroughs in AI.
Research is focused on developing more efficient and interpretable deep learning models, as well
as exploring new architectures and training techniques.
Explainable AI (XAI)
Explainable AI (XAI) aims to make AI systems more transparent and understandable. Research in
XAI focuses on developing methods for explaining AI decisions, enhancing trust, and improving
accountability.
AI and Neuroscience
The intersection of AI and neuroscience is a promising area of research. Understanding the
human brain can inspire new AI algorithms and architectures,
=====================================
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
Based on the evaluation criteria, I would assign a score of 0.8 to the AI assistant's response.

The AI assistant's response is very close to the true response, but there are some minor differences. The true response mentions "transparency" and "accountability" explicitly, which are not mentioned in the AI assistant's response. However, the overall meaning and content of the response are identical, and the AI assistant's response effectively conveys the importance of Explainable AI in building trust and ensuring fairness in AI systems.

Therefore, the score of 0.8 reflects the AI assistant's response being very close to the true response, but not perfectly aligned.
```
