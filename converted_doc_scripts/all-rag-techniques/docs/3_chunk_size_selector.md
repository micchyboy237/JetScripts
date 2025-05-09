## Evaluating Chunk Sizes in Simple RAG

Choosing the right chunk size is crucial for improving retrieval accuracy in a Retrieval-Augmented Generation (RAG) pipeline. The goal is to balance retrieval performance with response quality.

This section evaluates different chunk sizes by:

1. Extracting text from a PDF.
2. Splitting text into chunks of varying sizes.
3. Creating embeddings for each chunk.
4. Retrieving relevant chunks for a query.
5. Generating a response using retrieved chunks.
6. Evaluating faithfulness and relevancy.
7. Comparing results for different chunk sizes.

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

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")
)
```

## Extracting Text from the PDF
First, we will extract text from the `AI_Information.pdf` file.

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

## Chunking the Extracted Text
To improve retrieval, we split the extracted text into overlapping chunks of different sizes.

```python
def chunk_text(text, n, overlap):
    """
    Splits text into overlapping chunks.

    Args:
    text (str): The text to be chunked.
    n (int): Number of characters per chunk.
    overlap (int): Overlapping characters between chunks.

    Returns:
    List[str]: A list of text chunks.
    """
    chunks = []
    for i in range(0, len(text), n - overlap):

        chunks.append(text[i:i + n])

    return chunks


chunk_sizes = [128, 256, 512]


text_chunks_dict = {size: chunk_text(extracted_text, size, size // 5) for size in chunk_sizes}


for size, chunks in text_chunks_dict.items():
    print(f"Chunk Size: {size}, Number of Chunks: {len(chunks)}")
```

```output
Chunk Size: 128, Number of Chunks: 326
Chunk Size: 256, Number of Chunks: 164
Chunk Size: 512, Number of Chunks: 82
```

## Creating Embeddings for Text Chunks
Embeddings convert text into numerical representations for similarity search.

```python
from tqdm import tqdm

def create_embeddings(texts, model="BAAI/bge-en-icl"):
    """
    Generates embeddings for a list of texts.

    Args:
    texts (List[str]): List of input texts.
    model (str): Embedding model.

    Returns:
    List[np.ndarray]: List of numerical embeddings.
    """

    response = client.embeddings.create(model=model, input=texts)

    return [np.array(embedding.embedding) for embedding in response.data]



chunk_embeddings_dict = {size: create_embeddings(chunks) for size, chunks in tqdm(text_chunks_dict.items(), desc="Generating Embeddings")}
```

```output
Generating Embeddings: 100%|██████████| 3/3 [00:11<00:00,  3.71s/it]
```

## Performing Semantic Search
We use cosine similarity to find the most relevant text chunks for a user query.

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
def retrieve_relevant_chunks(query, text_chunks, chunk_embeddings, k=5):
    """
    Retrieves the top-k most relevant text chunks.

    Args:
    query (str): User query.
    text_chunks (List[str]): List of text chunks.
    chunk_embeddings (List[np.ndarray]): Embeddings of text chunks.
    k (int): Number of top chunks to return.

    Returns:
    List[str]: Most relevant text chunks.
    """

    query_embedding = create_embeddings([query])[0]


    similarities = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]


    top_indices = np.argsort(similarities)[-k:][::-1]


    return [text_chunks[i] for i in top_indices]
```

```python

with open('data/val.json') as f:
    data = json.load(f)


query = data[3]['question']


retrieved_chunks_dict = {size: retrieve_relevant_chunks(query, text_chunks_dict[size], chunk_embeddings_dict[size]) for size in chunk_sizes}


print(retrieved_chunks_dict[256])
```

```output
['AI enables personalized medicine by analyzing individual patient data, predicting treatment \nresponses, and tailoring interventions. Personalized medicine enhances treatment effectiveness \nand reduces adverse effects. \nRobotic Surgery \nAI-powered robotic s', ' analyzing biological data, predicting drug \nefficacy, and identifying potential drug candidates. AI-powered systems reduce the time and cost \nof bringing new treatments to market. \nPersonalized Medicine \nAI enables personalized medicine by analyzing indiv', 'g \npatient outcomes, and assisting in treatment planning. AI-powered tools enhance accuracy, \nefficiency, and patient care. \nDrug Discovery and Development \nAI accelerates drug discovery and development by analyzing biological data, predicting drug \neffica', 'mains. \nThese applications include: \nHealthcare \nAI is transforming healthcare through applications such as medical diagnosis, drug discovery, \npersonalized medicine, and robotic surgery. AI-powered tools can analyze medical images, \npredict patient outcom', 'Personalized Learning \nAI enables personalized learning experiences by adapting to individual student needs and \nlearning styles. AI-powered platforms provide customized content, feedback, and pacing, \nenhancing student engagement and outcomes. \nAdaptive A']
```

## Generating a Response Based on Retrieved Chunks
Let's  generate a response based on the retrieved text for chunk size `256`.

```python

system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"

def generate_response(query, system_prompt, retrieved_chunks, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Generates an AI response based on retrieved chunks.

    Args:
    query (str): User query.
    retrieved_chunks (List[str]): List of retrieved text chunks.
    model (str): AI model.

    Returns:
    str: AI-generated response.
    """

    context = "\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])


    user_prompt = f"{context}\n\nQuestion: {query}"


    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )


    return response.choices[0].message.content


ai_responses_dict = {size: generate_response(query, system_prompt, retrieved_chunks_dict[size]) for size in chunk_sizes}


print(ai_responses_dict[256])
```

```output
AI contributes to personalized medicine by analyzing individual patient data, predicting treatment responses, and tailoring interventions. This enables personalized medicine to enhance treatment effectiveness and reduce adverse effects.
```

## Evaluating the AI Response
We score responses based on faithfulness and relevancy using powerfull llm

```python

SCORE_FULL = 1.0
SCORE_PARTIAL = 0.5
SCORE_NONE = 0.0
```

```python

FAITHFULNESS_PROMPT_TEMPLATE = """
Evaluate the faithfulness of the AI response compared to the true answer.
User Query: {question}
AI Response: {response}
True Answer: {true_answer}

Faithfulness measures how well the AI response aligns with facts in the true answer, without hallucinations.

INSTRUCTIONS:
- Score STRICTLY using only these values:
    * {full} = Completely faithful, no contradictions with true answer
    * {partial} = Partially faithful, minor contradictions
    * {none} = Not faithful, major contradictions or hallucinations
- Return ONLY the numerical score ({full}, {partial}, or {none}) with no explanation or additional text.
"""
```

```python
RELEVANCY_PROMPT_TEMPLATE = """
Evaluate the relevancy of the AI response to the user query.
User Query: {question}
AI Response: {response}

Relevancy measures how well the response addresses the user's question.

INSTRUCTIONS:
- Score STRICTLY using only these values:
    * {full} = Completely relevant, directly addresses the query
    * {partial} = Partially relevant, addresses some aspects
    * {none} = Not relevant, fails to address the query
- Return ONLY the numerical score ({full}, {partial}, or {none}) with no explanation or additional text.
"""
```

```python
def evaluate_response(question, response, true_answer):
        """
        Evaluates the quality of an AI-generated response based on faithfulness and relevancy.

        Args:
        question (str): The user's original question.
        response (str): The AI-generated response being evaluated.
        true_answer (str): The correct answer used as ground truth.

        Returns:
        Tuple[float, float]: A tuple containing (faithfulness_score, relevancy_score).
                                                Each score is one of: 1.0 (full), 0.5 (partial), or 0.0 (none).
        """

        faithfulness_prompt = FAITHFULNESS_PROMPT_TEMPLATE.format(
                question=question,
                response=response,
                true_answer=true_answer,
                full=SCORE_FULL,
                partial=SCORE_PARTIAL,
                none=SCORE_NONE
        )

        relevancy_prompt = RELEVANCY_PROMPT_TEMPLATE.format(
                question=question,
                response=response,
                full=SCORE_FULL,
                partial=SCORE_PARTIAL,
                none=SCORE_NONE
        )


        faithfulness_response = client.chat.completions.create(
               model="meta-llama/Llama-3.2-3B-Instruct",
                temperature=0,
                messages=[
                        {"role": "system", "content": "You are an objective evaluator. Return ONLY the numerical score."},
                        {"role": "user", "content": faithfulness_prompt}
                ]
        )


        relevancy_response = client.chat.completions.create(
                model="meta-llama/Llama-3.2-3B-Instruct",
                temperature=0,
                messages=[
                        {"role": "system", "content": "You are an objective evaluator. Return ONLY the numerical score."},
                        {"role": "user", "content": relevancy_prompt}
                ]
        )


        try:
                faithfulness_score = float(faithfulness_response.choices[0].message.content.strip())
        except ValueError:
                print("Warning: Could not parse faithfulness score, defaulting to 0")
                faithfulness_score = 0.0

        try:
                relevancy_score = float(relevancy_response.choices[0].message.content.strip())
        except ValueError:
                print("Warning: Could not parse relevancy score, defaulting to 0")
                relevancy_score = 0.0

        return faithfulness_score, relevancy_score


true_answer = data[3]['ideal_answer']


faithfulness, relevancy = evaluate_response(query, ai_responses_dict[256], true_answer)
faithfulness2, relevancy2 = evaluate_response(query, ai_responses_dict[128], true_answer)


print(f"Faithfulness Score (Chunk Size 256): {faithfulness}")
print(f"Relevancy Score (Chunk Size 256): {relevancy}")

print(f"\n")

print(f"Faithfulness Score (Chunk Size 128): {faithfulness2}")
print(f"Relevancy Score (Chunk Size 128): {relevancy2}")
```

```output
Faithfulness Score (Chunk Size 256): 0.5
Relevancy Score (Chunk Size 256): 0.5


Faithfulness Score (Chunk Size 128): 0.5
Relevancy Score (Chunk Size 128): 0.5
```
