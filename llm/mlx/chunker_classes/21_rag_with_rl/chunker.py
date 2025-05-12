```python
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Chunker:
    def __init__(self, chunk_size=100):
        self.chunk_size = chunk_size
        self.vector_store = {}

    def load_documents(self, directory):
        documents = []
        for file in os.listdir(directory):
            if file.endswith(".txt"):
                with open(os.path.join(directory, file), 'r') as f:
                    documents.append(f.read())
        return documents

    def chunk_documents(self, documents):
        chunks = []
        for document in documents:
            for i in range(0, len(document), self.chunk_size):
                chunk = document[i:i + self.chunk_size]
                chunks.append(chunk)
        return chunks

    def preprocess_chunks(self, chunks):
        preprocessed_chunks = []
        for chunk in chunks:
            chunk = chunk.lower()
            chunk = ''.join(e for e in chunk if e.isalnum())
            preprocessed_chunks.append(chunk)
        return preprocessed_chunks

    def generate_embeddings(self, chunks, model="BAAI/bge-en-icl"):
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForCausalLM.from_pretrained(model)
        embeddings = []
        for chunk in chunks:
            inputs = tokenizer(chunk, return_tensors="pt")
            output = model(**inputs)
            embedding = output.last_hidden_state[:, 0, :]
            embeddings.append(embedding)
        return embeddings

    def evaluate_relevance(self, retrieved_chunks, ground_truth_chunks):
        relevance_scores = []
        for retrieved, ground_truth in zip(retrieved_chunks, ground_truth_chunks):
            relevance = cosine_similarity(
                generate_embeddings([retrieved])[0],
                generate_embeddings([ground_truth])[0]
            )
            relevance_scores.append(relevance)
        return np.mean(relevance_scores)

    def add_to_vector_store(self, embeddings, chunks):
        for embedding, chunk in zip(embeddings, chunks):
            self.vector_store[len(self.vector_store)] = {"embedding": embedding, "chunk": chunk}

    def rewrite_query(self, query, context_chunks, model="google/gemma-2-2b-it", max_tokens=100, temperature=0.3):
        rewrite_prompt = f"""
You are a query optimization assistant. Your task is to rewrite the given query to make it more effective
for retrieving relevant information. The query will be used for document retrieval. Original query: {query}
Based on the context retrieved so far:
{' '.join(context_chunks[:2]) if context_chunks else 'No context available yet'}
Rewrite the query to be more specific and targeted to retrieve better information.

Rewritten query:
"""
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": rewrite_prompt
                }
            ]
        )
        rewritten_query = response.choices[0].message.content.strip()
        return rewritten_query

    def generate_response(self, prompt):
        response = client.chat.completions.create(
            model="google/gemma-2-2b-it",
            max_tokens=100,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response.choices[0].message.content.strip()

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    return dot_product / (magnitude_a * magnitude_b)

def retrieve_relevant_chunks(query, context_chunks):
    relevance_scores = []
    for chunk in context_chunks:
        similarity = cosine_similarity(
            generate_embeddings([query])[0],
            generate_embeddings([chunk])[0]
        )
        relevance_scores.append(similarity)
    return np.argsort(relevance_scores)[-2:]

def filter_context(context_chunks):
    return context_chunks[:10]

def generate_response(prompt):
    response = client.chat.completions.create(
        model="google/gemma-2-2b-it",
        max_tokens=100,
        temperature=0.3,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.choices[0].message.content.strip()

def basic_rag_pipeline(query):
    relevant_chunks = retrieve_relevant_chunks(query, [])
    prompt = "You are a query optimization assistant. Your task is to generate a response to the given query. Original query: " + query
    response = generate_response(prompt)
    return response

def simple_retrieval(query, context_chunks):
    relevance_scores = []
    for chunk in context_chunks:
        similarity = cosine_similarity(
            generate_embeddings([query])[0],
            generate_embeddings([chunk])[0]
        )
        relevance_scores.append(similarity)
    return np.argsort(relevance_scores)[-2:]

def evaluate_relevance(retrieved_chunks, ground_truth_chunks):
    relevance_scores = []
    for retrieved, ground_truth in zip(retrieved_chunks, ground_truth_chunks):
        relevance = cosine_similarity(
            generate_embeddings([retrieved])[0],
            generate_embeddings([ground_truth])[0]
        )
        relevance_scores.append(relevance)
    return np.mean(relevance_scores)
```