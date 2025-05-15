import numpy as np
from scipy import spatial
import faiss
import time
import pandas as pd
from chunker import ChunkText

def calculate_cosine_similarity(v1, v2):
    """Calculate the cosine similarity between two vectors."""
    return 1 - spatial.distance.cosine(v1, v2)

def chunk_text(text, chunk_size, chunk_overlap):
    """Splits a single text document into overlapping chunks based on word count."""
    words = text.split()  # Split the text into a list of individual words
    total_words = len(words)  # Calculate the total number of words in the text
    chunks = []  # Initialize an empty list to store the generated chunks
    start_index = 0  # Initialize the starting word index for the first chunk

    # --- Input Validation ---
    # Ensure chunk_size is a positive integer. if not isinstance(chunk_size, int) or chunk_size <= 0:
    print(f"  Warning: Invalid chunk_size ({chunk_size}).  Must be a positive integer. Returning the whole text as one chunk.")  # Return the whole text as one chunk
    chunks.append(text)
    return chunks  # Return the complete list of text chunks

def run_and_evaluate(corpus_texts, test_query, true_answer_for_query, chunk_size, chunk_overlap, top_k, strategies):
    """Run the RAG system with different strategies and evaluate the results."""
    all_results = []
    for strategy in strategies:
        # Prepare data (chunking/embedding/indexing)
        chunked_texts = chunk_text(" ".join(corpus_texts), chunk_size, chunk_overlap)
        embeddings = []
        for text in chunked_texts:
            embedding = NEBIUS_EMBEDDING_MODEL.encode(text)
            embeddings.append(embedding)

        index = faiss.IndexFlatL2(embeddings)
        index.add(embeddings)

        # Test RAG strategies
        results = []
        for _ in range(top_k):
            query_embedding = NEBIUS_EMBEDDING_MODEL.encode(test_query)
            D, I = index.search(query_embedding, 1)
            chunk_index = I[0][0]
            chunk = chunked_texts[chunk_index]
            rewritten_query = NEBIUS_GENERATION_MODEL.generate(chunk, test_query)
            generated_answer = NEBIUS_GENERATION_MODEL.generate(chunk, rewritten_query)
            faithfulness_score = NEBIUS_EVALUATION_MODEL.evaluate(generated_answer, true_answer_for_query, "Faithfulness")
            relevancy_score = NEBIUS_EVALUATION_MODEL.evaluate(generated_answer, test_query, "Relevancy")
            similarity_score = calculate_cosine_similarity(query_embedding, generated_answer)
            average_score = (faithfulness_score + relevancy_score + similarity_score) / 3
            results.append({
                "strategy": strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "top_k": top_k,
                "result": generated_answer,
                "faithfulness": faithfulness_score,
                "relevancy": relevancy_score,
                "similarity": similarity_score,
                "average_score": average_score
            })

        # Evaluate & store results
        for result in results:
            all_results.append(result)
    return all_results

# Example usage:
corpus_texts = ["This is a sample text.", "Another sample text."]
test_query = "What is the meaning of life?"
true_answer_for_query = "The meaning of life is to find your purpose."
chunk_size = 150
chunk_overlap = 30
top_k = 10
strategies = ["Simple RAG", "Query Rewrite RAG", "Rerank (Simulated) RAG"]

results = run_and_evaluate(corpus_texts, test_query, true_answer_for_query, chunk_size, chunk_overlap, top_k, strategies)
print(pd.DataFrame(results))