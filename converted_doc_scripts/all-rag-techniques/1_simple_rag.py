from jet.llm.mlx.base import MLX
from jet.llm.utils.embeddings import get_embedding_function
from jet.logger import CustomLogger
import pypdf  # Replace fitz with pypdf
import json
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

DATA_DIR = os.path.join(script_dir, "data")

"""
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
"""
logger.info("# Introduction to Simple RAG")

"""
## Extracting Text from a PDF File
To implement RAG, we first need a source of textual data. In this case, we extract text from a PDF file using the PyMuPDF library.
"""
logger.info("## Extracting Text from a PDF File")


def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    all_text = ""  # Initialize an empty string to store the extracted text
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)  # Create a PdfReader object
        for page in reader.pages:
            text = page.extract_text() or ""  # Extract text from the page, handle None
            all_text += text  # Append the extracted text to the all_text string
    return all_text  # Return the extracted text


"""
## Chunking the Extracted Text
Once we have the extracted text, we divide it into smaller, overlapping chunks to improve retrieval accuracy.
"""
logger.info("## Chunking the Extracted Text")


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


"""
## Setting Up the LLM API Client
We initialize the LLM client to generate embeddings and responses.
"""
logger.info("## Setting Up the LLM API Client")
mlx = MLX()
embed_func = get_embedding_function("mxbai-embed-large")

"""
## Extracting and Chunking Text from a PDF File
Now, we load the PDF, extract text, and split it into chunks.
"""
logger.info("## Extracting and Chunking Text from a PDF File")
pdf_path = f"{DATA_DIR}/AI_Information.pdf"
extracted_text = extract_text_from_pdf(pdf_path)
text_chunks = chunk_text(extracted_text, 1000, 200)
logger.debug("Number of text chunks:", len(text_chunks))
logger.debug("\nFirst text chunk:")
logger.debug(text_chunks[0])

"""
## Creating Embeddings for Text Chunks
Embeddings transform text into numerical vectors, which allow for efficient similarity search.
"""
logger.info("## Creating Embeddings for Text Chunks")


def create_embeddings(text):
    response = embed_func(text)
    return response


response = create_embeddings(text_chunks)

"""
## Performing Semantic Search
We implement cosine similarity to find the most relevant text chunks for a user query.
"""
logger.info("## Performing Semantic Search")


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


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
    query_embedding = embed_func(query)
    similarity_scores = []
    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(
            np.array(query_embedding), np.array(chunk_embedding))
        similarity_scores.append((i, similarity_score))
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [index for index, _ in similarity_scores[:k]]
    return [text_chunks[index] for index in top_indices]


"""
## Running a Query on Extracted Chunks
"""
logger.info("## Running a Query on Extracted Chunks")

with open(f"{DATA_DIR}/val.json") as f:
    data = json.load(f)

query = data[0]['question']
top_chunks = semantic_search(query, text_chunks, response, k=2)
logger.debug("Query:", query)
for i, chunk in enumerate(top_chunks):
    logger.debug(
        f"Context {i + 1}:\n{chunk}\n=====================================")

"""
## Generating a Response Based on Retrieved Chunks
"""
logger.info("## Generating a Response Based on Retrieved Chunks")

system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"


def generate_response(system_prompt, user_message, model="meta-llama/Llama-3.2-3B-Instruct"):
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        # temperature=0,
    )
    return response


user_prompt = "\n".join(
    [f"Context {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(top_chunks)])
user_prompt = f"{user_prompt}\nQuestion: {query}"

ai_response = generate_response(system_prompt, user_prompt)
logger.success(ai_response)

"""
## Evaluating the AI Response
We compare the AI response with the expected answer and assign a score.
"""
logger.info("## Evaluating the AI Response")

evaluate_system_prompt = "You are an intelligent evaluation system tasked with assessing the AI assistant's responses. If the AI assistant's response is very close to the true response, assign a score of 1. If the response is incorrect or unsatisfactory in relation to the true response, assign a score of 0. If the response is partially aligned with the true response, assign a score of 0.5."

evaluation_prompt = f"User Query: {query}\nAI Response:\n{ai_response}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"

evaluation_response = generate_response(
    evaluate_system_prompt, evaluation_prompt)

logger.success(evaluation_response)

logger.info("\n\n[DONE]", bright=True)
