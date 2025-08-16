from jet.logger import CustomLogger
from openai import Ollama
import fitz
import json
import numpy as np
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

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
    Extracts text from a PDF file and prints the first `num_chars` characters.

    Args:
    pdf_path (str): Path to the PDF file.

    Returns:
    str: Extracted text from the PDF.
    """
    mypdf = fitz.open(pdf_path)
    all_text = ""  # Initialize an empty string to store the extracted text

    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]  # Get the page
        text = page.get_text("text")  # Extract text from the page
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
    chunks = []  # Initialize an empty list to store the chunks

    for i in range(0, len(text), n - overlap):
        chunks.append(text[i:i + n])

    return chunks  # Return the list of text chunks

"""
## Setting Up the Ollama API Client
We initialize the Ollama client to generate embeddings and responses.
"""
logger.info("## Setting Up the Ollama API Client")

client = Ollama(
    base_url="https://api.studio.nebius.com/v1/",
#     api_key=os.getenv("OPENAI_API_KEY")  # Retrieve the API key from environment variables
)

"""
## Extracting and Chunking Text from a PDF File
Now, we load the PDF, extract text, and split it into chunks.
"""
logger.info("## Extracting and Chunking Text from a PDF File")

pdf_path = f"{GENERATED_DIR}/AI_Information.pdf"

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

def create_embeddings(text, model="BAAI/bge-en-icl"):
    """
    Creates embeddings for the given text using the specified Ollama model.

    Args:
    text (str): The input text for which embeddings are to be created.
    model (str): The model to be used for creating embeddings. Default is "BAAI/bge-en-icl".

    Returns:
    dict: The response from the Ollama API containing the embeddings.
    """
    response = client.embeddings.create(
        model=model,
        input=text
    )

    return response  # Return the response containing the embeddings

response = create_embeddings(text_chunks)

"""
## Performing Semantic Search
We implement cosine similarity to find the most relevant text chunks for a user query.
"""
logger.info("## Performing Semantic Search")

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
    similarity_scores = []  # Initialize a list to store similarity scores

    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(np.array(query_embedding), np.array(chunk_embedding.embedding))
        similarity_scores.append((i, similarity_score))  # Append the index and similarity score

    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [index for index, _ in similarity_scores[:k]]
    return [text_chunks[index] for index in top_indices]

"""
## Running a Query on Extracted Chunks
"""
logger.info("## Running a Query on Extracted Chunks")

with open('data/val.json') as f:
    data = json.load(f)

query = data[0]['question']

top_chunks = semantic_search(query, text_chunks, response.data, k=2)

logger.debug("Query:", query)

for i, chunk in enumerate(top_chunks):
    logger.debug(f"Context {i + 1}:\n{chunk}\n=====================================")

"""
## Generating a Response Based on Retrieved Chunks
"""
logger.info("## Generating a Response Based on Retrieved Chunks")

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

"""
## Evaluating the AI Response
We compare the AI response with the expected answer and assign a score.
"""
logger.info("## Evaluating the AI Response")

evaluate_system_prompt = "You are an intelligent evaluation system tasked with assessing the AI assistant's responses. If the AI assistant's response is very close to the true response, assign a score of 1. If the response is incorrect or unsatisfactory in relation to the true response, assign a score of 0. If the response is partially aligned with the true response, assign a score of 0.5."

evaluation_prompt = f"User Query: {query}\nAI Response:\n{ai_response.choices[0].message.content}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"

evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)

logger.debug(evaluation_response.choices[0].message.content)

logger.info("\n\n[DONE]", bright=True)