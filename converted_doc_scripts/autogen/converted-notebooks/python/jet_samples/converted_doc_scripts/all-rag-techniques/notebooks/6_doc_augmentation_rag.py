from openai import Ollama
from tqdm import tqdm
import fitz
import json
import numpy as np
import os
import re

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
# Document Augmentation RAG with Question Generation

This notebook implements an enhanced RAG approach using document augmentation through question generation. By generating relevant questions for each text chunk, we improve the retrieval process, leading to better responses from the language model.

In this implementation, we follow these steps:

1. **Data Ingestion**: Extract text from a PDF file.
2. **Chunking**: Split the text into manageable chunks.
3. **Question Generation**: Generate relevant questions for each chunk.
4. **Embedding Creation**: Create embeddings for both chunks and generated questions.
5. **Vector Store Creation**: Build a simple vector store using NumPy.
6. **Semantic Search**: Retrieve relevant chunks and questions for user queries.
7. **Response Generation**: Generate answers based on retrieved content.
8. **Evaluation**: Assess the quality of the generated responses.

## Setting Up the Environment
We begin by importing necessary libraries.
"""
logger.info("# Document Augmentation RAG with Question Generation")


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
## Generating Questions for Text Chunks
This is the key enhancement over simple RAG. We generate questions that could be answered by each text chunk.
"""
logger.info("## Generating Questions for Text Chunks")

def generate_questions(text_chunk, num_questions=5, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Generates relevant questions that can be answered from the given text chunk.

    Args:
    text_chunk (str): The text chunk to generate questions from.
    num_questions (int): Number of questions to generate.
    model (str): The model to use for question generation.

    Returns:
    List[str]: List of generated questions.
    """
    system_prompt = "You are an expert at generating relevant questions from text. Create concise questions that can be answered using only the provided text. Focus on key information and concepts."

    user_prompt = f"""
    Based on the following text, generate {num_questions} different questions that can be answered using only this text:

    {text_chunk}

    Format your response as a numbered list of questions only, with no additional text.
    """

    response = client.chat.completions.create(
        model=model,
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    questions_text = response.choices[0].message.content.strip()
    questions = []

    for line in questions_text.split('\n'):
        cleaned_line = re.sub(r'^\d+\.\s*', '', line.strip())
        if cleaned_line and cleaned_line.endswith('?'):
            questions.append(cleaned_line)

    return questions

"""
## Creating Embeddings for Text
We generate embeddings for both text chunks and generated questions.
"""
logger.info("## Creating Embeddings for Text")

def create_embeddings(text, model="BAAI/bge-en-icl"):
    """
    Creates embeddings for the given text using the specified Ollama model.

    Args:
    text (str): The input text for which embeddings are to be created.
    model (str): The model to be used for creating embeddings.

    Returns:
    dict: The response from the Ollama API containing the embeddings.
    """
    response = client.embeddings.create(
        model=model,
        input=text
    )

    return response  # Return the response containing the embeddings

"""
## Building a Simple Vector Store
We'll implement a simple vector store using NumPy.
"""
logger.info("## Building a Simple Vector Store")

class SimpleVectorStore:
    """
    A simple vector store implementation using NumPy.
    """
    def __init__(self):
        """
        Initialize the vector store.
        """
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add_item(self, text, embedding, metadata=None):
        """
        Add an item to the vector store.

        Args:
        text (str): The original text.
        embedding (List[float]): The embedding vector.
        metadata (dict, optional): Additional metadata.
        """
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})

    def similarity_search(self, query_embedding, k=5):
        """
        Find the most similar items to a query embedding.

        Args:
        query_embedding (List[float]): Query embedding vector.
        k (int): Number of results to return.

        Returns:
        List[Dict]: Top k most similar items with their texts and metadata.
        """
        if not self.vectors:
            return []

        query_vector = np.array(query_embedding)

        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": score
            })

        return results

"""
## Processing Documents with Question Augmentation
Now we'll put everything together to process documents, generate questions, and build our augmented vector store.
"""
logger.info("## Processing Documents with Question Augmentation")

def process_document(pdf_path, chunk_size=1000, chunk_overlap=200, questions_per_chunk=5):
    """
    Process a document with question augmentation.

    Args:
    pdf_path (str): Path to the PDF file.
    chunk_size (int): Size of each text chunk in characters.
    chunk_overlap (int): Overlap between chunks in characters.
    questions_per_chunk (int): Number of questions to generate per chunk.

    Returns:
    Tuple[List[str], SimpleVectorStore]: Text chunks and vector store.
    """
    logger.debug("Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_path)

    logger.debug("Chunking text...")
    text_chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)
    logger.debug(f"Created {len(text_chunks)} text chunks")

    vector_store = SimpleVectorStore()

    logger.debug("Processing chunks and generating questions...")
    for i, chunk in enumerate(tqdm(text_chunks, desc="Processing Chunks")):
        chunk_embedding_response = create_embeddings(chunk)
        chunk_embedding = chunk_embedding_response.data[0].embedding

        vector_store.add_item(
            text=chunk,
            embedding=chunk_embedding,
            metadata={"type": "chunk", "index": i}
        )

        questions = generate_questions(chunk, num_questions=questions_per_chunk)

        for j, question in enumerate(questions):
            question_embedding_response = create_embeddings(question)
            question_embedding = question_embedding_response.data[0].embedding

            vector_store.add_item(
                text=question,
                embedding=question_embedding,
                metadata={"type": "question", "chunk_index": i, "original_chunk": chunk}
            )

    return text_chunks, vector_store

"""
## Extracting and Processing the Document
"""
logger.info("## Extracting and Processing the Document")

pdf_path = f"{GENERATED_DIR}/AI_Information.pdf"

text_chunks, vector_store = process_document(
    pdf_path,
    chunk_size=1000,
    chunk_overlap=200,
    questions_per_chunk=3
)

logger.debug(f"Vector store contains {len(vector_store.texts)} items")

"""
## Performing Semantic Search
We implement a semantic search function similar to the simple RAG implementation but adapted to our augmented vector store.
"""
logger.info("## Performing Semantic Search")

def semantic_search(query, vector_store, k=5):
    """
    Performs semantic search using the query and vector store.

    Args:
    query (str): The search query.
    vector_store (SimpleVectorStore): The vector store to search in.
    k (int): Number of results to return.

    Returns:
    List[Dict]: Top k most relevant items.
    """
    query_embedding_response = create_embeddings(query)
    query_embedding = query_embedding_response.data[0].embedding

    results = vector_store.similarity_search(query_embedding, k=k)

    return results

"""
## Running a Query on the Augmented Vector Store
"""
logger.info("## Running a Query on the Augmented Vector Store")

with open('data/val.json') as f:
    data = json.load(f)

query = data[0]['question']

search_results = semantic_search(query, vector_store, k=5)

logger.debug("Query:", query)
logger.debug("\nSearch Results:")

chunk_results = []
question_results = []

for result in search_results:
    if result["metadata"]["type"] == "chunk":
        chunk_results.append(result)
    else:
        question_results.append(result)

logger.debug("\nRelevant Document Chunks:")
for i, result in enumerate(chunk_results):
    logger.debug(f"Context {i + 1} (similarity: {result['similarity']:.4f}):")
    logger.debug(result["text"][:300] + "...")
    logger.debug("=====================================")

logger.debug("\nMatched Questions:")
for i, result in enumerate(question_results):
    logger.debug(f"Question {i + 1} (similarity: {result['similarity']:.4f}):")
    logger.debug(result["text"])
    chunk_idx = result["metadata"]["chunk_index"]
    logger.debug(f"From chunk {chunk_idx}")
    logger.debug("=====================================")

"""
## Generating Context for Response
Now we prepare the context by combining information from relevant chunks and questions.
"""
logger.info("## Generating Context for Response")

def prepare_context(search_results):
    """
    Prepares a unified context from search results for response generation.

    Args:
    search_results (List[Dict]): Results from semantic search.

    Returns:
    str: Combined context string.
    """
    chunk_indices = set()
    context_chunks = []

    for result in search_results:
        if result["metadata"]["type"] == "chunk":
            chunk_indices.add(result["metadata"]["index"])
            context_chunks.append(f"Chunk {result['metadata']['index']}:\n{result['text']}")

    for result in search_results:
        if result["metadata"]["type"] == "question":
            chunk_idx = result["metadata"]["chunk_index"]
            if chunk_idx not in chunk_indices:
                chunk_indices.add(chunk_idx)
                context_chunks.append(f"Chunk {chunk_idx} (referenced by question '{result['text']}'):\n{result['metadata']['original_chunk']}")

    full_context = "\n\n".join(context_chunks)
    return full_context

"""
## Generating a Response Based on Retrieved Chunks
"""
logger.info("## Generating a Response Based on Retrieved Chunks")

def generate_response(query, context, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Generates a response based on the query and context.

    Args:
    query (str): User's question.
    context (str): Context information retrieved from the vector store.
    model (str): Model to use for response generation.

    Returns:
    str: Generated response.
    """
    system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"

    user_prompt = f"""
        Context:
        {context}

        Question: {query}

        Please answer the question based only on the context provided above. Be concise and accurate.
    """

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content

"""
## Generating and Displaying the Response
"""
logger.info("## Generating and Displaying the Response")

context = prepare_context(search_results)

response_text = generate_response(query, context)

logger.debug("\nQuery:", query)
logger.debug("\nResponse:")
logger.debug(response_text)

"""
## Evaluating the AI Response
We compare the AI response with the expected answer and assign a score.
"""
logger.info("## Evaluating the AI Response")

def evaluate_response(query, response, reference_answer, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Evaluates the AI response against a reference answer.

    Args:
    query (str): The user's question.
    response (str): The AI-generated response.
    reference_answer (str): The reference/ideal answer.
    model (str): Model to use for evaluation.

    Returns:
    str: Evaluation feedback.
    """
    evaluate_system_prompt = """You are an intelligent evaluation system tasked with assessing AI responses.

        Compare the AI assistant's response to the true/reference answer, and evaluate based on:
        1. Factual correctness - Does the response contain accurate information?
        2. Completeness - Does it cover all important aspects from the reference?
        3. Relevance - Does it directly address the question?

        Assign a score from 0 to 1:
        - 1.0: Perfect match in content and meaning
        - 0.8: Very good, with minor omissions/differences
        - 0.6: Good, covers main points but misses some details
        - 0.4: Partial answer with significant omissions
        - 0.2: Minimal relevant information
        - 0.0: Incorrect or irrelevant

        Provide your score with justification.
    """

    evaluation_prompt = f"""
        User Query: {query}

        AI Response:
        {response}

        Reference Answer:
        {reference_answer}

        Please evaluate the AI response against the reference answer.
    """

    eval_response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": evaluate_system_prompt},
            {"role": "user", "content": evaluation_prompt}
        ]
    )

    return eval_response.choices[0].message.content

"""
## Running the Evaluation
"""
logger.info("## Running the Evaluation")

reference_answer = data[0]['ideal_answer']

evaluation = evaluate_response(query, response_text, reference_answer)

logger.debug("\nEvaluation:")
logger.debug(evaluation)

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