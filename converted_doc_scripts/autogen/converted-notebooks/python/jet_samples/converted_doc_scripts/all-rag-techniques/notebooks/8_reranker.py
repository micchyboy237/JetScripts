from jet.logger import CustomLogger
from openai import MLX
import fitz
import json
import numpy as np
import os
import re
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
# Reranking for Enhanced RAG Systems

This notebook implements reranking techniques to improve retrieval quality in RAG systems. Reranking acts as a second filtering step after initial retrieval to ensure the most relevant content is used for response generation.

## Key Concepts of Reranking

1. **Initial Retrieval**: First pass using basic similarity search (less accurate but faster)
2. **Document Scoring**: Evaluating each retrieved document's relevance to the query
3. **Reordering**: Sorting documents by their relevance scores
4. **Selection**: Using only the most relevant documents for response generation

## Setting Up the Environment
We begin by importing necessary libraries.
"""
logger.info("# Reranking for Enhanced RAG Systems")


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
## Setting Up the MLX API Client
We initialize the MLX client to generate embeddings and responses.
"""
logger.info("## Setting Up the MLX API Client")

client = MLXAutogenChatLLMAdapter(
    base_url="https://api.studio.nebius.com/v1/",
#     api_key=os.getenv("OPENAI_API_KEY")  # Retrieve the API key from environment variables
)

"""
## Building a Simple Vector Store
To demonstrate how reranking integrate with retrieval, let's implement a simple vector store.
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
        self.vectors = []  # List to store embedding vectors
        self.texts = []  # List to store original texts
        self.metadata = []  # List to store metadata for each text

    def add_item(self, text, embedding, metadata=None):
        """
        Add an item to the vector store.

        Args:
        text (str): The original text.
        embedding (List[float]): The embedding vector.
        metadata (dict, optional): Additional metadata.
        """
        self.vectors.append(np.array(embedding))  # Convert embedding to numpy array and add to vectors list
        self.texts.append(text)  # Add the original text to texts list
        self.metadata.append(metadata or {})  # Add metadata to metadata list, use empty dict if None

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
            return []  # Return empty list if no vectors are stored

        query_vector = np.array(query_embedding)

        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))  # Append index and similarity score

        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],  # Add the corresponding text
                "metadata": self.metadata[idx],  # Add the corresponding metadata
                "similarity": score  # Add the similarity score
            })

        return results  # Return the list of top k similar items

"""
## Creating Embeddings
"""
logger.info("## Creating Embeddings")

def create_embeddings(text, model="BAAI/bge-en-icl"):
    """
    Creates embeddings for the given text using the specified MLX model.

    Args:
    text (str): The input text for which embeddings are to be created.
    model (str): The model to be used for creating embeddings.

    Returns:
    List[float]: The embedding vector.
    """
    input_text = text if isinstance(text, list) else [text]

    response = client.embeddings.create(
        model=model,
        input=input_text
    )

    if isinstance(text, str):
        return response.data[0].embedding

    return [item.embedding for item in response.data]

"""
## Document Processing Pipeline
Now that we have defined the necessary functions and classes, we can proceed to define the document processing pipeline.
"""
logger.info("## Document Processing Pipeline")

def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    Process a document for RAG.

    Args:
    pdf_path (str): Path to the PDF file.
    chunk_size (int): Size of each chunk in characters.
    chunk_overlap (int): Overlap between chunks in characters.

    Returns:
    SimpleVectorStore: A vector store containing document chunks and their embeddings.
    """
    logger.debug("Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_path)

    logger.debug("Chunking text...")
    chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)
    logger.debug(f"Created {len(chunks)} text chunks")

    logger.debug("Creating embeddings for chunks...")
    chunk_embeddings = create_embeddings(chunks)

    store = SimpleVectorStore()

    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={"index": i, "source": pdf_path}
        )

    logger.debug(f"Added {len(chunks)} chunks to the vector store")
    return store

"""
## Implementing LLM-based Reranking
Let's implement the LLM-based reranking function using the MLX API.
"""
logger.info("## Implementing LLM-based Reranking")

def rerank_with_llm(query, results, top_n=3, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Reranks search results using LLM relevance scoring.

    Args:
        query (str): User query
        results (List[Dict]): Initial search results
        top_n (int): Number of results to return after reranking
        model (str): Model to use for scoring

    Returns:
        List[Dict]: Reranked results
    """
    logger.debug(f"Reranking {len(results)} documents...")  # Print the number of documents to be reranked

    scored_results = []  # Initialize an empty list to store scored results

    system_prompt = """You are an expert at evaluating document relevance for search queries.
Your task is to rate documents on a scale from 0 to 10 based on how well they answer the given query.

Guidelines:
- Score 0-2: Document is completely irrelevant
- Score 3-5: Document has some relevant information but doesn't directly answer the query
- Score 6-8: Document is relevant and partially answers the query
- Score 9-10: Document is highly relevant and directly answers the query

You MUST respond with ONLY a single integer score between 0 and 10. Do not include ANY other text."""

    for i, result in enumerate(results):
        if i % 5 == 0:
            logger.debug(f"Scoring document {i+1}/{len(results)}...")

        user_prompt = f"""Query: {query}

Document:
{result['text']}

Rate this document's relevance to the query on a scale from 0 to 10:"""

        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        score_text = response.choices[0].message.content.strip()

        score_match = re.search(r'\b(10|[0-9])\b', score_text)
        if score_match:
            score = float(score_match.group(1))
        else:
            logger.debug(f"Warning: Could not extract score from response: '{score_text}', using similarity score instead")
            score = result["similarity"] * 10

        scored_results.append({
            "text": result["text"],
            "metadata": result["metadata"],
            "similarity": result["similarity"],
            "relevance_score": score
        })

    reranked_results = sorted(scored_results, key=lambda x: x["relevance_score"], reverse=True)

    return reranked_results[:top_n]

"""
## Simple Keyword-based Reranking
"""
logger.info("## Simple Keyword-based Reranking")

def rerank_with_keywords(query, results, top_n=3):
    """
    A simple alternative reranking method based on keyword matching and position.

    Args:
        query (str): User query
        results (List[Dict]): Initial search results
        top_n (int): Number of results to return after reranking

    Returns:
        List[Dict]: Reranked results
    """
    keywords = [word.lower() for word in query.split() if len(word) > 3]

    scored_results = []  # Initialize a list to store scored results

    for result in results:
        document_text = result["text"].lower()  # Convert document text to lowercase

        base_score = result["similarity"] * 0.5

        keyword_score = 0
        for keyword in keywords:
            if keyword in document_text:
                keyword_score += 0.1

                first_position = document_text.find(keyword)
                if first_position < len(document_text) / 4:  # In the first quarter of the text
                    keyword_score += 0.1

                frequency = document_text.count(keyword)
                keyword_score += min(0.05 * frequency, 0.2)  # Cap at 0.2

        final_score = base_score + keyword_score

        scored_results.append({
            "text": result["text"],
            "metadata": result["metadata"],
            "similarity": result["similarity"],
            "relevance_score": final_score
        })

    reranked_results = sorted(scored_results, key=lambda x: x["relevance_score"], reverse=True)

    return reranked_results[:top_n]

"""
## Response Generation
"""
logger.info("## Response Generation")

def generate_response(query, context, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Generates a response based on the query and context.

    Args:
        query (str): User query
        context (str): Retrieved context
        model (str): Model to use for response generation

    Returns:
        str: Generated response
    """
    system_prompt = "You are a helpful AI assistant. Answer the user's question based only on the provided context. If you cannot find the answer in the context, state that you don't have enough information."

    user_prompt = f"""
        Context:
        {context}

        Question: {query}

        Please provide a comprehensive answer based only on the context above.
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
## Full RAG Pipeline with Reranking
So far, we have implemented the core components of the RAG pipeline, including document processing, question answering, and reranking. Now, we will combine these components to create a full RAG pipeline.
"""
logger.info("## Full RAG Pipeline with Reranking")

def rag_with_reranking(query, vector_store, reranking_method="llm", top_n=3, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Complete RAG pipeline incorporating reranking.

    Args:
        query (str): User query
        vector_store (SimpleVectorStore): Vector store
        reranking_method (str): Method for reranking ('llm' or 'keywords')
        top_n (int): Number of results to return after reranking
        model (str): Model for response generation

    Returns:
        Dict: Results including query, context, and response
    """
    query_embedding = create_embeddings(query)

    initial_results = vector_store.similarity_search(query_embedding, k=10)

    if reranking_method == "llm":
        reranked_results = rerank_with_llm(query, initial_results, top_n=top_n)
    elif reranking_method == "keywords":
        reranked_results = rerank_with_keywords(query, initial_results, top_n=top_n)
    else:
        reranked_results = initial_results[:top_n]

    context = "\n\n===\n\n".join([result["text"] for result in reranked_results])

    response = generate_response(query, context, model)

    return {
        "query": query,
        "reranking_method": reranking_method,
        "initial_results": initial_results[:top_n],
        "reranked_results": reranked_results,
        "context": context,
        "response": response
    }

"""
## Evaluating Reranking Quality
"""
logger.info("## Evaluating Reranking Quality")

with open('data/val.json') as f:
    data = json.load(f)

query = data[0]['question']

reference_answer = data[0]['ideal_answer']

pdf_path = f"{GENERATED_DIR}/AI_Information.pdf"

vector_store = process_document(pdf_path)

query = "Does AI have the potential to transform the way we live and work?"

logger.debug("Comparing retrieval methods...")

logger.debug("\n=== STANDARD RETRIEVAL ===")
standard_results = rag_with_reranking(query, vector_store, reranking_method="none")
logger.debug(f"\nQuery: {query}")
logger.debug(f"\nResponse:\n{standard_results['response']}")

logger.debug("\n=== LLM-BASED RERANKING ===")
llm_results = rag_with_reranking(query, vector_store, reranking_method="llm")
logger.debug(f"\nQuery: {query}")
logger.debug(f"\nResponse:\n{llm_results['response']}")

logger.debug("\n=== KEYWORD-BASED RERANKING ===")
keyword_results = rag_with_reranking(query, vector_store, reranking_method="keywords")
logger.debug(f"\nQuery: {query}")
logger.debug(f"\nResponse:\n{keyword_results['response']}")

def evaluate_reranking(query, standard_results, reranked_results, reference_answer=None):
    """
    Evaluates the quality of reranked results compared to standard results.

    Args:
        query (str): User query
        standard_results (Dict): Results from standard retrieval
        reranked_results (Dict): Results from reranked retrieval
        reference_answer (str, optional): Reference answer for comparison

    Returns:
        str: Evaluation output
    """
    system_prompt = """You are an expert evaluator of RAG systems.
    Compare the retrieved contexts and responses from two different retrieval methods.
    Assess which one provides better context and a more accurate, comprehensive answer."""

    comparison_text = f"""Query: {query}

Standard Retrieval Context:
{standard_results['context'][:1000]}... [truncated]

Standard Retrieval Answer:
{standard_results['response']}

Reranked Retrieval Context:
{reranked_results['context'][:1000]}... [truncated]

Reranked Retrieval Answer:
{reranked_results['response']}"""

    if reference_answer:
        comparison_text += f"""

Reference Answer:
{reference_answer}"""

    user_prompt = f"""
{comparison_text}

Please evaluate which retrieval method provided:
1. More relevant context
2. More accurate answer
3. More comprehensive answer
4. Better overall performance

Provide a detailed analysis with specific examples.
"""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content

evaluation = evaluate_reranking(
    query=query,  # The user query
    standard_results=standard_results,  # Results from standard retrieval
    reranked_results=llm_results,  # Results from LLM-based reranking
    reference_answer=reference_answer  # Reference answer for comparison
)

logger.debug("\n=== EVALUATION RESULTS ===")
logger.debug(evaluation)

logger.info("\n\n[DONE]", bright=True)