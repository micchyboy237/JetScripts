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
# Contextual Compression for Enhanced RAG Systems
In this notebook, I implement a contextual compression technique to improve our RAG system's efficiency. We'll filter and compress retrieved text chunks to keep only the most relevant parts, reducing noise and improving response quality.

When retrieving documents for RAG, we often get chunks containing both relevant and irrelevant information. Contextual compression helps us:

- Remove irrelevant sentences and paragraphs
- Focus only on query-relevant information
- Maximize the useful signal in our context window

Let's implement this approach from scratch!

## Setting Up the Environment
We begin by importing necessary libraries.
"""
logger.info("# Contextual Compression for Enhanced RAG Systems")


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

def chunk_text(text, n=1000, overlap=200):
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
#     api_key= os.environ.get("OPENAI_API_KEY") # Use your Ollama API key
)

"""
## Building a Simple Vector Store
let's implement a simple vector store since we cannot use FAISS.
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
                "text": self.texts[idx],  # Add the text corresponding to the index
                "metadata": self.metadata[idx],  # Add the metadata corresponding to the index
                "similarity": score  # Add the similarity score
            })

        return results  # Return the list of top k results

"""
## Embedding Generation
"""
logger.info("## Embedding Generation")

def create_embeddings(text,  model="BAAI/bge-en-icl"):
    """
    Creates embeddings for the given text.

    Args:
    text (str or List[str]): The input text(s) for which embeddings are to be created.
    model (str): The model to be used for creating embeddings.

    Returns:
    List[float] or List[List[float]]: The embedding vector(s).
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
## Building Our Document Processing Pipeline
"""
logger.info("## Building Our Document Processing Pipeline")

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
## Implementing Contextual Compression
This is the core of our approach - we'll use an LLM to filter and compress retrieved content.
"""
logger.info("## Implementing Contextual Compression")

def compress_chunk(chunk, query, compression_type="selective", model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Compress a retrieved chunk by keeping only the parts relevant to the query.

    Args:
        chunk (str): Text chunk to compress
        query (str): User query
        compression_type (str): Type of compression ("selective", "summary", or "extraction")
        model (str): LLM model to use

    Returns:
        str: Compressed chunk
    """
    if compression_type == "selective":
        system_prompt = """You are an expert at information filtering.
        Your task is to analyze a document chunk and extract ONLY the sentences or paragraphs that are directly
        relevant to the user's query. Remove all irrelevant content.

        Your output should:
        1. ONLY include text that helps answer the query
        2. Preserve the exact wording of relevant sentences (do not paraphrase)
        3. Maintain the original order of the text
        4. Include ALL relevant content, even if it seems redundant
        5. EXCLUDE any text that isn't relevant to the query

        Format your response as plain text with no additional comments."""
    elif compression_type == "summary":
        system_prompt = """You are an expert at summarization.
        Your task is to create a concise summary of the provided chunk that focuses ONLY on
        information relevant to the user's query.

        Your output should:
        1. Be brief but comprehensive regarding query-relevant information
        2. Focus exclusively on information related to the query
        3. Omit irrelevant details
        4. Be written in a neutral, factual tone

        Format your response as plain text with no additional comments."""
    else:  # extraction
        system_prompt = """You are an expert at information extraction.
        Your task is to extract ONLY the exact sentences from the document chunk that contain information relevant
        to answering the user's query.

        Your output should:
        1. Include ONLY direct quotes of relevant sentences from the original text
        2. Preserve the original wording (do not modify the text)
        3. Include ONLY sentences that directly relate to the query
        4. Separate extracted sentences with newlines
        5. Do not add any commentary or additional text

        Format your response as plain text with no additional comments."""

    user_prompt = f"""
        Query: {query}

        Document Chunk:
        {chunk}

        Extract only the content relevant to answering this query.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    compressed_chunk = response.choices[0].message.content.strip()

    original_length = len(chunk)
    compressed_length = len(compressed_chunk)
    compression_ratio = (original_length - compressed_length) / original_length * 100

    return compressed_chunk, compression_ratio

"""
## Implementing Batch Compression
For efficiency, we'll compress multiple chunks in one go when possible.
"""
logger.info("## Implementing Batch Compression")

def batch_compress_chunks(chunks, query, compression_type="selective", model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Compress multiple chunks individually.

    Args:
        chunks (List[str]): List of text chunks to compress
        query (str): User query
        compression_type (str): Type of compression ("selective", "summary", or "extraction")
        model (str): LLM model to use

    Returns:
        List[Tuple[str, float]]: List of compressed chunks with compression ratios
    """
    logger.debug(f"Compressing {len(chunks)} chunks...")  # Print the number of chunks to be compressed
    results = []  # Initialize an empty list to store the results
    total_original_length = 0  # Initialize a variable to store the total original length of chunks
    total_compressed_length = 0  # Initialize a variable to store the total compressed length of chunks

    for i, chunk in enumerate(chunks):
        logger.debug(f"Compressing chunk {i+1}/{len(chunks)}...")  # Print the progress of compression
        compressed_chunk, compression_ratio = compress_chunk(chunk, query, compression_type, model)
        results.append((compressed_chunk, compression_ratio))  # Append the result to the results list

        total_original_length += len(chunk)  # Add the length of the original chunk to the total original length
        total_compressed_length += len(compressed_chunk)  # Add the length of the compressed chunk to the total compressed length

    overall_ratio = (total_original_length - total_compressed_length) / total_original_length * 100
    logger.debug(f"Overall compression ratio: {overall_ratio:.2f}%")  # Print the overall compression ratio

    return results  # Return the list of compressed chunks with compression ratios

"""
## Response Generation Function
"""
logger.info("## Response Generation Function")

def generate_response(query, context, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Generate a response based on the query and context.

    Args:
        query (str): User query
        context (str): Context text from compressed chunks
        model (str): LLM model to use

    Returns:
        str: Generated response
    """
    system_prompt = """You are a helpful AI assistant. Answer the user's question based only on the provided context.
    If you cannot find the answer in the context, state that you don't have enough information."""

    user_prompt = f"""
        Context:
        {context}

        Question: {query}

        Please provide a comprehensive answer based only on the context above.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content

"""
## The Complete RAG Pipeline with Contextual Compression


"""
logger.info("## The Complete RAG Pipeline with Contextual Compression")

def rag_with_compression(pdf_path, query, k=10, compression_type="selective", model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Complete RAG pipeline with contextual compression.

    Args:
        pdf_path (str): Path to PDF document
        query (str): User query
        k (int): Number of chunks to retrieve initially
        compression_type (str): Type of compression
        model (str): LLM model to use

    Returns:
        dict: Results including query, compressed chunks, and response
    """
    logger.debug("\n=== RAG WITH CONTEXTUAL COMPRESSION ===")
    logger.debug(f"Query: {query}")
    logger.debug(f"Compression type: {compression_type}")

    vector_store = process_document(pdf_path)

    query_embedding = create_embeddings(query)

    logger.debug(f"Retrieving top {k} chunks...")
    results = vector_store.similarity_search(query_embedding, k=k)
    retrieved_chunks = [result["text"] for result in results]

    compressed_results = batch_compress_chunks(retrieved_chunks, query, compression_type, model)
    compressed_chunks = [result[0] for result in compressed_results]
    compression_ratios = [result[1] for result in compressed_results]

    filtered_chunks = [(chunk, ratio) for chunk, ratio in zip(compressed_chunks, compression_ratios) if chunk.strip()]

    if not filtered_chunks:
        logger.debug("Warning: All chunks were compressed to empty strings. Using original chunks.")
        filtered_chunks = [(chunk, 0.0) for chunk in retrieved_chunks]
    else:
        compressed_chunks, compression_ratios = zip(*filtered_chunks)

    context = "\n\n---\n\n".join(compressed_chunks)

    logger.debug("Generating response based on compressed chunks...")
    response = generate_response(query, context, model)

    result = {
        "query": query,
        "original_chunks": retrieved_chunks,
        "compressed_chunks": compressed_chunks,
        "compression_ratios": compression_ratios,
        "context_length_reduction": f"{sum(compression_ratios)/len(compression_ratios):.2f}%",
        "response": response
    }

    logger.debug("\n=== RESPONSE ===")
    logger.debug(response)

    return result

"""
## Comparing RAG With and Without Compression
Let's create a function to compare standard RAG with our compression-enhanced version:
"""
logger.info("## Comparing RAG With and Without Compression")

def standard_rag(pdf_path, query, k=10, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Standard RAG without compression.

    Args:
        pdf_path (str): Path to PDF document
        query (str): User query
        k (int): Number of chunks to retrieve
        model (str): LLM model to use

    Returns:
        dict: Results including query, chunks, and response
    """
    logger.debug("\n=== STANDARD RAG ===")
    logger.debug(f"Query: {query}")

    vector_store = process_document(pdf_path)

    query_embedding = create_embeddings(query)

    logger.debug(f"Retrieving top {k} chunks...")
    results = vector_store.similarity_search(query_embedding, k=k)
    retrieved_chunks = [result["text"] for result in results]

    context = "\n\n---\n\n".join(retrieved_chunks)

    logger.debug("Generating response...")
    response = generate_response(query, context, model)

    result = {
        "query": query,
        "chunks": retrieved_chunks,
        "response": response
    }

    logger.debug("\n=== RESPONSE ===")
    logger.debug(response)

    return result

"""
## Evaluating Our Approach
Now, let's implement a function to evaluate and compare the responses:
"""
logger.info("## Evaluating Our Approach")

def evaluate_responses(query, responses, reference_answer):
    """
    Evaluate multiple responses against a reference answer.

    Args:
        query (str): User query
        responses (Dict[str, str]): Dictionary of responses by method
        reference_answer (str): Reference answer

    Returns:
        str: Evaluation text
    """
    system_prompt = """You are an objective evaluator of RAG responses. Compare different responses to the same query
    and determine which is most accurate, comprehensive, and relevant to the query."""

    user_prompt = f"""
    Query: {query}

    Reference Answer: {reference_answer}

    """

    for method, response in responses.items():
        user_prompt += f"\n{method.capitalize()} Response:\n{response}\n"

    user_prompt += """
    Please evaluate these responses based on:
    1. Factual accuracy compared to the reference
    2. Comprehensiveness - how completely they answer the query
    3. Conciseness - whether they avoid irrelevant information
    4. Overall quality

    Rank the responses from best to worst with detailed explanations.
    """

    evaluation_response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return evaluation_response.choices[0].message.content

"""

"""

def evaluate_compression(pdf_path, query, reference_answer=None, compression_types=["selective", "summary", "extraction"]):
    """
    Compare different compression techniques with standard RAG.

    Args:
        pdf_path (str): Path to PDF document
        query (str): User query
        reference_answer (str): Optional reference answer
        compression_types (List[str]): Compression types to evaluate

    Returns:
        dict: Evaluation results
    """
    logger.debug("\n=== EVALUATING CONTEXTUAL COMPRESSION ===")
    logger.debug(f"Query: {query}")

    standard_result = standard_rag(pdf_path, query)

    compression_results = {}

    for comp_type in compression_types:
        logger.debug(f"\nTesting {comp_type} compression...")
        compression_results[comp_type] = rag_with_compression(pdf_path, query, compression_type=comp_type)

    responses = {
        "standard": standard_result["response"]
    }
    for comp_type in compression_types:
        responses[comp_type] = compression_results[comp_type]["response"]

    if reference_answer:
        evaluation = evaluate_responses(query, responses, reference_answer)
        logger.debug("\n=== EVALUATION RESULTS ===")
        logger.debug(evaluation)
    else:
        evaluation = "No reference answer provided for evaluation."

    metrics = {}
    for comp_type in compression_types:
        metrics[comp_type] = {
            "avg_compression_ratio": f"{sum(compression_results[comp_type]['compression_ratios'])/len(compression_results[comp_type]['compression_ratios']):.2f}%",
            "total_context_length": len("\n\n".join(compression_results[comp_type]['compressed_chunks'])),
            "original_context_length": len("\n\n".join(standard_result['chunks']))
        }

    return {
        "query": query,
        "responses": responses,
        "evaluation": evaluation,
        "metrics": metrics,
        "standard_result": standard_result,
        "compression_results": compression_results
    }

"""
## Running Our Complete System (Custom Query)
"""
logger.info("## Running Our Complete System (Custom Query)")

pdf_path = f"{GENERATED_DIR}/AI_Information.pdf"

query = "What are the ethical concerns surrounding the use of AI in decision-making?"

reference_answer = """
The use of AI in decision-making raises several ethical concerns.
- Bias in AI models can lead to unfair or discriminatory outcomes, especially in critical areas like hiring, lending, and law enforcement.
- Lack of transparency and explainability in AI-driven decisions makes it difficult for individuals to challenge unfair outcomes.
- Privacy risks arise as AI systems process vast amounts of personal data, often without explicit consent.
- The potential for job displacement due to automation raises social and economic concerns.
- AI decision-making may also concentrate power in the hands of a few large tech companies, leading to accountability challenges.
- Ensuring fairness, accountability, and transparency in AI systems is essential for ethical deployment.
"""

results = evaluate_compression(
    pdf_path=pdf_path,
    query=query,
    reference_answer=reference_answer,
    compression_types=["selective", "summary", "extraction"]
)

"""
## Visualizing Compression Results
"""
logger.info("## Visualizing Compression Results")

def visualize_compression_results(evaluation_results):
    """
    Visualize the results of different compression techniques.

    Args:
        evaluation_results (Dict): Results from evaluate_compression function
    """
    query = evaluation_results["query"]
    standard_chunks = evaluation_results["standard_result"]["chunks"]

    logger.debug(f"Query: {query}")
    logger.debug("\n" + "="*80 + "\n")

    original_chunk = standard_chunks[0]

    for comp_type in evaluation_results["compression_results"].keys():
        compressed_chunks = evaluation_results["compression_results"][comp_type]["compressed_chunks"]
        compression_ratios = evaluation_results["compression_results"][comp_type]["compression_ratios"]

        compressed_chunk = compressed_chunks[0]
        compression_ratio = compression_ratios[0]

        logger.debug(f"\n=== {comp_type.upper()} COMPRESSION EXAMPLE ===\n")

        logger.debug("ORIGINAL CHUNK:")
        logger.debug("-" * 40)
        if len(original_chunk) > 800:
            logger.debug(original_chunk[:800] + "... [truncated]")
        else:
            logger.debug(original_chunk)
        logger.debug("-" * 40)
        logger.debug(f"Length: {len(original_chunk)} characters\n")

        logger.debug("COMPRESSED CHUNK:")
        logger.debug("-" * 40)
        logger.debug(compressed_chunk)
        logger.debug("-" * 40)
        logger.debug(f"Length: {len(compressed_chunk)} characters")
        logger.debug(f"Compression ratio: {compression_ratio:.2f}%\n")

        avg_ratio = sum(compression_ratios) / len(compression_ratios)
        logger.debug(f"Average compression across all chunks: {avg_ratio:.2f}%")
        logger.debug(f"Total context length reduction: {evaluation_results['metrics'][comp_type]['avg_compression_ratio']}")
        logger.debug("=" * 80)

    logger.debug("\n=== COMPRESSION SUMMARY ===\n")
    logger.debug(f"{'Technique':<15} {'Avg Ratio':<15} {'Context Length':<15} {'Original Length':<15}")
    logger.debug("-" * 60)

    for comp_type, metrics in evaluation_results["metrics"].items():
        logger.debug(f"{comp_type:<15} {metrics['avg_compression_ratio']:<15} {metrics['total_context_length']:<15} {metrics['original_context_length']:<15}")

"""

"""

visualize_compression_results(results)

logger.info("\n\n[DONE]", bright=True)