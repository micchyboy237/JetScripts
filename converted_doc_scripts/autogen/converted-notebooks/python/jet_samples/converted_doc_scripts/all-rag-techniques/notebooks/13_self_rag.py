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
# Self-RAG: A Dynamic Approach to RAG

In this notebook, I implement Self-RAG, an advanced RAG system that dynamically decides when and how to use retrieved information. Unlike traditional RAG approaches, Self-RAG introduces reflection points throughout the retrieval and generation process, resulting in higher quality and more reliable responses.

## Key Components of Self-RAG

1. **Retrieval Decision**: Determines if retrieval is even necessary for a given query
2. **Document Retrieval**: Fetches potentially relevant documents when needed  
3. **Relevance Evaluation**: Assesses how relevant each retrieved document is
4. **Response Generation**: Creates responses based on relevant contexts
5. **Support Assessment**: Evaluates if responses are properly grounded in the context
6. **Utility Evaluation**: Rates the overall usefulness of generated responses

## Setting Up the Environment
We begin by importing necessary libraries.
"""
logger.info("# Self-RAG: A Dynamic Approach to RAG")


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
## Simple Vector Store Implementation
We'll create a basic vector store to manage document chunks and their embeddings.
"""
logger.info("## Simple Vector Store Implementation")

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
        self.metadata.append(metadata or {})  # Add metadata to metadata list, default to empty dict if None

    def similarity_search(self, query_embedding, k=5, filter_func=None):
        """
        Find the most similar items to a query embedding.

        Args:
        query_embedding (List[float]): Query embedding vector.
        k (int): Number of results to return.
        filter_func (callable, optional): Function to filter results.

        Returns:
        List[Dict]: Top k most similar items with their texts and metadata.
        """
        if not self.vectors:
            return []  # Return empty list if no vectors are stored

        query_vector = np.array(query_embedding)

        similarities = []
        for i, vector in enumerate(self.vectors):
            if filter_func and not filter_func(self.metadata[i]):
                continue

            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))  # Append index and similarity score

        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],  # Add the text
                "metadata": self.metadata[idx],  # Add the metadata
                "similarity": score  # Add the similarity score
            })

        return results  # Return the list of top k results

"""
## Creating Embeddings
"""
logger.info("## Creating Embeddings")

def create_embeddings(text, model="BAAI/bge-en-icl"):
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
## Document Processing Pipeline
"""
logger.info("## Document Processing Pipeline")

def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    Process a document for Self-RAG.

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
## Self-RAG Components
### 1. Retrieval Decision
"""
logger.info("## Self-RAG Components")

def determine_if_retrieval_needed(query):
    """
    Determines if retrieval is necessary for the given query.

    Args:
        query (str): User query

    Returns:
        bool: True if retrieval is needed, False otherwise
    """
    system_prompt = """You are an AI assistant that determines if retrieval is necessary to answer a query.
    For factual questions, specific information requests, or questions about events, people, or concepts, answer "Yes".
    For opinions, hypothetical scenarios, or simple queries with common knowledge, answer "No".
    Answer with ONLY "Yes" or "No"."""

    user_prompt = f"Query: {query}\n\nIs retrieval necessary to answer this query accurately?"

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    answer = response.choices[0].message.content.strip().lower()

    return "yes" in answer

"""
### 2. Relevance Evaluation
"""
logger.info("### 2. Relevance Evaluation")

def evaluate_relevance(query, context):
    """
    Evaluates the relevance of a context to the query.

    Args:
        query (str): User query
        context (str): Context text

    Returns:
        str: 'relevant' or 'irrelevant'
    """
    system_prompt = """You are an AI assistant that determines if a document is relevant to a query.
    Consider whether the document contains information that would be helpful in answering the query.
    Answer with ONLY "Relevant" or "Irrelevant"."""

    max_context_length = 2000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "... [truncated]"

    user_prompt = f"""Query: {query}
    Document content:
    {context}

    Is this document relevant to the query? Answer with ONLY "Relevant" or "Irrelevant".
    """

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    answer = response.choices[0].message.content.strip().lower()

    return answer  # Return the relevance evaluation

"""
### 3. Support Assessment
"""
logger.info("### 3. Support Assessment")

def assess_support(response, context):
    """
    Assesses how well a response is supported by the context.

    Args:
        response (str): Generated response
        context (str): Context text

    Returns:
        str: 'fully supported', 'partially supported', or 'no support'
    """
    system_prompt = """You are an AI assistant that determines if a response is supported by the given context.
    Evaluate if the facts, claims, and information in the response are backed by the context.
    Answer with ONLY one of these three options:
    - "Fully supported": All information in the response is directly supported by the context.
    - "Partially supported": Some information in the response is supported by the context, but some is not.
    - "No support": The response contains significant information not found in or contradicting the context.
    """

    max_context_length = 2000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "... [truncated]"

    user_prompt = f"""Context:
    {context}

    Response:
    {response}

    How well is this response supported by the context? Answer with ONLY "Fully supported", "Partially supported", or "No support".
    """

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    answer = response.choices[0].message.content.strip().lower()

    return answer  # Return the support assessment

"""
### 4. Utility Evaluation
"""
logger.info("### 4. Utility Evaluation")

def rate_utility(query, response):
    """
    Rates the utility of a response for the query.

    Args:
        query (str): User query
        response (str): Generated response

    Returns:
        int: Utility rating from 1 to 5
    """
    system_prompt = """You are an AI assistant that rates the utility of a response to a query.
    Consider how well the response answers the query, its completeness, correctness, and helpfulness.
    Rate the utility on a scale from 1 to 5, where:
    - 1: Not useful at all
    - 2: Slightly useful
    - 3: Moderately useful
    - 4: Very useful
    - 5: Exceptionally useful
    Answer with ONLY a single number from 1 to 5."""

    user_prompt = f"""Query: {query}
    Response:
    {response}

    Rate the utility of this response on a scale from 1 to 5:"""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    rating = response.choices[0].message.content.strip()

    rating_match = re.search(r'[1-5]', rating)
    if rating_match:
        return int(rating_match.group())  # Return the extracted rating as an integer

    return 3  # Default to middle rating if parsing fails

"""
## Response Generation
"""
logger.info("## Response Generation")

def generate_response(query, context=None):
    """
    Generates a response based on the query and optional context.

    Args:
        query (str): User query
        context (str, optional): Context text

    Returns:
        str: Generated response
    """
    system_prompt = """You are a helpful AI assistant. Provide a clear, accurate, and informative response to the query."""

    if context:
        user_prompt = f"""Context:
        {context}

        Query: {query}

        Please answer the query based on the provided context.
        """
    else:
        user_prompt = f"""Query: {query}

        Please answer the query to the best of your ability."""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

"""
## Complete Self-RAG Implementation
"""
logger.info("## Complete Self-RAG Implementation")

def self_rag(query, vector_store, top_k=3):
    """
    Implements the complete Self-RAG pipeline.

    Args:
        query (str): User query
        vector_store (SimpleVectorStore): Vector store containing document chunks
        top_k (int): Number of documents to retrieve initially

    Returns:
        dict: Results including query, response, and metrics from the Self-RAG process
    """
    logger.debug(f"\n=== Starting Self-RAG for query: {query} ===\n")

    logger.debug("Step 1: Determining if retrieval is necessary...")
    retrieval_needed = determine_if_retrieval_needed(query)
    logger.debug(f"Retrieval needed: {retrieval_needed}")

    metrics = {
        "retrieval_needed": retrieval_needed,
        "documents_retrieved": 0,
        "relevant_documents": 0,
        "response_support_ratings": [],
        "utility_ratings": []
    }

    best_response = None
    best_score = -1

    if retrieval_needed:
        logger.debug("\nStep 2: Retrieving relevant documents...")
        query_embedding = create_embeddings(query)
        results = vector_store.similarity_search(query_embedding, k=top_k)
        metrics["documents_retrieved"] = len(results)
        logger.debug(f"Retrieved {len(results)} documents")

        logger.debug("\nStep 3: Evaluating document relevance...")
        relevant_contexts = []

        for i, result in enumerate(results):
            context = result["text"]
            relevance = evaluate_relevance(query, context)
            logger.debug(f"Document {i+1} relevance: {relevance}")

            if relevance == "relevant":
                relevant_contexts.append(context)

        metrics["relevant_documents"] = len(relevant_contexts)
        logger.debug(f"Found {len(relevant_contexts)} relevant documents")

        if relevant_contexts:
            logger.debug("\nStep 4: Processing relevant contexts...")
            for i, context in enumerate(relevant_contexts):
                logger.debug(f"\nProcessing context {i+1}/{len(relevant_contexts)}...")

                logger.debug("Generating response...")
                response = generate_response(query, context)

                logger.debug("Assessing support...")
                support_rating = assess_support(response, context)
                logger.debug(f"Support rating: {support_rating}")
                metrics["response_support_ratings"].append(support_rating)

                logger.debug("Rating utility...")
                utility_rating = rate_utility(query, response)
                logger.debug(f"Utility rating: {utility_rating}/5")
                metrics["utility_ratings"].append(utility_rating)

                support_score = {
                    "fully supported": 3,
                    "partially supported": 1,
                    "no support": 0
                }.get(support_rating, 0)

                overall_score = support_score * 5 + utility_rating
                logger.debug(f"Overall score: {overall_score}")

                if overall_score > best_score:
                    best_response = response
                    best_score = overall_score
                    logger.debug("New best response found!")

        if not relevant_contexts or best_score <= 0:
            logger.debug("\nNo suitable context found or poor responses, generating without retrieval...")
            best_response = generate_response(query)
    else:
        logger.debug("\nNo retrieval needed, generating response directly...")
        best_response = generate_response(query)

    metrics["best_score"] = best_score
    metrics["used_retrieval"] = retrieval_needed and best_score > 0

    logger.debug("\n=== Self-RAG Completed ===")

    return {
        "query": query,
        "response": best_response,
        "metrics": metrics
    }

"""
## Running the Complete Self-RAG System
"""
logger.info("## Running the Complete Self-RAG System")

def run_self_rag_example():
    """
    Demonstrates the complete Self-RAG system with examples.
    """
    pdf_path = f"{GENERATED_DIR}/AI_Information.pdf"  # Path to the PDF document
    logger.debug(f"Processing document: {pdf_path}")
    vector_store = process_document(pdf_path)  # Process the document and create a vector store

    query1 = "What are the main ethical concerns in AI development?"
    logger.debug("\n" + "="*80)
    logger.debug(f"EXAMPLE 1: {query1}")
    result1 = self_rag(query1, vector_store)  # Run Self-RAG for the first query
    logger.debug("\nFinal response:")
    logger.debug(result1["response"])  # Print the final response for the first query
    logger.debug("\nMetrics:")
    logger.debug(json.dumps(result1["metrics"], indent=2))  # Print the metrics for the first query

    query2 = "Can you write a short poem about artificial intelligence?"
    logger.debug("\n" + "="*80)
    logger.debug(f"EXAMPLE 2: {query2}")
    result2 = self_rag(query2, vector_store)  # Run Self-RAG for the second query
    logger.debug("\nFinal response:")
    logger.debug(result2["response"])  # Print the final response for the second query
    logger.debug("\nMetrics:")
    logger.debug(json.dumps(result2["metrics"], indent=2))  # Print the metrics for the second query

    query3 = "How might AI impact economic growth in developing countries?"
    logger.debug("\n" + "="*80)
    logger.debug(f"EXAMPLE 3: {query3}")
    result3 = self_rag(query3, vector_store)  # Run Self-RAG for the third query
    logger.debug("\nFinal response:")
    logger.debug(result3["response"])  # Print the final response for the third query
    logger.debug("\nMetrics:")
    logger.debug(json.dumps(result3["metrics"], indent=2))  # Print the metrics for the third query

    return {
        "example1": result1,
        "example2": result2,
        "example3": result3
    }

"""
## Evaluating Self-RAG Against Traditional RAG
"""
logger.info("## Evaluating Self-RAG Against Traditional RAG")

def traditional_rag(query, vector_store, top_k=3):
    """
    Implements a traditional RAG approach for comparison.

    Args:
        query (str): User query
        vector_store (SimpleVectorStore): Vector store containing document chunks
        top_k (int): Number of documents to retrieve

    Returns:
        str: Generated response
    """
    logger.debug(f"\n=== Running traditional RAG for query: {query} ===\n")

    logger.debug("Retrieving documents...")
    query_embedding = create_embeddings(query)  # Create embeddings for the query
    results = vector_store.similarity_search(query_embedding, k=top_k)  # Search for similar documents
    logger.debug(f"Retrieved {len(results)} documents")

    contexts = [result["text"] for result in results]  # Extract text from results
    combined_context = "\n\n".join(contexts)  # Combine texts into a single context

    logger.debug("Generating response...")
    response = generate_response(query, combined_context)  # Generate response based on the combined context

    return response

def evaluate_rag_approaches(pdf_path, test_queries, reference_answers=None):
    """
    Compare Self-RAG with traditional RAG.

    Args:
        pdf_path (str): Path to the document
        test_queries (List[str]): List of test queries
        reference_answers (List[str], optional): Reference answers for evaluation

    Returns:
        dict: Evaluation results
    """
    logger.debug("=== Evaluating RAG Approaches ===")

    vector_store = process_document(pdf_path)

    results = []

    for i, query in enumerate(test_queries):
        logger.debug(f"\nProcessing query {i+1}: {query}")

        self_rag_result = self_rag(query, vector_store)  # Get response from Self-RAG
        self_rag_response = self_rag_result["response"]

        trad_rag_response = traditional_rag(query, vector_store)  # Get response from traditional RAG

        reference = reference_answers[i] if reference_answers and i < len(reference_answers) else None
        comparison = compare_responses(query, self_rag_response, trad_rag_response, reference)  # Compare responses

        results.append({
            "query": query,
            "self_rag_response": self_rag_response,
            "traditional_rag_response": trad_rag_response,
            "reference_answer": reference,
            "comparison": comparison,
            "self_rag_metrics": self_rag_result["metrics"]
        })

    overall_analysis = generate_overall_analysis(results)

    return {
        "results": results,
        "overall_analysis": overall_analysis
    }

def compare_responses(query, self_rag_response, trad_rag_response, reference=None):
    """
    Compare responses from Self-RAG and traditional RAG.

    Args:
        query (str): User query
        self_rag_response (str): Response from Self-RAG
        trad_rag_response (str): Response from traditional RAG
        reference (str, optional): Reference answer

    Returns:
        str: Comparison analysis
    """
    system_prompt = """You are an expert evaluator of RAG systems. Your task is to compare responses from two different RAG approaches:
1. Self-RAG: A dynamic approach that decides if retrieval is needed and evaluates information relevance and response quality
2. Traditional RAG: Always retrieves documents and uses them to generate a response

Compare the responses based on:
- Relevance to the query
- Factual correctness
- Completeness and informativeness
- Conciseness and focus"""

    user_prompt = f"""Query: {query}

Response from Self-RAG:
{self_rag_response}

Response from Traditional RAG:
{trad_rag_response}
"""

    if reference:
        user_prompt += f"""
Reference Answer (for factual checking):
{reference}
"""

    user_prompt += """
Compare these responses and explain which one is better and why.
Focus on accuracy, relevance, completeness, and quality.
"""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # Using a stronger model for evaluation
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content

def generate_overall_analysis(results):
    """
    Generate an overall analysis of Self-RAG vs traditional RAG.

    Args:
        results (List[Dict]): Results from evaluate_rag_approaches

    Returns:
        str: Overall analysis
    """
    system_prompt = """You are an expert evaluator of RAG systems. Your task is to provide an overall analysis comparing
    Self-RAG and Traditional RAG based on multiple test queries.

    Focus your analysis on:
    1. When Self-RAG performs better and why
    2. When Traditional RAG performs better and why
    3. The impact of dynamic retrieval decisions in Self-RAG
    4. The value of relevance and support evaluation in Self-RAG
    5. Overall recommendations on which approach to use for different types of queries"""

    comparisons_summary = ""
    for i, result in enumerate(results):
        comparisons_summary += f"Query {i+1}: {result['query']}\n"
        comparisons_summary += f"Self-RAG metrics: Retrieval needed: {result['self_rag_metrics']['retrieval_needed']}, "
        comparisons_summary += f"Relevant docs: {result['self_rag_metrics']['relevant_documents']}/{result['self_rag_metrics']['documents_retrieved']}\n"
        comparisons_summary += f"Comparison summary: {result['comparison'][:200]}...\n\n"

        user_prompt = f"""Based on the following comparison results from {len(results)} test queries, please provide an overall analysis of
    Self-RAG versus Traditional RAG:

    {comparisons_summary}

    Please provide your comprehensive analysis.
    """

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content

"""
## Evaluating the Self-RAG System

The final step is to evaluate the Self-RAG system against traditional RAG approaches. We'll compare the quality of responses generated by both systems and analyze the performance of Self-RAG in different scenarios.
"""
logger.info("## Evaluating the Self-RAG System")

pdf_path = f"{GENERATED_DIR}/AI_Information.pdf"

test_queries = [
    "What are the main ethical concerns in AI development?",        # Document-focused query
]

reference_answers = [
    "The main ethical concerns in AI development include bias and fairness, privacy, transparency, accountability, safety, and the potential for misuse or harmful applications.",
]

evaluation_results = evaluate_rag_approaches(
    pdf_path=pdf_path,                  # Source document containing AI information
    test_queries=test_queries,          # List of AI-related test queries
    reference_answers=reference_answers  # Ground truth answers for evaluation
)

logger.debug("\n=== OVERALL ANALYSIS ===\n")
logger.debug(evaluation_results["overall_analysis"])

logger.info("\n\n[DONE]", bright=True)