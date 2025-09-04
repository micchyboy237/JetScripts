from jet.logger import CustomLogger
from openai import OllamaFunctionCallingAdapter
import fitz
import json
import numpy as np
import os
import re
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# Adaptive Retrieval for Enhanced RAG Systems

In this notebook, I implement an Adaptive Retrieval system that dynamically selects the most appropriate retrieval strategy based on the type of query. This approach significantly enhances our RAG system's ability to provide accurate and relevant responses across a diverse range of questions.

Different questions demand different retrieval strategies. Our system:

1. Classifies the query type (Factual, Analytical, Opinion, or Contextual)
2. Selects the appropriate retrieval strategy
3. Executes specialized retrieval techniques
4. Generates a tailored response

## Setting Up the Environment
We begin by importing necessary libraries.
"""
logger.info("# Adaptive Retrieval for Enhanced RAG Systems")


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
## Setting Up the OllamaFunctionCallingAdapter API Client
We initialize the OllamaFunctionCallingAdapter client to generate embeddings and responses.
"""
logger.info("## Setting Up the OllamaFunctionCallingAdapter API Client")

client = OllamaFunctionCallingAdapter(
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
    Process a document for use with adaptive retrieval.

    Args:
    pdf_path (str): Path to the PDF file.
    chunk_size (int): Size of each chunk in characters.
    chunk_overlap (int): Overlap between chunks in characters.

    Returns:
    Tuple[List[str], SimpleVectorStore]: Document chunks and vector store.
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

    return chunks, store

"""
## Query Classification
"""
logger.info("## Query Classification")

def classify_query(query, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Classify a query into one of four categories: Factual, Analytical, Opinion, or Contextual.

    Args:
        query (str): User query
        model (str): LLM model to use

    Returns:
        str: Query category
    """
    system_prompt = """You are an expert at classifying questions.
        Classify the given query into exactly one of these categories:
        - Factual: Queries seeking specific, verifiable information.
        - Analytical: Queries requiring comprehensive analysis or explanation.
        - Opinion: Queries about subjective matters or seeking diverse viewpoints.
        - Contextual: Queries that depend on user-specific context.

        Return ONLY the category name, without any explanation or additional text.
    """

    user_prompt = f"Classify this query: {query}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    category = response.choices[0].message.content.strip()

    valid_categories = ["Factual", "Analytical", "Opinion", "Contextual"]

    for valid in valid_categories:
        if valid in category:
            return valid

    return "Factual"

"""
## Implementing Specialized Retrieval Strategies
### 1. Factual Strategy - Focus on Precision
"""
logger.info("## Implementing Specialized Retrieval Strategies")

def factual_retrieval_strategy(query, vector_store, k=4):
    """
    Retrieval strategy for factual queries focusing on precision.

    Args:
        query (str): User query
        vector_store (SimpleVectorStore): Vector store
        k (int): Number of documents to return

    Returns:
        List[Dict]: Retrieved documents
    """
    logger.debug(f"Executing Factual retrieval strategy for: '{query}'")

    system_prompt = """You are an expert at enhancing search queries.
        Your task is to reformulate the given factual query to make it more precise and
        specific for information retrieval. Focus on key entities and their relationships.

        Provide ONLY the enhanced query without any explanation.
    """

    user_prompt = f"Enhance this factual query: {query}"

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    enhanced_query = response.choices[0].message.content.strip()
    logger.debug(f"Enhanced query: {enhanced_query}")

    query_embedding = create_embeddings(enhanced_query)

    initial_results = vector_store.similarity_search(query_embedding, k=k*2)

    ranked_results = []

    for doc in initial_results:
        relevance_score = score_document_relevance(enhanced_query, doc["text"])
        ranked_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "similarity": doc["similarity"],
            "relevance_score": relevance_score
        })

    ranked_results.sort(key=lambda x: x["relevance_score"], reverse=True)

    return ranked_results[:k]

"""
### 2. Analytical Strategy - Comprehensive Coverage
"""
logger.info("### 2. Analytical Strategy - Comprehensive Coverage")

def analytical_retrieval_strategy(query, vector_store, k=4):
    """
    Retrieval strategy for analytical queries focusing on comprehensive coverage.

    Args:
        query (str): User query
        vector_store (SimpleVectorStore): Vector store
        k (int): Number of documents to return

    Returns:
        List[Dict]: Retrieved documents
    """
    logger.debug(f"Executing Analytical retrieval strategy for: '{query}'")

    system_prompt = """You are an expert at breaking down complex questions.
    Generate sub-questions that explore different aspects of the main analytical query.
    These sub-questions should cover the breadth of the topic and help retrieve
    comprehensive information.

    Return a list of exactly 3 sub-questions, one per line.
    """

    user_prompt = f"Generate sub-questions for this analytical query: {query}"

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )

    sub_queries = response.choices[0].message.content.strip().split('\n')
    sub_queries = [q.strip() for q in sub_queries if q.strip()]
    logger.debug(f"Generated sub-queries: {sub_queries}")

    all_results = []
    for sub_query in sub_queries:
        sub_query_embedding = create_embeddings(sub_query)
        results = vector_store.similarity_search(sub_query_embedding, k=2)
        all_results.extend(results)

    unique_texts = set()
    diverse_results = []

    for result in all_results:
        if result["text"] not in unique_texts:
            unique_texts.add(result["text"])
            diverse_results.append(result)

    if len(diverse_results) < k:
        main_query_embedding = create_embeddings(query)
        main_results = vector_store.similarity_search(main_query_embedding, k=k)

        for result in main_results:
            if result["text"] not in unique_texts and len(diverse_results) < k:
                unique_texts.add(result["text"])
                diverse_results.append(result)

    return diverse_results[:k]

"""
### 3. Opinion Strategy - Diverse Perspectives
"""
logger.info("### 3. Opinion Strategy - Diverse Perspectives")

def opinion_retrieval_strategy(query, vector_store, k=4):
    """
    Retrieval strategy for opinion queries focusing on diverse perspectives.

    Args:
        query (str): User query
        vector_store (SimpleVectorStore): Vector store
        k (int): Number of documents to return

    Returns:
        List[Dict]: Retrieved documents
    """
    logger.debug(f"Executing Opinion retrieval strategy for: '{query}'")

    system_prompt = """You are an expert at identifying different perspectives on a topic.
        For the given query about opinions or viewpoints, identify different perspectives
        that people might have on this topic.

        Return a list of exactly 3 different viewpoint angles, one per line.
    """

    user_prompt = f"Identify different perspectives on: {query}"

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )

    viewpoints = response.choices[0].message.content.strip().split('\n')
    viewpoints = [v.strip() for v in viewpoints if v.strip()]
    logger.debug(f"Identified viewpoints: {viewpoints}")

    all_results = []
    for viewpoint in viewpoints:
        combined_query = f"{query} {viewpoint}"
        viewpoint_embedding = create_embeddings(combined_query)
        results = vector_store.similarity_search(viewpoint_embedding, k=2)

        for result in results:
            result["viewpoint"] = viewpoint

        all_results.extend(results)

    selected_results = []
    for viewpoint in viewpoints:
        viewpoint_docs = [r for r in all_results if r.get("viewpoint") == viewpoint]
        if viewpoint_docs:
            selected_results.append(viewpoint_docs[0])

    remaining_slots = k - len(selected_results)
    if remaining_slots > 0:
        remaining_docs = [r for r in all_results if r not in selected_results]
        remaining_docs.sort(key=lambda x: x["similarity"], reverse=True)
        selected_results.extend(remaining_docs[:remaining_slots])

    return selected_results[:k]

"""
### 4. Contextual Strategy - User Context Integration
"""
logger.info("### 4. Contextual Strategy - User Context Integration")

def contextual_retrieval_strategy(query, vector_store, k=4, user_context=None):
    """
    Retrieval strategy for contextual queries integrating user context.

    Args:
        query (str): User query
        vector_store (SimpleVectorStore): Vector store
        k (int): Number of documents to return
        user_context (str): Additional user context

    Returns:
        List[Dict]: Retrieved documents
    """
    logger.debug(f"Executing Contextual retrieval strategy for: '{query}'")

    if not user_context:
        system_prompt = """You are an expert at understanding implied context in questions.
For the given query, infer what contextual information might be relevant or implied
but not explicitly stated. Focus on what background would help answering this query.

Return a brief description of the implied context."""

        user_prompt = f"Infer the implied context in this query: {query}"

        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )

        user_context = response.choices[0].message.content.strip()
        logger.debug(f"Inferred context: {user_context}")

    system_prompt = """You are an expert at reformulating questions with context.
    Given a query and some contextual information, create a more specific query that
    incorporates the context to get more relevant information.

    Return ONLY the reformulated query without explanation."""

    user_prompt = f"""
    Query: {query}
    Context: {user_context}

    Reformulate the query to incorporate this context:"""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    contextualized_query = response.choices[0].message.content.strip()
    logger.debug(f"Contextualized query: {contextualized_query}")

    query_embedding = create_embeddings(contextualized_query)
    initial_results = vector_store.similarity_search(query_embedding, k=k*2)

    ranked_results = []

    for doc in initial_results:
        context_relevance = score_document_context_relevance(query, user_context, doc["text"])
        ranked_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "similarity": doc["similarity"],
            "context_relevance": context_relevance
        })

    ranked_results.sort(key=lambda x: x["context_relevance"], reverse=True)
    return ranked_results[:k]

"""
## Helper Functions for Document Scoring
"""
logger.info("## Helper Functions for Document Scoring")

def score_document_relevance(query, document, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Score document relevance to a query using LLM.

    Args:
        query (str): User query
        document (str): Document text
        model (str): LLM model

    Returns:
        float: Relevance score from 0-10
    """
    system_prompt = """You are an expert at evaluating document relevance.
        Rate the relevance of a document to a query on a scale from 0 to 10, where:
        0 = Completely irrelevant
        10 = Perfectly addresses the query

        Return ONLY a numerical score between 0 and 10, nothing else.
    """

    doc_preview = document[:1500] + "..." if len(document) > 1500 else document

    user_prompt = f"""
        Query: {query}

        Document: {doc_preview}

        Relevance score (0-10):
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    score_text = response.choices[0].message.content.strip()

    match = re.search(r'(\d+(\.\d+)?)', score_text)
    if match:
        score = float(match.group(1))
        return min(10, max(0, score))  # Ensure score is within 0-10
    else:
        return 5.0

def score_document_context_relevance(query, context, document, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Score document relevance considering both query and context.

    Args:
        query (str): User query
        context (str): User context
        document (str): Document text
        model (str): LLM model

    Returns:
        float: Relevance score from 0-10
    """
    system_prompt = """You are an expert at evaluating document relevance considering context.
        Rate the document on a scale from 0 to 10 based on how well it addresses the query
        when considering the provided context, where:
        0 = Completely irrelevant
        10 = Perfectly addresses the query in the given context

        Return ONLY a numerical score between 0 and 10, nothing else.
    """

    doc_preview = document[:1500] + "..." if len(document) > 1500 else document

    user_prompt = f"""
    Query: {query}
    Context: {context}

    Document: {doc_preview}

    Relevance score considering context (0-10):
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    score_text = response.choices[0].message.content.strip()

    match = re.search(r'(\d+(\.\d+)?)', score_text)
    if match:
        score = float(match.group(1))
        return min(10, max(0, score))  # Ensure score is within 0-10
    else:
        return 5.0

"""
## The Core Adaptive Retriever
"""
logger.info("## The Core Adaptive Retriever")

def adaptive_retrieval(query, vector_store, k=4, user_context=None):
    """
    Perform adaptive retrieval by selecting and executing the appropriate strategy.

    Args:
        query (str): User query
        vector_store (SimpleVectorStore): Vector store
        k (int): Number of documents to retrieve
        user_context (str): Optional user context for contextual queries

    Returns:
        List[Dict]: Retrieved documents
    """
    query_type = classify_query(query)
    logger.debug(f"Query classified as: {query_type}")

    if query_type == "Factual":
        results = factual_retrieval_strategy(query, vector_store, k)
    elif query_type == "Analytical":
        results = analytical_retrieval_strategy(query, vector_store, k)
    elif query_type == "Opinion":
        results = opinion_retrieval_strategy(query, vector_store, k)
    elif query_type == "Contextual":
        results = contextual_retrieval_strategy(query, vector_store, k, user_context)
    else:
        results = factual_retrieval_strategy(query, vector_store, k)

    return results  # Return the retrieved documents

"""
## Response Generation
"""
logger.info("## Response Generation")

def generate_response(query, results, query_type, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Generate a response based on query, retrieved documents, and query type.

    Args:
        query (str): User query
        results (List[Dict]): Retrieved documents
        query_type (str): Type of query
        model (str): LLM model

    Returns:
        str: Generated response
    """
    context = "\n\n---\n\n".join([r["text"] for r in results])

    if query_type == "Factual":
        system_prompt = """You are a helpful assistant providing factual information.
    Answer the question based on the provided context. Focus on accuracy and precision.
    If the context doesn't contain the information needed, acknowledge the limitations."""

    elif query_type == "Analytical":
        system_prompt = """You are a helpful assistant providing analytical insights.
    Based on the provided context, offer a comprehensive analysis of the topic.
    Cover different aspects and perspectives in your explanation.
    If the context has gaps, acknowledge them while providing the best analysis possible."""

    elif query_type == "Opinion":
        system_prompt = """You are a helpful assistant discussing topics with multiple viewpoints.
    Based on the provided context, present different perspectives on the topic.
    Ensure fair representation of diverse opinions without showing bias.
    Acknowledge where the context presents limited viewpoints."""

    elif query_type == "Contextual":
        system_prompt = """You are a helpful assistant providing contextually relevant information.
    Answer the question considering both the query and its context.
    Make connections between the query context and the information in the provided documents.
    If the context doesn't fully address the specific situation, acknowledge the limitations."""

    else:
        system_prompt = """You are a helpful assistant. Answer the question based on the provided context. If you cannot answer from the context, acknowledge the limitations."""

    user_prompt = f"""
    Context:
    {context}

    Question: {query}

    Please provide a helpful response based on the context.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

"""
## Complete RAG Pipeline with Adaptive Retrieval
"""
logger.info("## Complete RAG Pipeline with Adaptive Retrieval")

def rag_with_adaptive_retrieval(pdf_path, query, k=4, user_context=None):
    """
    Complete RAG pipeline with adaptive retrieval.

    Args:
        pdf_path (str): Path to PDF document
        query (str): User query
        k (int): Number of documents to retrieve
        user_context (str): Optional user context

    Returns:
        Dict: Results including query, retrieved documents, query type, and response
    """
    logger.debug("\n=== RAG WITH ADAPTIVE RETRIEVAL ===")
    logger.debug(f"Query: {query}")

    chunks, vector_store = process_document(pdf_path)

    query_type = classify_query(query)
    logger.debug(f"Query classified as: {query_type}")

    retrieved_docs = adaptive_retrieval(query, vector_store, k, user_context)

    response = generate_response(query, retrieved_docs, query_type)

    result = {
        "query": query,
        "query_type": query_type,
        "retrieved_documents": retrieved_docs,
        "response": response
    }

    logger.debug("\n=== RESPONSE ===")
    logger.debug(response)

    return result

"""
## Evaluation Framework
"""
logger.info("## Evaluation Framework")

def evaluate_adaptive_vs_standard(pdf_path, test_queries, reference_answers=None):
    """
    Compare adaptive retrieval with standard retrieval on a set of test queries.

    This function processes a document, runs both standard and adaptive retrieval methods
    on each test query, and compares their performance. If reference answers are provided,
    it also evaluates the quality of responses against these references.

    Args:
        pdf_path (str): Path to PDF document to be processed as the knowledge source
        test_queries (List[str]): List of test queries to evaluate both retrieval methods
        reference_answers (List[str], optional): Reference answers for evaluation metrics

    Returns:
        Dict: Evaluation results containing individual query results and overall comparison
    """
    logger.debug("=== EVALUATING ADAPTIVE VS. STANDARD RETRIEVAL ===")

    chunks, vector_store = process_document(pdf_path)

    results = []

    for i, query in enumerate(test_queries):
        logger.debug(f"\n\nQuery {i+1}: {query}")

        logger.debug("\n--- Standard Retrieval ---")
        query_embedding = create_embeddings(query)
        standard_docs = vector_store.similarity_search(query_embedding, k=4)
        standard_response = generate_response(query, standard_docs, "General")

        logger.debug("\n--- Adaptive Retrieval ---")
        query_type = classify_query(query)
        adaptive_docs = adaptive_retrieval(query, vector_store, k=4)
        adaptive_response = generate_response(query, adaptive_docs, query_type)

        result = {
            "query": query,
            "query_type": query_type,
            "standard_retrieval": {
                "documents": standard_docs,
                "response": standard_response
            },
            "adaptive_retrieval": {
                "documents": adaptive_docs,
                "response": adaptive_response
            }
        }

        if reference_answers and i < len(reference_answers):
            result["reference_answer"] = reference_answers[i]

        results.append(result)

        logger.debug("\n--- Responses ---")
        logger.debug(f"Standard: {standard_response[:200]}...")
        logger.debug(f"Adaptive: {adaptive_response[:200]}...")

    if reference_answers:
        comparison = compare_responses(results)
        logger.debug("\n=== EVALUATION RESULTS ===")
        logger.debug(comparison)

    return {
        "results": results,
        "comparison": comparison if reference_answers else "No reference answers provided for evaluation"
    }

def compare_responses(results):
    """
    Compare standard and adaptive responses against reference answers.

    Args:
        results (List[Dict]): Results containing both types of responses

    Returns:
        str: Comparison analysis
    """
    comparison_prompt = """You are an expert evaluator of information retrieval systems.
    Compare the standard retrieval and adaptive retrieval responses for each query.
    Consider factors like accuracy, relevance, comprehensiveness, and alignment with the reference answer.
    Provide a detailed analysis of the strengths and weaknesses of each approach."""

    comparison_text = "# Evaluation of Standard vs. Adaptive Retrieval\n\n"

    for i, result in enumerate(results):
        if "reference_answer" not in result:
            continue

        comparison_text += f"## Query {i+1}: {result['query']}\n"
        comparison_text += f"*Query Type: {result['query_type']}*\n\n"
        comparison_text += f"**Reference Answer:**\n{result['reference_answer']}\n\n"

        comparison_text += f"**Standard Retrieval Response:**\n{result['standard_retrieval']['response']}\n\n"

        comparison_text += f"**Adaptive Retrieval Response:**\n{result['adaptive_retrieval']['response']}\n\n"

        user_prompt = f"""
        Reference Answer: {result['reference_answer']}

        Standard Retrieval Response: {result['standard_retrieval']['response']}

        Adaptive Retrieval Response: {result['adaptive_retrieval']['response']}

        Provide a detailed comparison of the two responses.
        """

        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[
                {"role": "system", "content": comparison_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )

        comparison_text += f"**Comparison Analysis:**\n{response.choices[0].message.content}\n\n"

    return comparison_text  # Return the complete comparison analysis

"""
## Evaluating the Adaptive Retrieval System (Customized Queries)

The final step to use the adaptive RAG evaluation system is to call the evaluate_adaptive_vs_standard() function with your PDF document and test queries:
"""
logger.info("## Evaluating the Adaptive Retrieval System (Customized Queries)")

pdf_path = "data/AI_Information.pdf"

test_queries = [
    "What is Explainable AI (XAI)?",                                              # Factual query - seeking definition/specific information
]

reference_answers = [
    "Explainable AI (XAI) aims to make AI systems transparent and understandable by providing clear explanations of how decisions are made. This helps users trust and effectively manage AI technologies.",
]

evaluation_results = evaluate_adaptive_vs_standard(
    pdf_path=pdf_path,                  # Source document for knowledge extraction
    test_queries=test_queries,          # List of test queries to evaluate
    reference_answers=reference_answers  # Optional ground truth for comparison
)

logger.debug(evaluation_results["comparison"])

logger.info("\n\n[DONE]", bright=True)