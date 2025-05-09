# Relevant Segment Extraction (RSE) for Enhanced RAG

In this notebook, we implement a Relevant Segment Extraction (RSE) technique to improve the context quality in our RAG system. Rather than simply retrieving a collection of isolated chunks, we identify and reconstruct continuous segments of text that provide better context to our language model.

## Key Concept

Relevant chunks tend to be clustered together within documents. By identifying these clusters and preserving their continuity, we provide more coherent context for the LLM to work with.

## Setting Up the Environment
We begin by importing necessary libraries.

```python
import fitz
import os
import numpy as np
import json
from openai import OpenAI
import re
```

## Extracting Text from a PDF File
To implement RAG, we first need a source of textual data. In this case, we extract text from a PDF file using the PyMuPDF library.

```python
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file and prints the first `num_chars` characters.

    Args:
    pdf_path (str): Path to the PDF file.

    Returns:
    str: Extracted text from the PDF.
    """

    mypdf = fitz.open(pdf_path)
    all_text = ""


    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]
        text = page.get_text("text")
        all_text += text

    return all_text
```

## Chunking the Extracted Text
Once we have the extracted text, we divide it into smaller, overlapping chunks to improve retrieval accuracy.

```python
def chunk_text(text, chunk_size=800, overlap=0):
    """
    Split text into non-overlapping chunks.
    For RSE, we typically want non-overlapping chunks so we can reconstruct segments properly.

    Args:
        text (str): Input text to chunk
        chunk_size (int): Size of each chunk in characters
        overlap (int): Overlap between chunks in characters

    Returns:
        List[str]: List of text chunks
    """
    chunks = []


    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)

    return chunks
```

## Setting Up the OpenAI API Client
We initialize the OpenAI client to generate embeddings and responses.

```python

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")
)
```

## Building a Simple Vector Store
let's implement a simple vector store.

```python
class SimpleVectorStore:
    """
    A lightweight vector store implementation using NumPy.
    """
    def __init__(self, dimension=1536):
        """
        Initialize the vector store.

        Args:
            dimension (int): Dimension of embeddings
        """
        self.dimension = dimension
        self.vectors = []
        self.documents = []
        self.metadata = []

    def add_documents(self, documents, vectors=None, metadata=None):
        """
        Add documents to the vector store.

        Args:
            documents (List[str]): List of document chunks
            vectors (List[List[float]], optional): List of embedding vectors
            metadata (List[Dict], optional): List of metadata dictionaries
        """
        if vectors is None:
            vectors = [None] * len(documents)

        if metadata is None:
            metadata = [{} for _ in range(len(documents))]

        for doc, vec, meta in zip(documents, vectors, metadata):
            self.documents.append(doc)
            self.vectors.append(vec)
            self.metadata.append(meta)

    def search(self, query_vector, top_k=5):
        """
        Search for most similar documents.

        Args:
            query_vector (List[float]): Query embedding vector
            top_k (int): Number of results to return

        Returns:
            List[Dict]: List of results with documents, scores, and metadata
        """
        if not self.vectors or not self.documents:
            return []


        query_array = np.array(query_vector)


        similarities = []
        for i, vector in enumerate(self.vectors):
            if vector is not None:

                similarity = np.dot(query_array, vector) / (
                    np.linalg.norm(query_array) * np.linalg.norm(vector)
                )
                similarities.append((i, similarity))


        similarities.sort(key=lambda x: x[1], reverse=True)


        results = []
        for i, score in similarities[:top_k]:
            results.append({
                "document": self.documents[i],
                "score": float(score),
                "metadata": self.metadata[i]
            })

        return results
```

## Creating Embeddings for Text Chunks
Embeddings transform text into numerical vectors, which allow for efficient similarity search.

```python
def create_embeddings(texts, model="BAAI/bge-en-icl"):
    """
    Generate embeddings for texts.

    Args:
        texts (List[str]): List of texts to embed
        model (str): Embedding model to use

    Returns:
        List[List[float]]: List of embedding vectors
    """
    if not texts:
        return []


    batch_size = 100
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]


        response = client.embeddings.create(
            input=batch,
            model=model
        )


        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings
```

## Processing Documents with RSE
Now let's implement the core RSE functionality.

```python
def process_document(pdf_path, chunk_size=800):
    """
    Process a document for use with RSE.

    Args:
        pdf_path (str): Path to the PDF document
        chunk_size (int): Size of each chunk in characters

    Returns:
        Tuple[List[str], SimpleVectorStore, Dict]: Chunks, vector store, and document info
    """
    print("Extracting text from document...")

    text = extract_text_from_pdf(pdf_path)

    print("Chunking text into non-overlapping segments...")

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=0)
    print(f"Created {len(chunks)} chunks")

    print("Generating embeddings for chunks...")

    chunk_embeddings = create_embeddings(chunks)


    vector_store = SimpleVectorStore()


    metadata = [{"chunk_index": i, "source": pdf_path} for i in range(len(chunks))]
    vector_store.add_documents(chunks, chunk_embeddings, metadata)


    doc_info = {
        "chunks": chunks,
        "source": pdf_path,
    }

    return chunks, vector_store, doc_info
```

## RSE Core Algorithm: Computing Chunk Values and Finding Best Segments
Now that we have the necessary functions to process a document and generate embeddings for its chunks, we can implement the core algorithm for RSE.

```python
def calculate_chunk_values(query, chunks, vector_store, irrelevant_chunk_penalty=0.2):
    """
    Calculate chunk values by combining relevance and position.

    Args:
        query (str): Query text
        chunks (List[str]): List of document chunks
        vector_store (SimpleVectorStore): Vector store containing the chunks
        irrelevant_chunk_penalty (float): Penalty for irrelevant chunks

    Returns:
        List[float]: List of chunk values
    """

    query_embedding = create_embeddings([query])[0]


    num_chunks = len(chunks)
    results = vector_store.search(query_embedding, top_k=num_chunks)


    relevance_scores = {result["metadata"]["chunk_index"]: result["score"] for result in results}


    chunk_values = []
    for i in range(num_chunks):

        score = relevance_scores.get(i, 0.0)

        value = score - irrelevant_chunk_penalty
        chunk_values.append(value)

    return chunk_values
```

```python
def find_best_segments(chunk_values, max_segment_length=20, total_max_length=30, min_segment_value=0.2):
    """
    Find the best segments using a variant of the maximum sum subarray algorithm.

    Args:
        chunk_values (List[float]): Values for each chunk
        max_segment_length (int): Maximum length of a single segment
        total_max_length (int): Maximum total length across all segments
        min_segment_value (float): Minimum value for a segment to be considered

    Returns:
        List[Tuple[int, int]]: List of (start, end) indices for best segments
    """
    print("Finding optimal continuous text segments...")

    best_segments = []
    segment_scores = []
    total_included_chunks = 0


    while total_included_chunks < total_max_length:
        best_score = min_segment_value
        best_segment = None


        for start in range(len(chunk_values)):

            if any(start >= s[0] and start < s[1] for s in best_segments):
                continue


            for length in range(1, min(max_segment_length, len(chunk_values) - start) + 1):
                end = start + length


                if any(end > s[0] and end <= s[1] for s in best_segments):
                    continue


                segment_value = sum(chunk_values[start:end])


                if segment_value > best_score:
                    best_score = segment_value
                    best_segment = (start, end)


        if best_segment:
            best_segments.append(best_segment)
            segment_scores.append(best_score)
            total_included_chunks += best_segment[1] - best_segment[0]
            print(f"Found segment {best_segment} with score {best_score:.4f}")
        else:

            break


    best_segments = sorted(best_segments, key=lambda x: x[0])

    return best_segments, segment_scores
```

## Reconstructing and Using Segments for RAG

```python
def reconstruct_segments(chunks, best_segments):
    """
    Reconstruct text segments based on chunk indices.

    Args:
        chunks (List[str]): List of all document chunks
        best_segments (List[Tuple[int, int]]): List of (start, end) indices for segments

    Returns:
        List[str]: List of reconstructed text segments
    """
    reconstructed_segments = []

    for start, end in best_segments:

        segment_text = " ".join(chunks[start:end])

        reconstructed_segments.append({
            "text": segment_text,
            "segment_range": (start, end),
        })

    return reconstructed_segments
```

```python
def format_segments_for_context(segments):
    """
    Format segments into a context string for the LLM.

    Args:
        segments (List[Dict]): List of segment dictionaries

    Returns:
        str: Formatted context text
    """
    context = []

    for i, segment in enumerate(segments):

        segment_header = f"SEGMENT {i+1} (Chunks {segment['segment_range'][0]}-{segment['segment_range'][1]-1}):"
        context.append(segment_header)
        context.append(segment['text'])
        context.append("-" * 80)


    return "\n\n".join(context)
```

## Generating Responses with RSE Context

```python
def generate_response(query, context, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Generate a response based on the query and context.

    Args:
        query (str): User query
        context (str): Context text from relevant segments
        model (str): LLM model to use

    Returns:
        str: Generated response
    """
    print("Generating response using relevant segments as context...")


    system_prompt = """You are a helpful assistant that answers questions based on the provided context.
    The context consists of document segments that have been retrieved as relevant to the user's query.
    Use the information from these segments to provide a comprehensive and accurate answer.
    If the context doesn't contain relevant information to answer the question, say so clearly."""


    user_prompt = f"""
Context:
{context}

Question: {query}

Please provide a helpful answer based on the context provided.
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
```

## Complete RSE Pipeline Function

```python
def rag_with_rse(pdf_path, query, chunk_size=800, irrelevant_chunk_penalty=0.2):
    """
    Complete RAG pipeline with Relevant Segment Extraction.

    Args:
        pdf_path (str): Path to the document
        query (str): User query
        chunk_size (int): Size of chunks
        irrelevant_chunk_penalty (float): Penalty for irrelevant chunks

    Returns:
        Dict: Result with query, segments, and response
    """
    print("\n=== STARTING RAG WITH RELEVANT SEGMENT EXTRACTION ===")
    print(f"Query: {query}")


    chunks, vector_store, doc_info = process_document(pdf_path, chunk_size)


    print("\nCalculating relevance scores and chunk values...")
    chunk_values = calculate_chunk_values(query, chunks, vector_store, irrelevant_chunk_penalty)


    best_segments, scores = find_best_segments(
        chunk_values,
        max_segment_length=20,
        total_max_length=30,
        min_segment_value=0.2
    )


    print("\nReconstructing text segments from chunks...")
    segments = reconstruct_segments(chunks, best_segments)


    context = format_segments_for_context(segments)


    response = generate_response(query, context)


    result = {
        "query": query,
        "segments": segments,
        "response": response
    }

    print("\n=== FINAL RESPONSE ===")
    print(response)

    return result
```

## Comparing with Standard Retrieval
Let's implement a standard retrieval approach to compare with RSE:

```python
def standard_top_k_retrieval(pdf_path, query, k=10, chunk_size=800):
    """
    Standard RAG with top-k retrieval.

    Args:
        pdf_path (str): Path to the document
        query (str): User query
        k (int): Number of chunks to retrieve
        chunk_size (int): Size of chunks

    Returns:
        Dict: Result with query, chunks, and response
    """
    print("\n=== STARTING STANDARD TOP-K RETRIEVAL ===")
    print(f"Query: {query}")


    chunks, vector_store, doc_info = process_document(pdf_path, chunk_size)


    print("Creating query embedding and retrieving chunks...")
    query_embedding = create_embeddings([query])[0]


    results = vector_store.search(query_embedding, top_k=k)
    retrieved_chunks = [result["document"] for result in results]


    context = "\n\n".join([
        f"CHUNK {i+1}:\n{chunk}"
        for i, chunk in enumerate(retrieved_chunks)
    ])


    response = generate_response(query, context)


    result = {
        "query": query,
        "chunks": retrieved_chunks,
        "response": response
    }

    print("\n=== FINAL RESPONSE ===")
    print(response)

    return result
```

## Evaluation of RSE

```python
def evaluate_methods(pdf_path, query, reference_answer=None):
    """
    Compare RSE with standard top-k retrieval.

    Args:
        pdf_path (str): Path to the document
        query (str): User query
        reference_answer (str, optional): Reference answer for evaluation
    """
    print("\n========= EVALUATION =========\n")


    rse_result = rag_with_rse(pdf_path, query)


    standard_result = standard_top_k_retrieval(pdf_path, query)


    if reference_answer:
        print("\n=== COMPARING RESULTS ===")


        evaluation_prompt = f"""
            Query: {query}

            Reference Answer:
            {reference_answer}

            Response from Standard Retrieval:
            {standard_result["response"]}

            Response from Relevant Segment Extraction:
            {rse_result["response"]}

            Compare these two responses against the reference answer. Which one is:
            1. More accurate and comprehensive
            2. Better at addressing the user's query
            3. Less likely to include irrelevant information

            Explain your reasoning for each point.
        """

        print("Evaluating responses against reference answer...")


        evaluation = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[
                {"role": "system", "content": "You are an objective evaluator of RAG system responses."},
                {"role": "user", "content": evaluation_prompt}
            ]
        )


        print("\n=== EVALUATION RESULTS ===")
        print(evaluation.choices[0].message.content)


    return {
        "rse_result": rse_result,
        "standard_result": standard_result
    }
```

```python

with open('data/val.json') as f:
    data = json.load(f)


query = data[0]['question']


reference_answer = data[0]['ideal_answer']


pdf_path = "data/AI_Information.pdf"


results = evaluate_methods(pdf_path, query, reference_answer)
```

```output

========= EVALUATION =========


=== STARTING RAG WITH RELEVANT SEGMENT EXTRACTION ===
Query: What is 'Explainable AI' and why is it considered important?
Extracting text from document...
Chunking text into non-overlapping segments...
Created 42 chunks
Generating embeddings for chunks...

Calculating relevance scores and chunk values...
Finding optimal continuous text segments...
Found segment (21, 41) with score 9.0718
Found segment (0, 20) with score 8.8685

Reconstructing text segments from chunks...
Generating response using relevant segments as context...

=== FINAL RESPONSE ===
Based on the context provided, Explainable AI (XAI) refers to the development of techniques that make AI systems more transparent and understandable. The goal of XAI is to provide insights into how AI models make decisions, enhancing trust and accountability in AI systems.

XAI is considered important for several reasons:

1. **Building trust**: XAI helps users understand how AI systems arrive at their decisions, which is essential for building trust in AI. When users can see how AI systems work, they are more likely to accept the results.
2. **Addressing bias**: XAI can help identify biases in AI systems by providing insights into how they make decisions. By understanding how AI systems work, developers can identify and address biases in the data they are trained on.
3. **Improving accountability**: XAI enables developers to take responsibility for the decisions made by AI systems. By providing explanations for AI decisions, developers can be held accountable for any errors or biases in the system.
4. **Enhancing transparency**: XAI provides insights into how AI systems work, which is essential for transparency in AI decision-making. This is particularly important in high-stakes applications, such as healthcare or finance, where users need to understand how AI systems arrive at their decisions.

Overall, XAI is considered important because it addresses the need for transparency, accountability, and trust in AI systems. By providing insights into how AI models make decisions, XAI can help build trust, address bias, and improve accountability in AI development and deployment.

=== STARTING STANDARD TOP-K RETRIEVAL ===
Query: What is 'Explainable AI' and why is it considered important?
Extracting text from document...
Chunking text into non-overlapping segments...
Created 42 chunks
Generating embeddings for chunks...
Creating query embedding and retrieving chunks...
Generating response using relevant segments as context...

=== FINAL RESPONSE ===
Based on the provided context, Explainable AI (XAI) is a technique that aims to make AI decisions more understandable, enabling users to assess their fairness and accuracy. XAI techniques are designed to provide insights into how AI systems arrive at their decisions, enhancing transparency and explainability.

XAI is considered important for several reasons:

1. **Building trust in AI**: By making AI decisions more understandable, XAI helps build trust in AI systems, which is essential for their widespread adoption.
2. **Addressing potential harms**: XAI can help identify potential biases and errors in AI decision-making, allowing for more effective mitigation and prevention of harms.
3. **Ensuring accountability**: XAI provides a way to establish accountability for AI decisions, which is crucial for addressing potential consequences and ensuring ethical behavior.
4. **Improving fairness and accuracy**: By providing insights into AI decision-making, XAI can help identify and address biases and errors, leading to more fair and accurate outcomes.

Overall, Explainable AI is a critical aspect of developing trustworthy, fair, and accurate AI systems, and its importance will only continue to grow as AI becomes increasingly pervasive in various domains.

=== COMPARING RESULTS ===
Evaluating responses against reference answer...

=== EVALUATION RESULTS ===
Based on the comparison, I would conclude that:

1. **The Response from Standard Retrieval is more accurate and comprehensive:**
   The Response from Standard Retrieval provides a clear definition of Explainable AI (XAI) and its importance. It explains the goals of XAI, its key aspects (transparency and understandability), and its benefits. The explanation highlights the reasons why XAI is considered important, which includes building trust, addressing potential harms, ensuring accountability, and improving fairness and accuracy.

   On the other hand, the Response from Relevant Segment Extraction provides a simplified explanation of XAI but focuses more on the aspects of trust, bias, accountability, and transparency. While it does provide a clear overview of XAI, it is more concise and somewhat less detailed than the Response from Standard Retrieval.

2. **The Response from Relevant Segment Extraction is better at addressing the user's query:**
   The Response from Standard Retrieval not only answers the question but also provides additional context and importance. The Response from Relevant Segment Extraction, however, more closely addresses the original question by focusing on the core aspects of XAI and its advantages.

3. **The Response from Standard Retrieval is less likely to include irrelevant information:**
   This response is less likely to contain unnecessary details. The Response from Standard Retrieval provides a clear and concise answer with a clear structure, focusing on the key points of XAI and its importance. In contrast, the Response from Relevant Segment Extraction might include some details about AI systems, updates, or other related information that is not essential to answering the original question.

However, the Response from Standard Retrieval includes a more relevant and comprehensive discussion of the reference answer's points, than the Response from Relevant Segment Extraction, which is closer to the reference answer but strays closer to an ensuing span-specific update concerning explainable AI underpoint of the start precisuliar generation machinery.
```
