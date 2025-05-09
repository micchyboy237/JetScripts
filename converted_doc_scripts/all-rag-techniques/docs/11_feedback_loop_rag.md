# Feedback Loop in RAG

In this notebook, I implement a RAG system with a feedback loop mechanism that continuously improves over time. By collecting and incorporating user feedback, our system learns to provide more relevant and higher-quality responses with each interaction.

Traditional RAG systems are static - they retrieve information based solely on embedding similarity. With a feedback loop, we create a dynamic system that:

- Remembers what worked (and what didn't)
- Adjusts document relevance scores over time
- Incorporates successful Q&A pairs into its knowledge base
- Gets smarter with each user interaction

## Setting Up the Environment
We begin by importing necessary libraries.

```python
import fitz
import os
import numpy as np
import json
from openai import OpenAI
from datetime import datetime
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
```

## Setting Up the OpenAI API Client
We initialize the OpenAI client to generate embeddings and responses.

```python

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")
)
```

## Simple Vector Store Implementation
We'll create a basic vector store to manage document chunks and their embeddings.

```python
class SimpleVectorStore:
    """
    A simple vector store implementation using NumPy.

    This class provides an in-memory storage and retrieval system for
    embedding vectors and their corresponding text chunks and metadata.
    It supports basic similarity search functionality using cosine similarity.
    """
    def __init__(self):
        """
        Initialize the vector store with empty lists for vectors, texts, and metadata.

        The vector store maintains three parallel lists:
        - vectors: NumPy arrays of embedding vectors
        - texts: Original text chunks corresponding to each vector
        - metadata: Optional metadata dictionaries for each item
        """
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add_item(self, text, embedding, metadata=None):
        """
        Add an item to the vector store.

        Args:
            text (str): The original text chunk to store.
            embedding (List[float]): The embedding vector representing the text.
            metadata (dict, optional): Additional metadata for the text chunk,
                                      such as source, timestamp, or relevance scores.
        """
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})

    def similarity_search(self, query_embedding, k=5, filter_func=None):
        """
        Find the most similar items to a query embedding using cosine similarity.

        Args:
            query_embedding (List[float]): Query embedding vector to compare against stored vectors.
            k (int): Number of most similar results to return.
            filter_func (callable, optional): Function to filter results based on metadata.
                                             Takes metadata dict as input and returns boolean.

        Returns:
            List[Dict]: Top k most similar items, each containing:
                - text: The original text
                - metadata: Associated metadata
                - similarity: Raw cosine similarity score
                - relevance_score: Either metadata-based relevance or calculated similarity

        Note: Returns empty list if no vectors are stored or none pass the filter.
        """
        if not self.vectors:
            return []


        query_vector = np.array(query_embedding)


        similarities = []
        for i, vector in enumerate(self.vectors):

            if filter_func and not filter_func(self.metadata[i]):
                continue


            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))


        similarities.sort(key=lambda x: x[1], reverse=True)


        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": score,

                "relevance_score": self.metadata[idx].get("relevance_score", score)
            })

        return results
```

## Creating Embeddings

```python
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
```

## Feedback System Functions
Now we'll implement the core feedback system components.

```python
def get_user_feedback(query, response, relevance, quality, comments=""):
    """
    Format user feedback in a dictionary.

    Args:
        query (str): User's query
        response (str): System's response
        relevance (int): Relevance score (1-5)
        quality (int): Quality score (1-5)
        comments (str): Optional feedback comments

    Returns:
        Dict: Formatted feedback
    """
    return {
        "query": query,
        "response": response,
        "relevance": int(relevance),
        "quality": int(quality),
        "comments": comments,
        "timestamp": datetime.now().isoformat()
    }
```

```python
def store_feedback(feedback, feedback_file="feedback_data.json"):
    """
    Store feedback in a JSON file.

    Args:
        feedback (Dict): Feedback data
        feedback_file (str): Path to feedback file
    """
    with open(feedback_file, "a") as f:
        json.dump(feedback, f)
        f.write("\n")
```

```python
def load_feedback_data(feedback_file="feedback_data.json"):
    """
    Load feedback data from file.

    Args:
        feedback_file (str): Path to feedback file

    Returns:
        List[Dict]: List of feedback entries
    """
    feedback_data = []
    try:
        with open(feedback_file, "r") as f:
            for line in f:
                if line.strip():
                    feedback_data.append(json.loads(line.strip()))
    except FileNotFoundError:
        print("No feedback data file found. Starting with empty feedback.")

    return feedback_data
```

## Document Processing with Feedback Awareness

```python
def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    Process a document for RAG (Retrieval Augmented Generation) with feedback loop.
    This function handles the complete document processing pipeline:
    1. Text extraction from PDF
    2. Text chunking with overlap
    3. Embedding creation for chunks
    4. Storage in vector database with metadata

    Args:
    pdf_path (str): Path to the PDF file to process.
    chunk_size (int): Size of each text chunk in characters.
    chunk_overlap (int): Number of overlapping characters between consecutive chunks.

    Returns:
    Tuple[List[str], SimpleVectorStore]: A tuple containing:
        - List of document chunks
        - Populated vector store with embeddings and metadata
    """

    print("Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_path)


    print("Chunking text...")
    chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} text chunks")


    print("Creating embeddings for chunks...")
    chunk_embeddings = create_embeddings(chunks)


    store = SimpleVectorStore()



    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={
                "index": i,
                "source": pdf_path,
                "relevance_score": 1.0,
                "feedback_count": 0
            }
        )

    print(f"Added {len(chunks)} chunks to the vector store")
    return chunks, store
```

## Relevance Adjustment Based on Feedback

```python
def assess_feedback_relevance(query, doc_text, feedback):
    """
    Use LLM to assess if a past feedback entry is relevant to the current query and document.

    This function helps determine which past feedback should influence the current retrieval
    by sending the current query, past query+feedback, and document content to an LLM
    for relevance assessment.

    Args:
        query (str): Current user query that needs information retrieval
        doc_text (str): Text content of the document being evaluated
        feedback (Dict): Previous feedback data containing 'query' and 'response' keys

    Returns:
        bool: True if the feedback is deemed relevant to current query/document, False otherwise
    """

    system_prompt = """You are an AI system that determines if a past feedback is relevant to a current query and document.
    Answer with ONLY 'yes' or 'no'. Your job is strictly to determine relevance, not to provide explanations."""


    user_prompt = f"""
    Current query: {query}
    Past query that received feedback: {feedback['query']}
    Document content: {doc_text[:500]}... [truncated]
    Past response that received feedback: {feedback['response'][:500]}... [truncated]

    Is this past feedback relevant to the current query and document? (yes/no)
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
    return 'yes' in answer
```

```python
def adjust_relevance_scores(query, results, feedback_data):
    """
    Adjust document relevance scores based on historical feedback to improve retrieval quality.

    This function analyzes past user feedback to dynamically adjust the relevance scores of
    retrieved documents. It identifies feedback that is relevant to the current query context,
    calculates score modifiers based on relevance ratings, and re-ranks the results accordingly.

    Args:
        query (str): Current user query
        results (List[Dict]): Retrieved documents with their original similarity scores
        feedback_data (List[Dict]): Historical feedback containing user ratings

    Returns:
        List[Dict]: Results with adjusted relevance scores, sorted by the new scores
    """

    if not feedback_data:
        return results

    print("Adjusting relevance scores based on feedback history...")


    for i, result in enumerate(results):
        document_text = result["text"]
        relevant_feedback = []



        for feedback in feedback_data:
            is_relevant = assess_feedback_relevance(query, document_text, feedback)
            if is_relevant:
                relevant_feedback.append(feedback)


        if relevant_feedback:


            avg_relevance = sum(f['relevance'] for f in relevant_feedback) / len(relevant_feedback)




            modifier = 0.5 + (avg_relevance / 5.0)


            original_score = result["similarity"]
            adjusted_score = original_score * modifier


            result["original_similarity"] = original_score
            result["similarity"] = adjusted_score
            result["relevance_score"] = adjusted_score
            result["feedback_applied"] = True
            result["feedback_count"] = len(relevant_feedback)


            print(f"  Document {i+1}: Adjusted score from {original_score:.4f} to {adjusted_score:.4f} based on {len(relevant_feedback)} feedback(s)")


    results.sort(key=lambda x: x["similarity"], reverse=True)

    return results
```

## Fine-tuning Our Index with Feedback

```python
def fine_tune_index(current_store, chunks, feedback_data):
    """
    Enhance vector store with high-quality feedback to improve retrieval quality over time.

    This function implements a continuous learning process by:
    1. Identifying high-quality feedback (highly rated Q&A pairs)
    2. Creating new retrieval items from successful interactions
    3. Adding these to the vector store with boosted relevance weights

    Args:
        current_store (SimpleVectorStore): Current vector store containing original document chunks
        chunks (List[str]): Original document text chunks
        feedback_data (List[Dict]): Historical user feedback with relevance and quality ratings

    Returns:
        SimpleVectorStore: Enhanced vector store containing both original chunks and feedback-derived content
    """
    print("Fine-tuning index with high-quality feedback...")



    good_feedback = [f for f in feedback_data if f['relevance'] >= 4 and f['quality'] >= 4]

    if not good_feedback:
        print("No high-quality feedback found for fine-tuning.")
        return current_store


    new_store = SimpleVectorStore()


    for i in range(len(current_store.texts)):
        new_store.add_item(
            text=current_store.texts[i],
            embedding=current_store.vectors[i],
            metadata=current_store.metadata[i].copy()
        )


    for feedback in good_feedback:


        enhanced_text = f"Question: {feedback['query']}\nAnswer: {feedback['response']}"


        embedding = create_embeddings(enhanced_text)


        new_store.add_item(
            text=enhanced_text,
            embedding=embedding,
            metadata={
                "type": "feedback_enhanced",
                "query": feedback["query"],
                "relevance_score": 1.2,
                "feedback_count": 1,
                "original_feedback": feedback
            }
        )

        print(f"Added enhanced content from feedback: {feedback['query'][:50]}...")


    print(f"Fine-tuned index now has {len(new_store.texts)} items (original: {len(chunks)})")
    return new_store
```

## Complete RAG Pipeline with Feedback Loop

```python
def generate_response(query, context, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Generate a response based on the query and context.

    Args:
        query (str): User query
        context (str): Context text from retrieved documents
        model (str): LLM model to use

    Returns:
        str: Generated response
    """

    system_prompt = """You are a helpful AI assistant. Answer the user's question based only on the provided context. If you cannot find the answer in the context, state that you don't have enough information."""


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
```

```python
def rag_with_feedback_loop(query, vector_store, feedback_data, k=5, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Complete RAG pipeline incorporating feedback loop.

    Args:
        query (str): User query
        vector_store (SimpleVectorStore): Vector store with document chunks
        feedback_data (List[Dict]): History of feedback
        k (int): Number of documents to retrieve
        model (str): LLM model for response generation

    Returns:
        Dict: Results including query, retrieved documents, and response
    """
    print(f"\n=== Processing query with feedback-enhanced RAG ===")
    print(f"Query: {query}")


    query_embedding = create_embeddings(query)


    results = vector_store.similarity_search(query_embedding, k=k)


    adjusted_results = adjust_relevance_scores(query, results, feedback_data)


    retrieved_texts = [result["text"] for result in adjusted_results]


    context = "\n\n---\n\n".join(retrieved_texts)


    print("Generating response...")
    response = generate_response(query, context, model)


    result = {
        "query": query,
        "retrieved_documents": adjusted_results,
        "response": response
    }

    print("\n=== Response ===")
    print(response)

    return result
```

## Complete Workflow: From Initial Setup to Feedback Collection

```python
def full_rag_workflow(pdf_path, query, feedback_data=None, feedback_file="feedback_data.json", fine_tune=False):
    """
    Execute a complete RAG workflow with feedback integration for continuous improvement.

    This function orchestrates the entire Retrieval-Augmented Generation process:
    1. Load historical feedback data
    2. Process and chunk the document
    3. Optionally fine-tune the vector index with prior feedback
    4. Perform retrieval and generation with feedback-adjusted relevance scores
    5. Collect new user feedback for future improvement
    6. Store feedback to enable system learning over time

    Args:
        pdf_path (str): Path to the PDF document to be processed
        query (str): User's natural language query
        feedback_data (List[Dict], optional): Pre-loaded feedback data, loads from file if None
        feedback_file (str): Path to the JSON file storing feedback history
        fine_tune (bool): Whether to enhance the index with successful past Q&A pairs

    Returns:
        Dict: Results containing the response and retrieval metadata
    """

    if feedback_data is None:
        feedback_data = load_feedback_data(feedback_file)
        print(f"Loaded {len(feedback_data)} feedback entries from {feedback_file}")


    chunks, vector_store = process_document(pdf_path)



    if fine_tune and feedback_data:
        vector_store = fine_tune_index(vector_store, chunks, feedback_data)



    result = rag_with_feedback_loop(query, vector_store, feedback_data)


    print("\n=== Would you like to provide feedback on this response? ===")
    print("Rate relevance (1-5, with 5 being most relevant):")
    relevance = input()

    print("Rate quality (1-5, with 5 being highest quality):")
    quality = input()

    print("Any comments? (optional, press Enter to skip)")
    comments = input()


    feedback = get_user_feedback(
        query=query,
        response=result["response"],
        relevance=int(relevance),
        quality=int(quality),
        comments=comments
    )


    store_feedback(feedback, feedback_file)
    print("Feedback recorded. Thank you!")

    return result
```

## Evaluating Our Feedback Loop

```python
def evaluate_feedback_loop(pdf_path, test_queries, reference_answers=None):
    """
    Evaluate the impact of feedback loop on RAG quality by comparing performance before and after feedback integration.

    This function runs a controlled experiment to measure how incorporating feedback affects retrieval and generation:
    1. First round: Run all test queries with no feedback
    2. Generate synthetic feedback based on reference answers (if provided)
    3. Second round: Run the same queries with feedback-enhanced retrieval
    4. Compare results between rounds to quantify feedback impact

    Args:
        pdf_path (str): Path to the PDF document used as the knowledge base
        test_queries (List[str]): List of test queries to evaluate system performance
        reference_answers (List[str], optional): Reference/gold standard answers for evaluation
                                                and synthetic feedback generation

    Returns:
        Dict: Evaluation results containing:
            - round1_results: Results without feedback
            - round2_results: Results with feedback
            - comparison: Quantitative comparison metrics between rounds
    """
    print("=== Evaluating Feedback Loop Impact ===")


    temp_feedback_file = "temp_evaluation_feedback.json"


    feedback_data = []



    print("\n=== ROUND 1: NO FEEDBACK ===")
    round1_results = []

    for i, query in enumerate(test_queries):
        print(f"\nQuery {i+1}: {query}")


        chunks, vector_store = process_document(pdf_path)


        result = rag_with_feedback_loop(query, vector_store, [])
        round1_results.append(result)



        if reference_answers and i < len(reference_answers):

            similarity_to_ref = calculate_similarity(result["response"], reference_answers[i])

            relevance = max(1, min(5, int(similarity_to_ref * 5)))
            quality = max(1, min(5, int(similarity_to_ref * 5)))


            feedback = get_user_feedback(
                query=query,
                response=result["response"],
                relevance=relevance,
                quality=quality,
                comments=f"Synthetic feedback based on reference similarity: {similarity_to_ref:.2f}"
            )


            feedback_data.append(feedback)
            store_feedback(feedback, temp_feedback_file)



    print("\n=== ROUND 2: WITH FEEDBACK ===")
    round2_results = []


    chunks, vector_store = process_document(pdf_path)
    vector_store = fine_tune_index(vector_store, chunks, feedback_data)

    for i, query in enumerate(test_queries):
        print(f"\nQuery {i+1}: {query}")


        result = rag_with_feedback_loop(query, vector_store, feedback_data)
        round2_results.append(result)



    comparison = compare_results(test_queries, round1_results, round2_results, reference_answers)


    if os.path.exists(temp_feedback_file):
        os.remove(temp_feedback_file)

    return {
        "round1_results": round1_results,
        "round2_results": round2_results,
        "comparison": comparison
    }
```

## Helper Functions for Evaluation

```python
def calculate_similarity(text1, text2):
    """
    Calculate semantic similarity between two texts using embeddings.

    Args:
        text1 (str): First text
        text2 (str): Second text

    Returns:
        float: Similarity score between 0 and 1
    """

    embedding1 = create_embeddings(text1)
    embedding2 = create_embeddings(text2)


    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)


    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    return similarity
```

```python
def compare_results(queries, round1_results, round2_results, reference_answers=None):
    """
    Compare results from two rounds of RAG.

    Args:
        queries (List[str]): Test queries
        round1_results (List[Dict]): Results from round 1
        round2_results (List[Dict]): Results from round 2
        reference_answers (List[str], optional): Reference answers

    Returns:
        str: Comparison analysis
    """
    print("\n=== COMPARING RESULTS ===")


    system_prompt = """You are an expert evaluator of RAG systems. Compare responses from two versions:
        1. Standard RAG: No feedback used
        2. Feedback-enhanced RAG: Uses a feedback loop to improve retrieval

        Analyze which version provides better responses in terms of:
        - Relevance to the query
        - Accuracy of information
        - Completeness
        - Clarity and conciseness
    """

    comparisons = []


    for i, (query, r1, r2) in enumerate(zip(queries, round1_results, round2_results)):

        comparison_prompt = f"""
        Query: {query}

        Standard RAG Response:
        {r1["response"]}

        Feedback-enhanced RAG Response:
        {r2["response"]}
        """


        if reference_answers and i < len(reference_answers):
            comparison_prompt += f"""
            Reference Answer:
            {reference_answers[i]}
            """

        comparison_prompt += """
        Compare these responses and explain which one is better and why.
        Focus specifically on how the feedback loop has (or hasn't) improved the response quality.
        """


        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": comparison_prompt}
            ],
            temperature=0
        )


        comparisons.append({
            "query": query,
            "analysis": response.choices[0].message.content
        })


        print(f"\nQuery {i+1}: {query}")
        print(f"Analysis: {response.choices[0].message.content[:200]}...")

    return comparisons
```

## Evaluation of the feedback loop (Custom Validation Queries)

```python

pdf_path = "data/AI_Information.pdf"


test_queries = [
    "What is a neural network and how does it function?",







]


reference_answers = [
    "A neural network is a series of algorithms that attempt to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. It consists of layers of nodes, with each node representing a neuron. Neural networks function by adjusting the weights of connections between nodes based on the error of the output compared to the expected result.",







]


evaluation_results = evaluate_feedback_loop(
    pdf_path=pdf_path,
    test_queries=test_queries,
    reference_answers=reference_answers
)
```

```output
=== Evaluating Feedback Loop Impact ===

=== ROUND 1: NO FEEDBACK ===

Query 1: What is a neural network and how does it function?
Extracting text from PDF...
Chunking text...
Created 42 text chunks
Creating embeddings for chunks...
Added 42 chunks to the vector store

=== Processing query with feedback-enhanced RAG ===
Query: What is a neural network and how does it function?
Generating response...

=== Response ===
Based on the provided context, a neural network is a type of deep neural network that is particularly effective for processing data. The context does not provide a detailed explanation of how a neural network functions, but it does mention that neural networks are inspired by the structure and function of the human brain.

However, it can be inferred that a neural network is a complex system that uses multiple layers to analyze data. The context mentions that deep learning is a subfield of machine learning that uses artificial neural networks with multiple layers (deep neural networks) to analyze data.

In the context of the provided text, neural networks are used in various applications such as image recognition, natural language processing, and speech recognition. They are also used in tasks like object detection, facial recognition, and medical image analysis.

While the context does not provide a comprehensive explanation of how a neural network functions, it can be inferred that neural networks are designed to learn patterns and structures in the data through a process of trial and error, similar to reinforcement learning. However, the exact mechanism of how neural networks function is not explicitly stated in the provided context.

=== ROUND 2: WITH FEEDBACK ===
Extracting text from PDF...
Chunking text...
Created 42 text chunks
Creating embeddings for chunks...
Added 42 chunks to the vector store
Fine-tuning index with high-quality feedback...
Added enhanced content from feedback: What is a neural network and how does it function?...
Fine-tuned index now has 43 items (original: 42)

Query 1: What is a neural network and how does it function?

=== Processing query with feedback-enhanced RAG ===
Query: What is a neural network and how does it function?
Adjusting relevance scores based on feedback history...
  Document 1: Adjusted score from 0.8386 to 1.0902 based on 1 feedback(s)
  Document 4: Adjusted score from 0.6162 to 0.8010 based on 1 feedback(s)
  Document 5: Adjusted score from 0.6023 to 0.7830 based on 1 feedback(s)
Generating response...

=== Response ===
Based on the provided context, a neural network is a complex system that uses multiple layers to analyze data. It is inspired by the structure and function of the human brain and is particularly effective for processing data. Neural networks are used in various applications such as image recognition, natural language processing, and speech recognition.

The context does not provide a detailed explanation of how a neural network functions, but it can be inferred that neural networks are designed to learn patterns and structures in the data through a process of trial and error. This process is similar to reinforcement learning, where the neural network receives feedback in the form of rewards or penalties, allowing it to adjust its parameters and improve its performance over time.

Neural networks are composed of multiple layers, including convolutional layers, recurrent layers, and others. These layers work together to analyze the input data and make predictions or take actions. The exact mechanism of how neural networks function is not explicitly stated in the provided context, but it is clear that they are a powerful tool for analyzing and processing complex data.

In the context of the provided text, neural networks are used in various applications such as:

* Image recognition
* Natural language processing
* Speech recognition
* Object detection
* Facial recognition
* Medical image analysis

Overall, neural networks are a complex and powerful tool for analyzing and processing data, and their applications continue to expand across various industries and domains.

=== COMPARING RESULTS ===

Query 1: What is a neural network and how does it function?
Analysis: Comparing the two responses, the feedback-enhanced RAG response is significantly better than the standard RAG response. Here's a breakdown of the improvements:

1. **Relevance to the query**: Both res...
```

```python

```

## Visualizing Feedback Impact

```python

comparisons = evaluation_results['comparison']


print("\n=== FEEDBACK IMPACT ANALYSIS ===\n")
for i, comparison in enumerate(comparisons):
    print(f"Query {i+1}: {comparison['query']}")
    print(f"\nAnalysis of feedback impact:")
    print(comparison['analysis'])
    print("\n" + "-"*50 + "\n")


round_responses = [evaluation_results[f'round{round_num}_results'] for round_num in range(1, len(evaluation_results) - 1)]
response_lengths = [[len(r["response"]) for r in round] for round in round_responses]

print("\nResponse length comparison (proxy for completeness):")
avg_lengths = [sum(lengths) / len(lengths) for lengths in response_lengths]
for round_num, avg_len in enumerate(avg_lengths, start=1):
    print(f"Round {round_num}: {avg_len:.1f} chars")

if len(avg_lengths) > 1:
    changes = [(avg_lengths[i] - avg_lengths[i-1]) / avg_lengths[i-1] * 100 for i in range(1, len(avg_lengths))]
    for round_num, change in enumerate(changes, start=2):
        print(f"Change from Round {round_num-1} to Round {round_num}: {change:.1f}%")
```

```output

=== FEEDBACK IMPACT ANALYSIS ===

Query 1: What is a neural network and how does it function?

Analysis of feedback impact:
Comparing the two responses, the feedback-enhanced RAG response is significantly better than the standard RAG response. Here's a breakdown of the improvements:

1. **Relevance to the query**: Both responses address the query, but the feedback-enhanced RAG response provides a more comprehensive and detailed explanation of what a neural network is and how it functions. It also provides more specific examples of applications, which shows a better understanding of the topic.

2. **Accuracy of information**: The feedback-enhanced RAG response is more accurate, as it provides a clear and concise explanation of how neural networks function, including the concept of trial and error, reinforcement learning, and the role of feedback. The standard RAG response is less accurate, as it only mentions that neural networks are inspired by the human brain and that they are used in various applications, but does not provide a clear explanation of how they function.

3. **Completeness**: The feedback-enhanced RAG response is more complete, as it provides a detailed explanation of the components of a neural network, including convolutional layers, recurrent layers, and others. The standard RAG response only mentions that neural networks are used in various applications, but does not provide a clear explanation of how they function.

4. **Clarity and conciseness**: The feedback-enhanced RAG response is more concise and clear, as it provides a clear and concise explanation of how neural networks function and their applications. The standard RAG response is less concise and less clear, as it only mentions that neural networks are inspired by the human brain and that they are used in various applications, but does not provide a clear explanation of how they function.

The feedback loop has significantly improved the response quality by:

* Providing a clear and concise explanation of how neural networks function
* Providing more specific examples of applications
* Addressing the limitations of the standard RAG response (e.g., lack of clarity and concision)
* Providing a more accurate explanation of the components of a neural network

The feedback loop has helped to identify the areas where the standard RAG response was lacking and has provided a more comprehensive and detailed explanation of the topic. This is a clear example of how a feedback loop can improve the quality of a response.

--------------------------------------------------


Response length comparison (proxy for completeness):
Round 1: 1256.0 chars
```
