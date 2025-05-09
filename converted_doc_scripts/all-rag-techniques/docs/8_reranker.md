# Reranking for Enhanced RAG Systems

This notebook implements reranking techniques to improve retrieval quality in RAG systems. Reranking acts as a second filtering step after initial retrieval to ensure the most relevant content is used for response generation.

## Key Concepts of Reranking

1. **Initial Retrieval**: First pass using basic similarity search (less accurate but faster)
2. **Document Scoring**: Evaluating each retrieved document's relevance to the query
3. **Reordering**: Sorting documents by their relevance scores
4. **Selection**: Using only the most relevant documents for response generation

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
    # Open the PDF file
    mypdf = fitz.open(pdf_path)
    all_text = ""  # Initialize an empty string to store the extracted text

    # Iterate through each page in the PDF
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]  # Get the page
        text = page.get_text("text")  # Extract text from the page
        all_text += text  # Append the extracted text to the all_text string

    return all_text  # Return the extracted text
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
    chunks = []  # Initialize an empty list to store the chunks

    # Loop through the text with a step size of (n - overlap)
    for i in range(0, len(text), n - overlap):
        # Append a chunk of text from index i to i + n to the chunks list
        chunks.append(text[i:i + n])

    return chunks  # Return the list of text chunks
```

## Setting Up the OpenAI API Client
We initialize the OpenAI client to generate embeddings and responses.

```python
# Initialize the OpenAI client with the base URL and API key
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")  # Retrieve the API key from environment variables
)
```

## Building a Simple Vector Store
To demonstrate how reranking integrate with retrieval, let's implement a simple vector store.

```python
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

        # Convert query embedding to numpy array
        query_vector = np.array(query_embedding)

        # Calculate similarities using cosine similarity
        similarities = []
        for i, vector in enumerate(self.vectors):
            # Compute cosine similarity between query vector and stored vector
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))  # Append index and similarity score

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top k results
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],  # Add the corresponding text
                "metadata": self.metadata[idx],  # Add the corresponding metadata
                "similarity": score  # Add the similarity score
            })

        return results  # Return the list of top k similar items
```

## Creating Embeddings

```python
def create_embeddings(text, model="BAAI/bge-en-icl"):
    """
    Creates embeddings for the given text using the specified OpenAI model.

    Args:
    text (str): The input text for which embeddings are to be created.
    model (str): The model to be used for creating embeddings.

    Returns:
    List[float]: The embedding vector.
    """
    # Handle both string and list inputs by converting string input to a list
    input_text = text if isinstance(text, list) else [text]

    # Create embeddings for the input text using the specified model
    response = client.embeddings.create(
        model=model,
        input=input_text
    )

    # If input was a string, return just the first embedding
    if isinstance(text, str):
        return response.data[0].embedding

    # Otherwise, return all embeddings as a list of vectors
    return [item.embedding for item in response.data]
```

## Document Processing Pipeline
Now that we have defined the necessary functions and classes, we can proceed to define the document processing pipeline.

```python
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
    # Extract text from the PDF file
    print("Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_path)

    # Chunk the extracted text
    print("Chunking text...")
    chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} text chunks")

    # Create embeddings for the text chunks
    print("Creating embeddings for chunks...")
    chunk_embeddings = create_embeddings(chunks)

    # Initialize a simple vector store
    store = SimpleVectorStore()

    # Add each chunk and its embedding to the vector store
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={"index": i, "source": pdf_path}
        )

    print(f"Added {len(chunks)} chunks to the vector store")
    return store
```

## Implementing LLM-based Reranking
Let's implement the LLM-based reranking function using the OpenAI API.

```python
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
    print(f"Reranking {len(results)} documents...")  # Print the number of documents to be reranked

    scored_results = []  # Initialize an empty list to store scored results

    # Define the system prompt for the LLM
    system_prompt = """You are an expert at evaluating document relevance for search queries.
Your task is to rate documents on a scale from 0 to 10 based on how well they answer the given query.

Guidelines:
- Score 0-2: Document is completely irrelevant
- Score 3-5: Document has some relevant information but doesn't directly answer the query
- Score 6-8: Document is relevant and partially answers the query
- Score 9-10: Document is highly relevant and directly answers the query

You MUST respond with ONLY a single integer score between 0 and 10. Do not include ANY other text."""

    # Iterate through each result
    for i, result in enumerate(results):
        # Show progress every 5 documents
        if i % 5 == 0:
            print(f"Scoring document {i+1}/{len(results)}...")

        # Define the user prompt for the LLM
        user_prompt = f"""Query: {query}

Document:
{result['text']}

Rate this document's relevance to the query on a scale from 0 to 10:"""

        # Get the LLM response
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        # Extract the score from the LLM response
        score_text = response.choices[0].message.content.strip()

        # Use regex to extract the numerical score
        score_match = re.search(r'\b(10|[0-9])\b', score_text)
        if score_match:
            score = float(score_match.group(1))
        else:
            # If score extraction fails, use similarity score as fallback
            print(f"Warning: Could not extract score from response: '{score_text}', using similarity score instead")
            score = result["similarity"] * 10

        # Append the scored result to the list
        scored_results.append({
            "text": result["text"],
            "metadata": result["metadata"],
            "similarity": result["similarity"],
            "relevance_score": score
        })

    # Sort results by relevance score in descending order
    reranked_results = sorted(scored_results, key=lambda x: x["relevance_score"], reverse=True)

    # Return the top_n results
    return reranked_results[:top_n]
```

## Simple Keyword-based Reranking

```python
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
    # Extract important keywords from the query
    keywords = [word.lower() for word in query.split() if len(word) > 3]

    scored_results = []  # Initialize a list to store scored results

    for result in results:
        document_text = result["text"].lower()  # Convert document text to lowercase

        # Base score starts with vector similarity
        base_score = result["similarity"] * 0.5

        # Initialize keyword score
        keyword_score = 0
        for keyword in keywords:
            if keyword in document_text:
                # Add points for each keyword found
                keyword_score += 0.1

                # Add more points if keyword appears near the beginning
                first_position = document_text.find(keyword)
                if first_position < len(document_text) / 4:  # In the first quarter of the text
                    keyword_score += 0.1

                # Add points for keyword frequency
                frequency = document_text.count(keyword)
                keyword_score += min(0.05 * frequency, 0.2)  # Cap at 0.2

        # Calculate the final score by combining base score and keyword score
        final_score = base_score + keyword_score

        # Append the scored result to the list
        scored_results.append({
            "text": result["text"],
            "metadata": result["metadata"],
            "similarity": result["similarity"],
            "relevance_score": final_score
        })

    # Sort results by final relevance score in descending order
    reranked_results = sorted(scored_results, key=lambda x: x["relevance_score"], reverse=True)

    # Return the top_n results
    return reranked_results[:top_n]
```

## Response Generation

```python
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
    # Define the system prompt to guide the AI's behavior
    system_prompt = "You are a helpful AI assistant. Answer the user's question based only on the provided context. If you cannot find the answer in the context, state that you don't have enough information."

    # Create the user prompt by combining the context and query
    user_prompt = f"""
        Context:
        {context}

        Question: {query}

        Please provide a comprehensive answer based only on the context above.
    """

    # Generate the response using the specified model
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # Return the generated response content
    return response.choices[0].message.content
```

## Full RAG Pipeline with Reranking
So far, we have implemented the core components of the RAG pipeline, including document processing, question answering, and reranking. Now, we will combine these components to create a full RAG pipeline.

```python
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
    # Create query embedding
    query_embedding = create_embeddings(query)

    # Initial retrieval (get more than we need for reranking)
    initial_results = vector_store.similarity_search(query_embedding, k=10)

    # Apply reranking
    if reranking_method == "llm":
        reranked_results = rerank_with_llm(query, initial_results, top_n=top_n)
    elif reranking_method == "keywords":
        reranked_results = rerank_with_keywords(query, initial_results, top_n=top_n)
    else:
        # No reranking, just use top results from initial retrieval
        reranked_results = initial_results[:top_n]

    # Combine context from reranked results
    context = "\n\n===\n\n".join([result["text"] for result in reranked_results])

    # Generate response based on context
    response = generate_response(query, context, model)

    return {
        "query": query,
        "reranking_method": reranking_method,
        "initial_results": initial_results[:top_n],
        "reranked_results": reranked_results,
        "context": context,
        "response": response
    }
```

## Evaluating Reranking Quality

```python
# Load the validation data from a JSON file
with open('data/val.json') as f:
    data = json.load(f)

# Extract the first query from the validation data
query = data[0]['question']

# Extract the reference answer from the validation data
reference_answer = data[0]['ideal_answer']

# pdf_path
pdf_path = "data/AI_Information.pdf"
```

```python
# Process document
vector_store = process_document(pdf_path)

# Example query
query = "Does AI have the potential to transform the way we live and work?"

# Compare different methods
print("Comparing retrieval methods...")

# 1. Standard retrieval (no reranking)
print("\n=== STANDARD RETRIEVAL ===")
standard_results = rag_with_reranking(query, vector_store, reranking_method="none")
print(f"\nQuery: {query}")
print(f"\nResponse:\n{standard_results['response']}")

# 2. LLM-based reranking
print("\n=== LLM-BASED RERANKING ===")
llm_results = rag_with_reranking(query, vector_store, reranking_method="llm")
print(f"\nQuery: {query}")
print(f"\nResponse:\n{llm_results['response']}")

# 3. Keyword-based reranking
print("\n=== KEYWORD-BASED RERANKING ===")
keyword_results = rag_with_reranking(query, vector_store, reranking_method="keywords")
print(f"\nQuery: {query}")
print(f"\nResponse:\n{keyword_results['response']}")
```

```output
Extracting text from PDF...
Chunking text...
Created 42 text chunks
Creating embeddings for chunks...
Added 42 chunks to the vector store
Comparing retrieval methods...

=== STANDARD RETRIEVAL ===

Query: Does AI have the potential to transform the way we live and work?

Response:
Based on the provided context, it is clear that AI has the potential to significantly transform the way we live and work. The context highlights the various applications of AI in different industries, including:

1. Automation and Job Displacement: AI can automate repetitive or routine tasks, potentially displacing some jobs, but also creating new opportunities and transforming existing roles.
2. Reskilling and Upskilling: AI requires workers to reskill and upskill to adapt to new roles and collaborate with AI systems.
3. Human-AI Collaboration: AI tools can augment human capabilities, automate mundane tasks, and provide insights that support decision-making, leading to increased collaboration between humans and AI systems.
4. New Job Roles: The development and deployment of AI create new job roles in areas such as AI development, data science, AI ethics, and AI training.
5. Ethical Considerations: AI raises ethical concerns, including ensuring fairness, transparency, and accountability in AI systems, as well as protecting worker rights and privacy.

In terms of its impact on daily life, AI is transforming business operations, leading to increased efficiency, reduced costs, and improved decision-making. AI-powered tools are also enhancing customer relationship management, supply chain management, and other areas, leading to improved customer experiences and satisfaction.

Furthermore, AI is being used as a creative tool, generating art, music, and literature, and assisting in design processes and scientific discovery. This suggests that AI has the potential to transform the way we live and work, not just in terms of efficiency and productivity, but also in terms of creativity and innovation.

Overall, the context suggests that AI has the potential to revolutionize various aspects of our lives and work, from automation and job displacement to human-AI collaboration, new job roles, and ethical considerations.

=== LLM-BASED RERANKING ===
Reranking 10 documents...
Scoring document 1/10...
Scoring document 6/10...

Query: Does AI have the potential to transform the way we live and work?

Response:
Based on the provided context, it is clear that AI has the potential to significantly transform the way we live and work. The context highlights the various applications of AI in different industries, including:

1. Automation and Job Displacement: AI can automate repetitive or routine tasks, potentially displacing some jobs, but also creating new opportunities and transforming existing roles.
2. Reskilling and Upskilling: AI requires workers to reskill and upskill to adapt to new roles and collaborate with AI systems.
3. Human-AI Collaboration: AI tools can augment human capabilities, automate mundane tasks, and provide insights that support decision-making, leading to increased collaboration between humans and AI systems.
4. New Job Roles: The development and deployment of AI create new job roles in areas such as AI development, data science, AI ethics, and AI training.
5. Ethical Considerations: AI raises ethical concerns, including ensuring fairness, transparency, and accountability in AI systems, as well as protecting worker rights and privacy.

In terms of its impact on daily life, AI is transforming business operations, leading to increased efficiency, reduced costs, and improved decision-making. AI-powered tools are also enhancing customer relationship management, supply chain management, and other areas, leading to improved customer experiences and satisfaction.

Furthermore, AI is being used as a creative tool, generating art, music, and literature, and assisting in design processes and scientific discovery. This suggests that AI has the potential to transform the way we live and work, not just in terms of efficiency and productivity, but also in terms of creativity and innovation.

Overall, the context suggests that AI has the potential to revolutionize various aspects of our lives and work, from automation and job displacement to human-AI collaboration, new job roles, and ethical considerations.

=== KEYWORD-BASED RERANKING ===

Query: Does AI have the potential to transform the way we live and work?

Response:
Based on the provided context, it appears that AI has the potential to significantly transform the way we live and work. The context highlights the various applications of AI in different industries, including business operations, customer service, supply chain management, and social good initiatives.

AI is transforming business operations by increasing efficiency, reducing costs, and improving decision-making. It is also enhancing customer relationship management by providing personalized experiences, predicting customer behavior, and automating customer service interactions. Additionally, AI is optimizing supply chain operations by predicting demand, managing inventory, and streamlining logistics.

Furthermore, AI is being used to address social and environmental challenges, such as climate change, poverty, and healthcare disparities. This suggests that AI has the potential to positively impact various aspects of our lives and work.

However, the context also raises concerns about job displacement, particularly in industries with repetitive or routine tasks. To mitigate these risks, reskilling and upskilling initiatives are necessary to equip workers with the skills needed to adapt to new roles and collaborate with AI systems.

Overall, the context suggests that AI has the potential to transform the way we live and work, but it is essential to address the challenges and risks associated with its development and deployment.
```

```python
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
    # Define the system prompt for the AI evaluator
    system_prompt = """You are an expert evaluator of RAG systems.
    Compare the retrieved contexts and responses from two different retrieval methods.
    Assess which one provides better context and a more accurate, comprehensive answer."""

    # Prepare the comparison text with truncated contexts and responses
    comparison_text = f"""Query: {query}

Standard Retrieval Context:
{standard_results['context'][:1000]}... [truncated]

Standard Retrieval Answer:
{standard_results['response']}

Reranked Retrieval Context:
{reranked_results['context'][:1000]}... [truncated]

Reranked Retrieval Answer:
{reranked_results['response']}"""

    # If a reference answer is provided, include it in the comparison text
    if reference_answer:
        comparison_text += f"""

Reference Answer:
{reference_answer}"""

    # Create the user prompt for the AI evaluator
    user_prompt = f"""
{comparison_text}

Please evaluate which retrieval method provided:
1. More relevant context
2. More accurate answer
3. More comprehensive answer
4. Better overall performance

Provide a detailed analysis with specific examples.
"""

    # Generate the evaluation response using the specified model
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # Return the evaluation output
    return response.choices[0].message.content
```

```python
# Evaluate the quality of reranked results compared to standard results
evaluation = evaluate_reranking(
    query=query,  # The user query
    standard_results=standard_results,  # Results from standard retrieval
    reranked_results=llm_results,  # Results from LLM-based reranking
    reference_answer=reference_answer  # Reference answer for comparison
)

# Print the evaluation results
print("\n=== EVALUATION RESULTS ===")
print(evaluation)
```

```output

=== EVALUATION RESULTS ===
After analyzing the three retrieval methods, I will evaluate which one provides better context and a more accurate, comprehensive answer.

**1. More relevant context:**
Both Standard Retrieval Context and Reranked Retrieval Context provide relevant context, but the Standard Retrieval Context is more comprehensive. It includes a broader range of topics related to AI, such as customer service, algorithmic trading, and management, which are all relevant to the query. The Reranked Retrieval Context, on the other hand, is more focused on the specific topics of automation, job displacement, and human-AI collaboration, which are also relevant but not as comprehensive as the Standard Retrieval Context.

**2. More accurate answer:**
Both Standard Retrieval Answer and Reranked Retrieval Answer provide accurate answers, but the Standard Retrieval Answer is more comprehensive. It covers a wider range of topics related to AI, including its impact on daily life, creativity, and innovation, whereas the Reranked Retrieval Answer is more focused on the specific topics of automation, job displacement, and human-AI collaboration.

**3. More comprehensive answer:**
The Standard Retrieval Answer is more comprehensive, covering a wider range of topics related to AI, including its impact on daily life, creativity, and innovation. The Reranked Retrieval Answer is more focused on the specific topics of automation, job displacement, and human-AI collaboration, which are all relevant but not as comprehensive as the Standard Retrieval Answer.

**4. Better overall performance:**
Based on the analysis, I would rate the Standard Retrieval Context as having better overall performance. It provides a more comprehensive and relevant context, which is essential for providing an accurate and comprehensive answer. The Standard Retrieval Answer also provides a more comprehensive answer, covering a wider range of topics related to AI.

**Detailed analysis with specific examples:**

* The Standard Retrieval Context includes a broader range of topics related to AI, such as customer service, algorithmic trading, and management, which are all relevant to the query. For example, the context mentions "customer service" and "algorithmic trading", which are both relevant to the query. In contrast, the Reranked Retrieval Context is more focused on the specific topics of automation, job displacement, and human-AI collaboration.
* The Standard Retrieval Answer covers a wider range of topics related to AI, including its impact on daily life, creativity, and innovation. For example, the answer mentions "AI is being used as a creative tool, generating art, music, and literature, and assisting in design processes and scientific discovery", which is not mentioned in the Reranked Retrieval Answer.
* The Standard Retrieval Answer also provides more specific examples of how AI is being used in different industries, such as "AI-powered tools are also enhancing customer relationship management, supply chain management, and other areas, leading to improved customer experiences and satisfaction". This provides more context and insight into how AI is being used in different industries.

In conclusion, the Standard Retrieval Context and Answer provide better context and a more accurate, comprehensive answer than the Reranked Retrieval Context and Answer. The Standard Retrieval Context is more comprehensive and relevant, and the Standard Retrieval Answer is more comprehensive and accurate.
```
