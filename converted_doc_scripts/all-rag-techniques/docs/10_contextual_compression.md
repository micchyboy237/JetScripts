# Contextual Compression for Enhanced RAG Systems
In this notebook, I implement a contextual compression technique to improve our RAG system's efficiency. We'll filter and compress retrieved text chunks to keep only the most relevant parts, reducing noise and improving response quality.

When retrieving documents for RAG, we often get chunks containing both relevant and irrelevant information. Contextual compression helps us:

- Remove irrelevant sentences and paragraphs
- Focus only on query-relevant information
- Maximize the useful signal in our context window

Let's implement this approach from scratch!

## Setting Up the Environment
We begin by importing necessary libraries.

```python
import fitz
import os
import numpy as np
import json
from openai import OpenAI
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
    api_key= os.environ.get("OPENAI_API_KEY")
)
```

## Building a Simple Vector Store
let's implement a simple vector store since we cannot use FAISS.

```python
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
```

## Embedding Generation

```python
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
```

## Building Our Document Processing Pipeline

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
            metadata={"index": i, "source": pdf_path}
        )

    print(f"Added {len(chunks)} chunks to the vector store")
    return store
```

## Implementing Contextual Compression
This is the core of our approach - we'll use an LLM to filter and compress retrieved content.

```python
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
    else:
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
```

## Implementing Batch Compression
For efficiency, we'll compress multiple chunks in one go when possible.

```python
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
    print(f"Compressing {len(chunks)} chunks...")
    results = []
    total_original_length = 0
    total_compressed_length = 0


    for i, chunk in enumerate(chunks):
        print(f"Compressing chunk {i+1}/{len(chunks)}...")

        compressed_chunk, compression_ratio = compress_chunk(chunk, query, compression_type, model)
        results.append((compressed_chunk, compression_ratio))

        total_original_length += len(chunk)
        total_compressed_length += len(compressed_chunk)


    overall_ratio = (total_original_length - total_compressed_length) / total_original_length * 100
    print(f"Overall compression ratio: {overall_ratio:.2f}%")

    return results
```

## Response Generation Function

```python
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
```

## The Complete RAG Pipeline with Contextual Compression

```python

```

```python
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
    print("\n=== RAG WITH CONTEXTUAL COMPRESSION ===")
    print(f"Query: {query}")
    print(f"Compression type: {compression_type}")


    vector_store = process_document(pdf_path)


    query_embedding = create_embeddings(query)


    print(f"Retrieving top {k} chunks...")
    results = vector_store.similarity_search(query_embedding, k=k)
    retrieved_chunks = [result["text"] for result in results]


    compressed_results = batch_compress_chunks(retrieved_chunks, query, compression_type, model)
    compressed_chunks = [result[0] for result in compressed_results]
    compression_ratios = [result[1] for result in compressed_results]


    filtered_chunks = [(chunk, ratio) for chunk, ratio in zip(compressed_chunks, compression_ratios) if chunk.strip()]

    if not filtered_chunks:

        print("Warning: All chunks were compressed to empty strings. Using original chunks.")
        filtered_chunks = [(chunk, 0.0) for chunk in retrieved_chunks]
    else:
        compressed_chunks, compression_ratios = zip(*filtered_chunks)


    context = "\n\n---\n\n".join(compressed_chunks)


    print("Generating response based on compressed chunks...")
    response = generate_response(query, context, model)


    result = {
        "query": query,
        "original_chunks": retrieved_chunks,
        "compressed_chunks": compressed_chunks,
        "compression_ratios": compression_ratios,
        "context_length_reduction": f"{sum(compression_ratios)/len(compression_ratios):.2f}%",
        "response": response
    }

    print("\n=== RESPONSE ===")
    print(response)

    return result
```

## Comparing RAG With and Without Compression
Let's create a function to compare standard RAG with our compression-enhanced version:


```python
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
    print("\n=== STANDARD RAG ===")
    print(f"Query: {query}")


    vector_store = process_document(pdf_path)


    query_embedding = create_embeddings(query)


    print(f"Retrieving top {k} chunks...")
    results = vector_store.similarity_search(query_embedding, k=k)
    retrieved_chunks = [result["text"] for result in results]


    context = "\n\n---\n\n".join(retrieved_chunks)


    print("Generating response...")
    response = generate_response(query, context, model)


    result = {
        "query": query,
        "chunks": retrieved_chunks,
        "response": response
    }

    print("\n=== RESPONSE ===")
    print(response)

    return result
```

## Evaluating Our Approach
Now, let's implement a function to evaluate and compare the responses:

```python
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
```

```python
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
    print("\n=== EVALUATING CONTEXTUAL COMPRESSION ===")
    print(f"Query: {query}")


    standard_result = standard_rag(pdf_path, query)


    compression_results = {}


    for comp_type in compression_types:
        print(f"\nTesting {comp_type} compression...")
        compression_results[comp_type] = rag_with_compression(pdf_path, query, compression_type=comp_type)


    responses = {
        "standard": standard_result["response"]
    }
    for comp_type in compression_types:
        responses[comp_type] = compression_results[comp_type]["response"]


    if reference_answer:
        evaluation = evaluate_responses(query, responses, reference_answer)
        print("\n=== EVALUATION RESULTS ===")
        print(evaluation)
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
```

## Running Our Complete System (Custom Query)

```python

pdf_path = "data/AI_Information.pdf"


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
```

```output

=== EVALUATING CONTEXTUAL COMPRESSION ===
Query: What are the ethical concerns surrounding the use of AI in decision-making?

=== STANDARD RAG ===
Query: What are the ethical concerns surrounding the use of AI in decision-making?
Extracting text from PDF...
Chunking text...
Created 42 text chunks
Creating embeddings for chunks...
Added 42 chunks to the vector store
Retrieving top 10 chunks...
Generating response...

=== RESPONSE ===
The ethical concerns surrounding the use of AI in decision-making include:

1. Bias and Fairness: AI systems can inherit and amplify biases present in the data they are trained on, leading to unfair or discriminatory outcomes.
2. Transparency and Explainability: Many AI systems, particularly deep learning models, are "black boxes," making it difficult to understand how they arrive at their decisions, which can lead to a lack of trust and accountability.
3. Privacy and Data Protection: AI systems often rely on large amounts of data, raising concerns about privacy and data protection, and ensuring responsible data handling is crucial.
4. Accountability and Responsibility: Establishing accountability and responsibility for AI systems is essential for addressing potential harms and ensuring ethical behavior.
5. Unintended Consequences: As AI systems become more autonomous, questions arise about control, accountability, and the potential for unintended consequences.

These concerns highlight the need for careful consideration of the ethical implications of AI in decision-making, including:

* Ensuring fairness and mitigating bias in AI systems
* Enhancing transparency and explainability in AI decision-making
* Protecting sensitive information and ensuring responsible data handling
* Establishing clear guidelines and ethical frameworks for AI development and deployment
* Addressing the potential for unintended consequences and ensuring accountability and responsibility

By addressing these concerns, we can build trust in AI systems and ensure that they are developed and deployed in a way that aligns with societal values and promotes the well-being of individuals and society.

Testing selective compression...

=== RAG WITH CONTEXTUAL COMPRESSION ===
Query: What are the ethical concerns surrounding the use of AI in decision-making?
Compression type: selective
Extracting text from PDF...
Chunking text...
Created 42 text chunks
Creating embeddings for chunks...
Added 42 chunks to the vector store
Retrieving top 10 chunks...
Compressing 10 chunks...
Compressing chunk 1/10...
Compressing chunk 2/10...
Compressing chunk 3/10...
Compressing chunk 4/10...
Compressing chunk 5/10...
Compressing chunk 6/10...
Compressing chunk 7/10...
Compressing chunk 8/10...
Compressing chunk 9/10...
Compressing chunk 10/10...
Overall compression ratio: 39.93%
Generating response based on compressed chunks...

=== RESPONSE ===
The ethical concerns surrounding the use of AI in decision-making include:

1. Bias and Fairness: AI systems can inherit and amplify biases present in the data they are trained on, leading to unfair or discriminatory outcomes.
2. Transparency and Explainability: Many AI systems, particularly deep learning models, are "black boxes," making it difficult to understand how they arrive at their decisions, which can lead to a lack of trust and accountability.
3. Privacy and Data Protection: AI systems often rely on large amounts of data, raising concerns about privacy and data protection, and ensuring responsible data handling is crucial.
4. Safety and Control: As AI systems become more autonomous, questions arise about control, accountability, and the potential for unintended consequences.
5. Accountability and Responsibility: Establishing accountability and responsibility for AI systems is essential for addressing potential harms and ensuring ethical behavior.
6. Economic and Social Impacts: Addressing the potential economic and social impacts of AI-driven automation is a key challenge.
7. Respect for Human Rights: AI systems should be designed to respect human rights, including privacy, non-discrimination, and beneficence.
8. Robustness and Reliability: Ensuring that AI systems are robust and reliable is essential for building trust.
9. Empowerment of Users: Empowering users with control over AI systems and providing them with agency in their interactions with AI enhances trust.
10. Ethical Considerations in Design and Development: Incorporating ethical considerations into the design and development of AI systems is crucial for building trust.

These concerns highlight the need for a comprehensive approach to addressing the ethical implications of AI in decision-making, including the development of clear guidelines, ethical frameworks, and regulations to ensure that AI systems are designed and deployed in a responsible and trustworthy manner.

Testing summary compression...

=== RAG WITH CONTEXTUAL COMPRESSION ===
Query: What are the ethical concerns surrounding the use of AI in decision-making?
Compression type: summary
Extracting text from PDF...
Chunking text...
Created 42 text chunks
Creating embeddings for chunks...
Added 42 chunks to the vector store
Retrieving top 10 chunks...
Compressing 10 chunks...
Compressing chunk 1/10...
Compressing chunk 2/10...
Compressing chunk 3/10...
Compressing chunk 4/10...
Compressing chunk 5/10...
Compressing chunk 6/10...
Compressing chunk 7/10...
Compressing chunk 8/10...
Compressing chunk 9/10...
Compressing chunk 10/10...
Overall compression ratio: 63.87%
Generating response based on compressed chunks...

=== RESPONSE ===
The ethical concerns surrounding the use of AI in decision-making include:

1. Bias and Fairness: AI systems can inherit and amplify biases present in the data they are trained on, leading to unfair or discriminatory outcomes.
2. Transparency and Explainability: Many AI systems, particularly deep learning models, are "black boxes," making it difficult to understand how they arrive at their decisions.
3. Privacy and Security: The reliance on large amounts of data raises concerns about data security and the potential for unauthorized access or misuse.
4. Job Displacement: The potential for AI systems to automate repetitive or routine tasks raises concerns about job displacement and the impact on workers.
5. Control, Accountability, and Unintended Consequences: As AI systems become more autonomous, there are concerns about who is responsible for their actions, and the potential for unintended consequences.
6. Need for Clear Guidelines and Ethical Frameworks: There is a need for clear guidelines and ethical frameworks to ensure that AI systems are developed and deployed in a responsible and ethical manner.

These concerns highlight the importance of addressing the ethical implications of AI in decision-making, and the need for a balanced approach that promotes innovation while protecting human rights, privacy, and well-being.

Testing extraction compression...

=== RAG WITH CONTEXTUAL COMPRESSION ===
Query: What are the ethical concerns surrounding the use of AI in decision-making?
Compression type: extraction
Extracting text from PDF...
Chunking text...
Created 42 text chunks
Creating embeddings for chunks...
Added 42 chunks to the vector store
Retrieving top 10 chunks...
Compressing 10 chunks...
Compressing chunk 1/10...
Compressing chunk 2/10...
Compressing chunk 3/10...
Compressing chunk 4/10...
Compressing chunk 5/10...
Compressing chunk 6/10...
Compressing chunk 7/10...
Compressing chunk 8/10...
Compressing chunk 9/10...
Compressing chunk 10/10...
Overall compression ratio: 54.41%
Generating response based on compressed chunks...

=== RESPONSE ===
The ethical concerns surrounding the use of AI in decision-making include:

1. Bias and Fairness: AI systems can inherit and amplify biases present in the data they are trained on, leading to unfair or discriminatory outcomes. Ensuring fairness and mitigating bias in AI systems is a critical challenge.

2. Lack of Transparency and Explainability: Many AI systems, particularly deep learning models, are "black boxes," making it difficult to understand how they arrive at their decisions. This lack of transparency and explainability can lead to a lack of trust in AI systems.

3. Accountability and Responsibility: Establishing accountability and responsibility for AI systems is essential for addressing potential harms and ensuring ethical behavior. This includes defining roles and responsibilities for developers, deployers, and users of AI systems.

4. Respect for Human Rights, Privacy, and Non-Discrimination: AI systems must be designed and deployed in a way that respects human rights, protects privacy, and avoids non-discrimination.

5. Beneficence: AI systems should be designed and deployed in a way that promotes the well-being and benefit of society.

6. Addressing Bias in Data Collection, Algorithm Design, and Ongoing Monitoring and Evaluation: Addressing bias requires careful data collection, algorithm design, and ongoing monitoring and evaluation.

7. Ensuring Robustness and Reliability: Ensuring that AI systems are robust and reliable is essential for building trust. This includes testing and validating AI models, monitoring their performance, and addressing potential vulnerabilities.

8. Empowering Users with Control: Empowering users with control over AI systems and providing them with agency in their interactions with AI enhances trust. This includes allowing users to customize AI settings, understand how their data is used, and opt out of AI-driven features.

9. International Discussions and Regulations: The potential use of AI in autonomous weapons systems raises significant ethical and security concerns, and international discussions and regulations are needed to address the risks associated with AI-powered weapons.

10. Public Perception and Trust: Public perception and trust in AI are essential for its widespread adoption and positive social impact.

=== EVALUATION RESULTS ===
Based on the evaluation criteria, here are the rankings from best to worst:

1. **Reference Answer**: This response is the most accurate, comprehensive, and relevant to the query. It provides a clear and concise overview of the ethical concerns surrounding the use of AI in decision-making, covering topics such as bias, transparency, privacy, accountability, and job displacement. The response is well-structured and easy to follow, making it an excellent example of a high-quality response.

2. **Standard Response**: This response is comprehensive and covers all the key ethical concerns surrounding AI in decision-making. It provides a clear and concise overview of the issues, including bias, transparency, privacy, accountability, and job displacement. The response is well-organized and easy to follow, making it a strong contender for the top spot.

3. **Selective Response**: This response is comprehensive, but it lacks some of the key points mentioned in the reference answer. It covers bias, transparency, privacy, and accountability, but misses out on job displacement and control. The response is still well-organized and easy to follow, but it falls short of the reference answer in terms of comprehensiveness.

4. **Summary Response**: This response is concise and covers the key points, but it lacks some of the detail and depth of the reference answer. It provides a good overview of the ethical concerns, but it doesn't delve as deeply into the issues as the reference answer. The response is still clear and easy to follow, but it falls short of the standard response in terms of comprehensiveness.

5. **Extraction Response**: This response is concise and covers the key points, but it lacks some of the detail and depth of the reference answer. It provides a good overview of the ethical concerns, but it doesn't delve as deeply into the issues as the reference answer. The response is still clear and easy to follow, but it falls short of the standard response in terms of comprehensiveness.

Ranking Criteria:

* Factual accuracy: Reference Answer (9/10), Standard Response (8.5/10), Selective Response (8/10), Summary Response (7.5/10), Extraction Response (7/10)
* Comprehensiveness: Reference Answer (9/10), Standard Response (8.5/10), Selective Response (8/10), Summary Response (7.5/10), Extraction Response (7/10)
* Conciseness: Summary Response (8/10), Extraction Response (7.5/10), Selective Response (7/10), Standard Response (6.5/10), Reference Answer (6/10)
* Overall quality: Reference Answer (9/10), Standard Response (8.5/10), Selective Response (8/10), Summary Response (7.5/10), Extraction Response (7/10)

Note: The scores are subjective and based on the evaluation criteria. They are intended to provide a general ranking of the responses rather than a precise numerical score.
```

## Visualizing Compression Results

```python
def visualize_compression_results(evaluation_results):
    """
    Visualize the results of different compression techniques.

    Args:
        evaluation_results (Dict): Results from evaluate_compression function
    """

    query = evaluation_results["query"]
    standard_chunks = evaluation_results["standard_result"]["chunks"]


    print(f"Query: {query}")
    print("\n" + "="*80 + "\n")


    original_chunk = standard_chunks[0]


    for comp_type in evaluation_results["compression_results"].keys():
        compressed_chunks = evaluation_results["compression_results"][comp_type]["compressed_chunks"]
        compression_ratios = evaluation_results["compression_results"][comp_type]["compression_ratios"]


        compressed_chunk = compressed_chunks[0]
        compression_ratio = compression_ratios[0]

        print(f"\n=== {comp_type.upper()} COMPRESSION EXAMPLE ===\n")


        print("ORIGINAL CHUNK:")
        print("-" * 40)
        if len(original_chunk) > 800:
            print(original_chunk[:800] + "... [truncated]")
        else:
            print(original_chunk)
        print("-" * 40)
        print(f"Length: {len(original_chunk)} characters\n")


        print("COMPRESSED CHUNK:")
        print("-" * 40)
        print(compressed_chunk)
        print("-" * 40)
        print(f"Length: {len(compressed_chunk)} characters")
        print(f"Compression ratio: {compression_ratio:.2f}%\n")


        avg_ratio = sum(compression_ratios) / len(compression_ratios)
        print(f"Average compression across all chunks: {avg_ratio:.2f}%")
        print(f"Total context length reduction: {evaluation_results['metrics'][comp_type]['avg_compression_ratio']}")
        print("=" * 80)


    print("\n=== COMPRESSION SUMMARY ===\n")
    print(f"{'Technique':<15} {'Avg Ratio':<15} {'Context Length':<15} {'Original Length':<15}")
    print("-" * 60)


    for comp_type, metrics in evaluation_results["metrics"].items():
        print(f"{comp_type:<15} {metrics['avg_compression_ratio']:<15} {metrics['total_context_length']:<15} {metrics['original_context_length']:<15}")
```

```python

visualize_compression_results(results)
```

```output
Query: What are the ethical concerns surrounding the use of AI in decision-making?

================================================================================


=== SELECTIVE COMPRESSION EXAMPLE ===

ORIGINAL CHUNK:
----------------------------------------
inability
Many AI systems, particularly deep learning models, are "black boxes," making it difficult to
understand how they arrive at their decisions. Enhancing transparency and explainability is
crucial for building trust and accountability.


Privacy and Security
AI systems often rely on large amounts of data, raising concerns about privacy and data security.
Protecting sensitive information and ensuring responsible data handling are essential.
Job Displacement
The automation capabilities of AI have raised concerns about job displacement, particularly in
industries with repetitive or routine tasks. Addressing the potential economic and social impacts
of AI-driven automation is a key challenge.
Autonomy and Control
As AI systems become more autonomous, questions arise about ... [truncated]
----------------------------------------
Length: 1000 characters

COMPRESSED CHUNK:
----------------------------------------
Many AI systems, particularly deep learning models, are "black boxes," making it difficult to
understand how they arrive at their decisions. Enhancing transparency and explainability is
crucial for building trust and accountability.

Establishing clear guidelines and ethical frameworks for AI development and deployment is crucial.

Protecting sensitive information and ensuring responsible data handling are essential.

Addressing the potential economic and social impacts of AI-driven automation is a key challenge.

As AI systems become more autonomous, questions arise about control, accountability, and the
potential for unintended consequences.
----------------------------------------
Length: 654 characters
Compression ratio: 34.60%

Average compression across all chunks: 39.93%
Total context length reduction: 39.93%
================================================================================

=== SUMMARY COMPRESSION EXAMPLE ===

ORIGINAL CHUNK:
----------------------------------------
inability
Many AI systems, particularly deep learning models, are "black boxes," making it difficult to
understand how they arrive at their decisions. Enhancing transparency and explainability is
crucial for building trust and accountability.


Privacy and Security
AI systems often rely on large amounts of data, raising concerns about privacy and data security.
Protecting sensitive information and ensuring responsible data handling are essential.
Job Displacement
The automation capabilities of AI have raised concerns about job displacement, particularly in
industries with repetitive or routine tasks. Addressing the potential economic and social impacts
of AI-driven automation is a key challenge.
Autonomy and Control
As AI systems become more autonomous, questions arise about ... [truncated]
----------------------------------------
Length: 1000 characters

COMPRESSED CHUNK:
----------------------------------------
The ethical concerns surrounding the use of AI in decision-making include:

- Lack of transparency and explainability in AI decision-making processes
- Privacy and data security concerns due to reliance on large amounts of data
- Potential for job displacement, particularly in industries with repetitive or routine tasks
- Questions about control, accountability, and unintended consequences as AI systems become more autonomous
- Need for clear guidelines and ethical frameworks for AI development and deployment
----------------------------------------
Length: 514 characters
Compression ratio: 48.60%

Average compression across all chunks: 63.87%
Total context length reduction: 63.87%
================================================================================

=== EXTRACTION COMPRESSION EXAMPLE ===

ORIGINAL CHUNK:
----------------------------------------
inability
Many AI systems, particularly deep learning models, are "black boxes," making it difficult to
understand how they arrive at their decisions. Enhancing transparency and explainability is
crucial for building trust and accountability.


Privacy and Security
AI systems often rely on large amounts of data, raising concerns about privacy and data security.
Protecting sensitive information and ensuring responsible data handling are essential.
Job Displacement
The automation capabilities of AI have raised concerns about job displacement, particularly in
industries with repetitive or routine tasks. Addressing the potential economic and social impacts
of AI-driven automation is a key challenge.
Autonomy and Control
As AI systems become more autonomous, questions arise about ... [truncated]
----------------------------------------
Length: 1000 characters

COMPRESSED CHUNK:
----------------------------------------
Many AI systems, particularly deep learning models, are "black boxes," making it difficult to
understand how they arrive at their decisions. Enhancing transparency and explainability is
crucial for building trust and accountability.

Establishing clear guidelines and ethical frameworks for AI development and deployment is crucial.
----------------------------------------
Length: 335 characters
Compression ratio: 66.50%

Average compression across all chunks: 54.41%
Total context length reduction: 54.41%
================================================================================

=== COMPRESSION SUMMARY ===

Technique       Avg Ratio       Context Length  Original Length
------------------------------------------------------------
selective       39.93%          6025            10018
summary         63.87%          3631            10018
extraction      54.41%          4577            10018
```
