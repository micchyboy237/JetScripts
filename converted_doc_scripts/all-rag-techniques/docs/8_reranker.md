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

    input_text = text if isinstance(text, list) else [text]


    response = client.embeddings.create(
        model=model,
        input=input_text
    )


    if isinstance(text, str):
        return response.data[0].embedding


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
    print(f"Reranking {len(results)} documents...")

    scored_results = []


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
            print(f"Scoring document {i+1}/{len(results)}...")


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

            print(f"Warning: Could not extract score from response: '{score_text}', using similarity score instead")
            score = result["similarity"] * 10


        scored_results.append({
            "text": result["text"],
            "metadata": result["metadata"],
            "similarity": result["similarity"],
            "relevance_score": score
        })


    reranked_results = sorted(scored_results, key=lambda x: x["relevance_score"], reverse=True)


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

    keywords = [word.lower() for word in query.split() if len(word) > 3]

    scored_results = []

    for result in results:
        document_text = result["text"].lower()


        base_score = result["similarity"] * 0.5


        keyword_score = 0
        for keyword in keywords:
            if keyword in document_text:

                keyword_score += 0.1


                first_position = document_text.find(keyword)
                if first_position < len(document_text) / 4:
                    keyword_score += 0.1


                frequency = document_text.count(keyword)
                keyword_score += min(0.05 * frequency, 0.2)


        final_score = base_score + keyword_score


        scored_results.append({
            "text": result["text"],
            "metadata": result["metadata"],
            "similarity": result["similarity"],
            "relevance_score": final_score
        })


    reranked_results = sorted(scored_results, key=lambda x: x["relevance_score"], reverse=True)


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
```

## Evaluating Reranking Quality

```python

with open('data/val.json') as f:
    data = json.load(f)


query = data[0]['question']


reference_answer = data[0]['ideal_answer']


pdf_path = "data/AI_Information.pdf"
```

```python

vector_store = process_document(pdf_path)


query = "Does AI have the potential to transform the way we live and work?"


print("Comparing retrieval methods...")


print("\n=== STANDARD RETRIEVAL ===")
standard_results = rag_with_reranking(query, vector_store, reranking_method="none")
print(f"\nQuery: {query}")
print(f"\nResponse:\n{standard_results['response']}")


print("\n=== LLM-BASED RERANKING ===")
llm_results = rag_with_reranking(query, vector_store, reranking_method="llm")
print(f"\nQuery: {query}")
print(f"\nResponse:\n{llm_results['response']}")


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
```

```python

evaluation = evaluate_reranking(
    query=query,
    standard_results=standard_results,
    reranked_results=llm_results,
    reference_answer=reference_answer
)


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
