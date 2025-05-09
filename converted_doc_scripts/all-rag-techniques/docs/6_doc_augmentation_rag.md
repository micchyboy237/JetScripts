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

```python
import fitz
import os
import numpy as np
import json
from openai import OpenAI
import re
from tqdm import tqdm
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

## Generating Questions for Text Chunks
This is the key enhancement over simple RAG. We generate questions that could be answered by each text chunk.

```python
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
```

## Creating Embeddings for Text
We generate embeddings for both text chunks and generated questions.

```python
def create_embeddings(text, model="BAAI/bge-en-icl"):
    """
    Creates embeddings for the given text using the specified OpenAI model.

    Args:
    text (str): The input text for which embeddings are to be created.
    model (str): The model to be used for creating embeddings.

    Returns:
    dict: The response from the OpenAI API containing the embeddings.
    """

    response = client.embeddings.create(
        model=model,
        input=text
    )

    return response
```

## Building a Simple Vector Store
We'll implement a simple vector store using NumPy.

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

## Processing Documents with Question Augmentation
Now we'll put everything together to process documents, generate questions, and build our augmented vector store.

```python
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
    print("Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_path)

    print("Chunking text...")
    text_chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)
    print(f"Created {len(text_chunks)} text chunks")

    vector_store = SimpleVectorStore()

    print("Processing chunks and generating questions...")
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
```

## Extracting and Processing the Document

```python

pdf_path = "data/AI_Information.pdf"


text_chunks, vector_store = process_document(
    pdf_path,
    chunk_size=1000,
    chunk_overlap=200,
    questions_per_chunk=3
)

print(f"Vector store contains {len(vector_store.texts)} items")
```

```output
Extracting text from PDF...
Chunking text...
Created 42 text chunks
Processing chunks and generating questions...
```

```output
Processing Chunks: 100%|██████████| 42/42 [01:30<00:00,  2.15s/it]
```

```output
Vector store contains 165 items
```

```output

```

## Performing Semantic Search
We implement a semantic search function similar to the simple RAG implementation but adapted to our augmented vector store.

```python
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
```

## Running a Query on the Augmented Vector Store

```python

with open('data/val.json') as f:
    data = json.load(f)


query = data[0]['question']


search_results = semantic_search(query, vector_store, k=5)

print("Query:", query)
print("\nSearch Results:")


chunk_results = []
question_results = []

for result in search_results:
    if result["metadata"]["type"] == "chunk":
        chunk_results.append(result)
    else:
        question_results.append(result)


print("\nRelevant Document Chunks:")
for i, result in enumerate(chunk_results):
    print(f"Context {i + 1} (similarity: {result['similarity']:.4f}):")
    print(result["text"][:300] + "...")
    print("=====================================")


print("\nMatched Questions:")
for i, result in enumerate(question_results):
    print(f"Question {i + 1} (similarity: {result['similarity']:.4f}):")
    print(result["text"])
    chunk_idx = result["metadata"]["chunk_index"]
    print(f"From chunk {chunk_idx}")
    print("=====================================")
```

```output
Query: What is 'Explainable AI' and why is it considered important?

Search Results:

Relevant Document Chunks:

Matched Questions:
Question 1 (similarity: 0.8629):
What is the main goal of Explainable AI (XAI)?
From chunk 10
=====================================
Question 2 (similarity: 0.8488):
What is the primary goal of Explainable AI (XAI) techniques?
From chunk 37
=====================================
Question 3 (similarity: 0.8414):
What is the focus of research on Explainable AI (XAI)?
From chunk 29
=====================================
Question 4 (similarity: 0.7995):
Why are transparency and explainability essential for building trust in AI systems?
From chunk 36
=====================================
Question 5 (similarity: 0.7841):
Why is transparency and explainability essential in building trust and accountability with AI systems?
From chunk 9
=====================================
```

## Generating Context for Response
Now we prepare the context by combining information from relevant chunks and questions.

```python
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
```

## Generating a Response Based on Retrieved Chunks

```python
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
```

## Generating and Displaying the Response

```python

context = prepare_context(search_results)


response_text = generate_response(query, context)

print("\nQuery:", query)
print("\nResponse:")
print(response_text)
```

```output

Query: What is 'Explainable AI' and why is it considered important?

Response:
Explainable AI (XAI) is a field that aims to make AI systems more transparent and understandable by providing insights into how AI models make decisions. This is essential for building trust and accountability in AI systems, as it enables users to assess their fairness and accuracy. XAI techniques are crucial for addressing potential harms, ensuring ethical behavior, and establishing clear guidelines and ethical frameworks for AI development and deployment.
```

## Evaluating the AI Response
We compare the AI response with the expected answer and assign a score.

```python
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
```

## Running the Evaluation

```python

reference_answer = data[0]['ideal_answer']


evaluation = evaluate_response(query, response_text, reference_answer)

print("\nEvaluation:")
print(evaluation)
```

```output

Evaluation:
Based on the evaluation criteria, I will assess the AI response as follows:

1. Factual correctness: The AI response contains accurate information about Explainable AI (XAI) and its importance. It correctly states that XAI aims to make AI systems more transparent and understandable, providing insights into how they make decisions.

2. Completeness: The AI response covers the main points of XAI, including its importance for building trust, accountability, and ensuring fairness in AI systems. However, it misses some details, such as the potential harms that XAI can address and the need for clear guidelines and ethical frameworks for AI development and deployment.

3. Relevance: The AI response directly addresses the question, providing a clear and concise explanation of XAI and its significance.

Based on the evaluation, I would assign a score of 0.8 to the AI response. The response is very good, with minor omissions and differences from the reference answer. It covers the main points of XAI and its importance, but misses some details that are present in the reference answer.
```

## Extracting and Chunking Text from a PDF File
Now, we load the PDF, extract text, and split it into chunks.

```python

pdf_path = "data/AI_Information.pdf"


extracted_text = extract_text_from_pdf(pdf_path)


text_chunks = chunk_text(extracted_text, 1000, 200)


print("Number of text chunks:", len(text_chunks))


print("\nFirst text chunk:")
print(text_chunks[0])
```

```output
Number of text chunks: 42

First text chunk:
Understanding Artificial Intelligence
Chapter 1: Introduction to Artificial Intelligence
Artificial intelligence (AI) refers to the ability of a digital computer or computer-controlled robot
to perform tasks commonly associated with intelligent beings. The term is frequently applied to
the project of developing systems endowed with the intellectual processes characteristic of
humans, such as the ability to reason, discover meaning, generalize, or learn from past
experience. Over the past few decades, advancements in computing power and data availability
have significantly accelerated the development and deployment of AI.
Historical Context
The idea of artificial intelligence has existed for centuries, often depicted in myths and fiction.
However, the formal field of AI research began in the mid-20th century. The Dartmouth Workshop
in 1956 is widely considered the birthplace of AI. Early AI research focused on problem-solving
and symbolic methods. The 1980s saw a rise in exp
```

## Creating Embeddings for Text Chunks
Embeddings transform text into numerical vectors, which allow for efficient similarity search.

```python
def create_embeddings(text, model="BAAI/bge-en-icl"):
    """
    Creates embeddings for the given text using the specified OpenAI model.

    Args:
    text (str): The input text for which embeddings are to be created.
    model (str): The model to be used for creating embeddings. Default is "BAAI/bge-en-icl".

    Returns:
    dict: The response from the OpenAI API containing the embeddings.
    """

    response = client.embeddings.create(
        model=model,
        input=text
    )

    return response


response = create_embeddings(text_chunks)
```

## Performing Semantic Search
We implement cosine similarity to find the most relevant text chunks for a user query.

```python
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
```

```python
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
    similarity_scores = []


    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(np.array(query_embedding), np.array(chunk_embedding.embedding))
        similarity_scores.append((i, similarity_score))


    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    top_indices = [index for index, _ in similarity_scores[:k]]

    return [text_chunks[index] for index in top_indices]
```

## Running a Query on Extracted Chunks

```python

with open('data/val.json') as f:
    data = json.load(f)


query = data[0]['question']


top_chunks = semantic_search(query, text_chunks, response.data, k=2)


print("Query:", query)


for i, chunk in enumerate(top_chunks):
    print(f"Context {i + 1}:\n{chunk}\n=====================================")
```

```output
Query: What is 'Explainable AI' and why is it considered important?
Context 1:
systems. Explainable AI (XAI)
techniques aim to make AI decisions more understandable, enabling users to assess their
fairness and accuracy.
Privacy and Data Protection
AI systems often rely on large amounts of data, raising concerns about privacy and data
protection. Ensuring responsible data handling, implementing privacy-preserving techniques,
and complying with data protection regulations are crucial.
Accountability and Responsibility
Establishing accountability and responsibility for AI systems is essential for addressing potential
harms and ensuring ethical behavior. This includes defining roles and responsibilities for
developers, deployers, and users of AI systems.
Chapter 20: Building Trust in AI
Transparency and Explainability
Transparency and explainability are key to building trust in AI. Making AI systems understandable
and providing insights into their decision-making processes helps users assess their reliability
and fairness.
Robustness and Reliability

=====================================
Context 2:
 incidents.
Environmental Monitoring
AI-powered environmental monitoring systems track air and water quality, detect pollution, and
support environmental protection efforts. These systems provide real-time data, identify
pollution sources, and inform environmental policies.
Chapter 15: The Future of AI Research
Advancements in Deep Learning
Continued advancements in deep learning are expected to drive further breakthroughs in AI.
Research is focused on developing more efficient and interpretable deep learning models, as well
as exploring new architectures and training techniques.
Explainable AI (XAI)
Explainable AI (XAI) aims to make AI systems more transparent and understandable. Research in
XAI focuses on developing methods for explaining AI decisions, enhancing trust, and improving
accountability.
AI and Neuroscience
The intersection of AI and neuroscience is a promising area of research. Understanding the
human brain can inspire new AI algorithms and architectures,
=====================================
```

## Generating a Response Based on Retrieved Chunks

```python

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
```

## Evaluating the AI Response
We compare the AI response with the expected answer and assign a score.

```python

evaluate_system_prompt = "You are an intelligent evaluation system tasked with assessing the AI assistant's responses. If the AI assistant's response is very close to the true response, assign a score of 1. If the response is incorrect or unsatisfactory in relation to the true response, assign a score of 0. If the response is partially aligned with the true response, assign a score of 0.5."


evaluation_prompt = f"User Query: {query}\nAI Response:\n{ai_response.choices[0].message.content}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"


evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)


print(evaluation_response.choices[0].message.content)
```

```output
Based on the evaluation criteria, I would assign a score of 0.8 to the AI assistant's response.

The response is very close to the true response, and it correctly conveys the main idea of Explainable AI (XAI) and its importance. The AI assistant has successfully identified the primary goal of XAI, its significance in building trust and accountability, and its relevance to areas such as privacy and data protection.

However, the response could be improved by providing more specific details and examples to support the claims made. For instance, the AI assistant could have elaborated on the techniques used in XAI, such as model interpretability, feature attribution, and explainability metrics. Additionally, the response could have provided more concrete examples of how XAI is being applied in various fields, such as healthcare and finance.

Overall, the response is a good start, but it could benefit from more depth and specificity to make it more accurate and informative.
```
