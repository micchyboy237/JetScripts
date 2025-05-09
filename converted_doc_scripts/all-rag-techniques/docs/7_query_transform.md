# Query Transformations for Enhanced RAG Systems

This notebook implements three query transformation techniques to enhance retrieval performance in RAG systems without relying on specialized libraries like LangChain. By modifying user queries, we can significantly improve the relevance and comprehensiveness of retrieved information.

## Key Transformation Techniques

1. **Query Rewriting**: Makes queries more specific and detailed for better search precision.
2. **Step-back Prompting**: Generates broader queries to retrieve useful contextual information.
3. **Sub-query Decomposition**: Breaks complex queries into simpler components for comprehensive retrieval.

## Setting Up the Environment
We begin by importing necessary libraries.

```python
import fitz
import os
import numpy as np
import json
from openai import OpenAI
```

## Setting Up the OpenAI API Client
We initialize the OpenAI client to generate embeddings and responses.

```python

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")
)
```

## Implementing Query Transformation Techniques
### 1. Query Rewriting
This technique makes queries more specific and detailed to improve precision in retrieval.

```python
def rewrite_query(original_query, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Rewrites a query to make it more specific and detailed for better retrieval.

    Args:
        original_query (str): The original user query
        model (str): The model to use for query rewriting

    Returns:
        str: The rewritten query
    """

    system_prompt = "You are an AI assistant specialized in improving search queries. Your task is to rewrite user queries to be more specific, detailed, and likely to retrieve relevant information."


    user_prompt = f"""
    Rewrite the following query to make it more specific and detailed. Include relevant terms and concepts that might help in retrieving accurate information.

    Original query: {original_query}

    Rewritten query:
    """


    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )


    return response.choices[0].message.content.strip()
```

### 2. Step-back Prompting
This technique generates broader queries to retrieve contextual background information.

```python
def generate_step_back_query(original_query, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Generates a more general 'step-back' query to retrieve broader context.

    Args:
        original_query (str): The original user query
        model (str): The model to use for step-back query generation

    Returns:
        str: The step-back query
    """

    system_prompt = "You are an AI assistant specialized in search strategies. Your task is to generate broader, more general versions of specific queries to retrieve relevant background information."


    user_prompt = f"""
    Generate a broader, more general version of the following query that could help retrieve useful background information.

    Original query: {original_query}

    Step-back query:
    """


    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )


    return response.choices[0].message.content.strip()
```

### 3. Sub-query Decomposition
This technique breaks down complex queries into simpler components for comprehensive retrieval.

```python
def decompose_query(original_query, num_subqueries=4, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Decomposes a complex query into simpler sub-queries.

    Args:
        original_query (str): The original complex query
        num_subqueries (int): Number of sub-queries to generate
        model (str): The model to use for query decomposition

    Returns:
        List[str]: A list of simpler sub-queries
    """

    system_prompt = "You are an AI assistant specialized in breaking down complex questions. Your task is to decompose complex queries into simpler sub-questions that, when answered together, address the original query."


    user_prompt = f"""
    Break down the following complex query into {num_subqueries} simpler sub-queries. Each sub-query should focus on a different aspect of the original question.

    Original query: {original_query}

    Generate {num_subqueries} sub-queries, one per line, in this format:
    1. [First sub-query]
    2. [Second sub-query]
    And so on...
    """


    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )


    content = response.choices[0].message.content.strip()


    lines = content.split("\n")
    sub_queries = []

    for line in lines:
        if line.strip() and any(line.strip().startswith(f"{i}.") for i in range(1, 10)):

            query = line.strip()
            query = query[query.find(".")+1:].strip()
            sub_queries.append(query)

    return sub_queries
```

## Demonstrating Query Transformation Techniques
Let's apply these techniques to an example query.

```python

original_query = "What are the impacts of AI on job automation and employment?"


print("Original Query:", original_query)


rewritten_query = rewrite_query(original_query)
print("\n1. Rewritten Query:")
print(rewritten_query)


step_back_query = generate_step_back_query(original_query)
print("\n2. Step-back Query:")
print(step_back_query)


sub_queries = decompose_query(original_query, num_subqueries=4)
print("\n3. Sub-queries:")
for i, query in enumerate(sub_queries, 1):
    print(f"   {i}. {query}")
```

```output
Original Query: What are the impacts of AI on job automation and employment?

1. Rewritten Query:
Here's a rewritten query:

"What are the current and projected impacts of artificial intelligence (AI) on job automation and employment, including the types of jobs most susceptible to automation, the skills required to remain employable in an AI-driven economy, and the potential effects on unemployment rates, social welfare systems, and the gig economy?"

This rewritten query is more specific and detailed because it:

1. Includes the term "artificial intelligence" to specify the type of technology being discussed.
2. Mentions "current and projected impacts" to indicate that the query is looking for both immediate and long-term effects.
3. Specifies the types of jobs most susceptible to automation, which can help retrieve information on the most affected industries and occupations.
4. Includes the skills required to remain employable in an AI-driven economy, which can help retrieve information on upskilling and reskilling programs.
5. Covers the potential effects on unemployment rates, social welfare systems, and the gig economy, which can help retrieve information on the broader societal implications of AI on employment.

By including these specific terms and concepts, the rewritten query is more likely to retrieve accurate and relevant information on the topic.

2. Step-back Query:
Here's a broader, more general version of the original query:

"Effects of automation and artificial intelligence on the modern workforce and labor market, including trends, challenges, and potential implications for employment and economic growth."

This step-back query can help retrieve relevant background information on the topic, including:

* Historical context of automation and AI on employment
* Current trends and statistics on job automation and AI adoption
* Expert opinions and research on the impact of AI on the workforce
* Potential solutions and strategies for mitigating the negative effects of AI on employment
* Broader societal implications of AI on the economy and labor market.

3. Sub-queries:
   1. What are the primary job roles that are most susceptible to automation by AI?
   2. How does AI-driven automation affect the overall job market, including the creation of new job opportunities?
   3. What are the potential consequences of widespread AI-driven job automation on employment rates and workforce demographics?
   4. How do governments, industries, and individuals respond to and mitigate the impacts of AI on job automation and employment?
```

## Building a Simple Vector Store
To demonstrate how query transformations integrate with retrieval, let's implement a simple vector store.

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

## Implementing RAG with Query Transformations

```python
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

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

## RAG with Query Transformations

```python
def transformed_search(query, vector_store, transformation_type, top_k=3):
    """
    Search using a transformed query.

    Args:
        query (str): Original query
        vector_store (SimpleVectorStore): Vector store to search
        transformation_type (str): Type of transformation ('rewrite', 'step_back', or 'decompose')
        top_k (int): Number of results to return

    Returns:
        List[Dict]: Search results
    """
    print(f"Transformation type: {transformation_type}")
    print(f"Original query: {query}")

    results = []

    if transformation_type == "rewrite":

        transformed_query = rewrite_query(query)
        print(f"Rewritten query: {transformed_query}")


        query_embedding = create_embeddings(transformed_query)


        results = vector_store.similarity_search(query_embedding, k=top_k)

    elif transformation_type == "step_back":

        transformed_query = generate_step_back_query(query)
        print(f"Step-back query: {transformed_query}")


        query_embedding = create_embeddings(transformed_query)


        results = vector_store.similarity_search(query_embedding, k=top_k)

    elif transformation_type == "decompose":

        sub_queries = decompose_query(query)
        print("Decomposed into sub-queries:")
        for i, sub_q in enumerate(sub_queries, 1):
            print(f"{i}. {sub_q}")


        sub_query_embeddings = create_embeddings(sub_queries)


        all_results = []
        for i, embedding in enumerate(sub_query_embeddings):
            sub_results = vector_store.similarity_search(embedding, k=2)
            all_results.extend(sub_results)


        seen_texts = {}
        for result in all_results:
            text = result["text"]
            if text not in seen_texts or result["similarity"] > seen_texts[text]["similarity"]:
                seen_texts[text] = result


        results = sorted(seen_texts.values(), key=lambda x: x["similarity"], reverse=True)[:top_k]

    else:

        query_embedding = create_embeddings(query)
        results = vector_store.similarity_search(query_embedding, k=top_k)

    return results
```

## Generating a Response with Transformed Queries

```python
def generate_response(query, context, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Generates a response based on the query and retrieved context.

    Args:
        query (str): User query
        context (str): Retrieved context
        model (str): The model to use for response generation

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


    return response.choices[0].message.content.strip()
```

## Running the Complete RAG Pipeline with Query Transformations

```python
def rag_with_query_transformation(pdf_path, query, transformation_type=None):
    """
    Run complete RAG pipeline with optional query transformation.

    Args:
        pdf_path (str): Path to PDF document
        query (str): User query
        transformation_type (str): Type of transformation (None, 'rewrite', 'step_back', or 'decompose')

    Returns:
        Dict: Results including query, transformed query, context, and response
    """

    vector_store = process_document(pdf_path)


    if transformation_type:

        results = transformed_search(query, vector_store, transformation_type)
    else:

        query_embedding = create_embeddings(query)
        results = vector_store.similarity_search(query_embedding, k=3)


    context = "\n\n".join([f"PASSAGE {i+1}:\n{result['text']}" for i, result in enumerate(results)])


    response = generate_response(query, context)


    return {
        "original_query": query,
        "transformation_type": transformation_type,
        "context": context,
        "response": response
    }
```

## Evaluating Transformation Techniques

```python

def compare_responses(results, reference_answer, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Compare responses from different query transformation techniques.

    Args:
        results (Dict): Results from different transformation techniques
        reference_answer (str): Reference answer for comparison
        model (str): Model for evaluation
    """

    system_prompt = """You are an expert evaluator of RAG systems.
    Your task is to compare different responses generated using various query transformation techniques
    and determine which technique produced the best response compared to the reference answer."""


    comparison_text = f"""Reference Answer: {reference_answer}\n\n"""

    for technique, result in results.items():
        comparison_text += f"{technique.capitalize()} Query Response:\n{result['response']}\n\n"


    user_prompt = f"""
    {comparison_text}

    Compare the responses generated by different query transformation techniques to the reference answer.

    For each technique (original, rewrite, step_back, decompose):
    1. Score the response from 1-10 based on accuracy, completeness, and relevance
    2. Identify strengths and weaknesses

    Then rank the techniques from best to worst and explain which technique performed best overall and why.
    """


    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )


    print("\n===== EVALUATION RESULTS =====")
    print(response.choices[0].message.content)
    print("=============================")
```

```python
def evaluate_transformations(pdf_path, query, reference_answer=None):
    """
    Evaluate different transformation techniques for the same query.

    Args:
        pdf_path (str): Path to PDF document
        query (str): Query to evaluate
        reference_answer (str): Optional reference answer for comparison

    Returns:
        Dict: Evaluation results
    """

    transformation_types = [None, "rewrite", "step_back", "decompose"]
    results = {}


    for transformation_type in transformation_types:
        type_name = transformation_type if transformation_type else "original"
        print(f"\n===== Running RAG with {type_name} query =====")


        result = rag_with_query_transformation(pdf_path, query, transformation_type)
        results[type_name] = result


        print(f"Response with {type_name} query:")
        print(result["response"])
        print("=" * 50)


    if reference_answer:
        compare_responses(results, reference_answer)

    return results
```

## Evaluation of Query Transformations

```python

with open('data/val.json') as f:
    data = json.load(f)


query = data[0]['question']


reference_answer = data[0]['ideal_answer']


pdf_path = "data/AI_Information.pdf"


evaluation_results = evaluate_transformations(pdf_path, query, reference_answer)
```

```output

===== Running RAG with original query =====
Extracting text from PDF...
Chunking text...
Created 42 text chunks
Creating embeddings for chunks...
Added 42 chunks to the vector store
Response with original query:
Explainable AI (XAI) refers to techniques used to make AI decisions more understandable and transparent. The primary goal of XAI is to enable users to assess the fairness and accuracy of AI systems. This is achieved by providing insights into how AI models make decisions, thereby enhancing trust and accountability in AI systems.

XAI is considered important for several reasons. Firstly, it addresses concerns about the fairness and accuracy of AI decisions, which is crucial for ensuring that AI systems are reliable and trustworthy. Secondly, XAI helps to establish accountability and responsibility for AI systems, which is essential for addressing potential harms and ensuring ethical behavior.

Furthermore, XAI is critical in the context of AI development and deployment, as it helps to mitigate the risks associated with AI systems, such as unintended consequences and potential misuse. By making AI systems more transparent and understandable, XAI can help to build trust in AI and ensure that it is developed and deployed in a responsible and ethical manner.

In the context of various domains, including environmental monitoring, AI-powered systems can be made more transparent and understandable through XAI techniques, which can provide insights into how these systems make decisions and support environmental protection efforts. Similarly, in the development of autonomous weapons systems, XAI can help to address the ethical and security concerns associated with AI-powered weapons.

Overall, Explainable AI is a crucial area of research that aims to make AI systems more transparent, trustworthy, and accountable. Its importance lies in its ability to enhance trust, accountability, and responsibility in AI systems, while mitigating the risks associated with AI development and deployment.
==================================================

===== Running RAG with rewrite query =====
Extracting text from PDF...
Chunking text...
Created 42 text chunks
Creating embeddings for chunks...
Added 42 chunks to the vector store
Transformation type: rewrite
Original query: What is 'Explainable AI' and why is it considered important?
Rewritten query: Here's a rewritten query:

"What is Explainable AI (XAI) and its significance in the context of machine learning, artificial intelligence, and data science, including its applications, benefits, and limitations, as well as the current state of research and development in this field?"

This rewritten query is more specific and detailed because it:

1. Includes the term "Explainable AI" (XAI) to ensure the search results are focused on the specific concept.
2. Provides context by mentioning machine learning, artificial intelligence, and data science, which are relevant fields that XAI is associated with.
3. Specifies the applications, benefits, and limitations of XAI, which will help retrieve information that addresses the user's specific interests.
4. Includes the current state of research and development in XAI, which will provide a more comprehensive understanding of the field.
5. Uses relevant terms and concepts, such as "XAI," "machine learning," "artificial intelligence," and "data science," to help the search engine understand the user's intent and retrieve more accurate results.

By rewriting the query in this way, the user is more likely to retrieve information that is relevant, accurate, and up-to-date.
Response with rewrite query:
Explainable AI (XAI) is a subfield of artificial intelligence that aims to make AI systems more transparent and understandable. The primary goal of XAI is to provide insights into how AI models make decisions, thereby enhancing trust, accountability, and fairness in AI systems.

XAI techniques are being developed to explain AI decisions, making it possible for users to assess the reliability and fairness of AI systems. This is crucial in various domains, including environmental monitoring, healthcare, and finance, where AI systems are increasingly being used to make critical decisions.

The importance of XAI lies in its potential to address several concerns associated with AI, such as:

1. Lack of transparency: AI systems can be complex and difficult to understand, making it challenging to assess their decision-making processes.
2. Bias and fairness: AI systems can perpetuate biases and unfairness if their decision-making processes are not transparent and explainable.
3. Accountability: XAI can help establish accountability for AI systems, enabling developers, deployers, and users to take responsibility for their actions.

By making AI systems more transparent and explainable, XAI can help build trust in AI, ensure fairness and accuracy, and mitigate potential risks associated with AI. As AI continues to advance and become more pervasive, the importance of XAI will only grow, and it is likely to play a critical role in shaping the future of AI research and development.
==================================================

===== Running RAG with step_back query =====
Extracting text from PDF...
Chunking text...
Created 42 text chunks
Creating embeddings for chunks...
Added 42 chunks to the vector store
Transformation type: step_back
Original query: What is 'Explainable AI' and why is it considered important?
Step-back query: Here's a broader, more general version of the original query:

"Background information on the concept and significance of Explainable AI in the field of Artificial Intelligence."

This step-back query can help retrieve useful background information on Explainable AI, including its definition, applications, benefits, and challenges, as well as its relevance to the broader field of Artificial Intelligence.
Response with step_back query:
Explainable AI (XAI) is a subfield of artificial intelligence that aims to make AI systems more transparent and understandable. The primary goal of XAI is to provide insights into how AI models make decisions, thereby enhancing trust, accountability, and fairness in AI systems.

XAI techniques are being developed to explain AI decisions, making it possible for users to assess the reliability and fairness of AI systems. This is crucial in various domains, including environmental monitoring, healthcare, and finance, where AI systems are increasingly being used to make critical decisions.

The importance of XAI lies in its potential to address several concerns associated with AI, such as:

1. Lack of transparency: AI systems can be complex and difficult to understand, making it challenging to assess their decision-making processes.
2. Bias and fairness: AI systems can perpetuate biases and unfairness if their decision-making processes are not transparent and explainable.
3. Accountability: XAI can help establish accountability for AI systems, enabling developers, deployers, and users to take responsibility for their actions.

By making AI systems more transparent and explainable, XAI can help build trust in AI, ensure fairness and accuracy, and mitigate potential risks associated with AI. As AI continues to advance and become more pervasive, the importance of XAI will only grow, and it is likely to play a critical role in shaping the future of AI research and development.
==================================================

===== Running RAG with decompose query =====
Extracting text from PDF...
Chunking text...
Created 42 text chunks
Creating embeddings for chunks...
Added 42 chunks to the vector store
Transformation type: decompose
Original query: What is 'Explainable AI' and why is it considered important?
Decomposed into sub-queries:
1. What is the definition of 'Explainable AI' and how does it differ from traditional machine learning models?
2. What are the primary goals and objectives of Explainable AI, and how do they align with broader societal needs?
3. What are the key challenges and limitations in developing and deploying Explainable AI systems, and how can they be addressed?
4. How does the importance of Explainable AI relate to broader societal concerns, such as trust, accountability, and fairness in AI decision-making?
Response with decompose query:
Explainable AI (XAI) is a set of techniques aimed at making AI decisions more understandable and transparent. The primary goal of XAI is to enable users to assess the fairness and accuracy of AI systems. This is achieved by providing insights into the decision-making processes of AI systems, thereby enhancing trust and accountability.

XAI is considered important for several reasons. Firstly, it addresses concerns about the fairness and accuracy of AI systems, which are crucial for building trust in AI. By making AI decisions more understandable, XAI helps users to evaluate the reliability and fairness of AI systems.

Secondly, XAI is essential for ensuring accountability and responsibility in AI systems. By providing explanations for AI decisions, XAI enables developers, deployers, and users to take ownership of AI systems and address potential harms.

Lastly, XAI is critical for ensuring the responsible handling of data in AI systems. By making AI systems more transparent, XAI helps to mitigate concerns about privacy and data protection, and ensures that data is handled in a way that is compliant with regulations.

In summary, Explainable AI is a crucial aspect of building trust in AI, ensuring accountability and responsibility, and promoting responsible data handling. Its importance lies in its ability to make AI systems more transparent, understandable, and accountable, thereby enhancing trust and reliability in AI.
==================================================

===== EVALUATION RESULTS =====
**Comparison of Query Transformation Techniques**

I will evaluate the responses generated by different query transformation techniques (original, rewrite, step_back, decompose) and compare them to the reference answer.

**1. Original Query Response**

* Score: 8/10
* Strengths:
	+ Accurately conveys the main idea of Explainable AI (XAI)
	+ Provides some supporting details about the importance of XAI
* Weaknesses:
	+ Lacks clarity and concision in some sentences
	+ Some sentences are wordy and could be rephrased for better clarity

**2. Rewrite Query Response**

* Score: 9/10
* Strengths:
	+ More concise and clear than the original response
	+ Provides a clearer structure and organization
	+ Accurately conveys the main idea of XAI
* Weaknesses:
	+ Some sentences are still wordy and could be rephrased for better clarity
	+ Lacks supporting details about the importance of XAI

**3. Step_back Query Response**

* Score: 8.5/10
* Strengths:
	+ Provides a clear and concise overview of XAI
	+ Accurately conveys the main idea of XAI
	+ Lacks some supporting details about the importance of XAI
* Weaknesses:
	+ Some sentences are wordy and could be rephrased for better clarity
	+ Lacks a clear conclusion or summary

**4. Decompose Query Response**

* Score: 9.5/10
* Strengths:
	+ Provides a clear and concise overview of XAI
	+ Accurately conveys the main idea of XAI
	+ Provides more supporting details about the importance of XAI
	+ Lacks some clarity in the introduction
* Weaknesses:
	+ Some sentences are wordy and could be rephrased for better clarity
	+ Lacks a clear conclusion or summary

**Ranking of Techniques**

1. Decompose Query Response (9.5/10)
2. Rewrite Query Response (9/10)
3. Step_back Query Response (8.5/10)
4. Original Query Response (8/10)

**Why Decompose Query Response Performed Best**

The Decompose Query Response performed best overall because it provided a clear and concise overview of XAI, accurately conveying the main idea of the topic. It also provided more supporting details about the importance of XAI, which helped to strengthen the response. Additionally, the Decompose Query Response was well-organized and easy to follow, making it a pleasure to read. While the Rewrite Query Response was close in score, it lacked some supporting details about the importance of XAI, which made it slightly less effective than the Decompose Query Response.
=============================
```
