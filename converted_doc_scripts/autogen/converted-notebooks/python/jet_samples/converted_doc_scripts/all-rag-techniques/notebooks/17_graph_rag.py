from PIL import Image
from collections import defaultdict
from jet.logger import CustomLogger
from openai import MLX
from typing import List, Dict, Tuple, Any
import fitz  # PyMuPDF
import heapq
import io
import json
import matplotlib.pyplot as plt
import networkx as nx
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
# Graph RAG: Graph-Enhanced Retrieval-Augmented Generation

In this notebook, I implement Graph RAG - a technique that enhances traditional RAG systems by organizing knowledge as a connected graph rather than a flat collection of documents. This allows the system to navigate related concepts and retrieve more contextually relevant information than standard vector similarity approaches.

Key Benefits of Graph RAG

- Preserves relationships between pieces of information
- Enables traversal through connected concepts to find relevant context
- Improves handling of complex, multi-part queries
- Provides better explainability through visualized knowledge paths

## Setting Up the Environment
We begin by importing necessary libraries.
"""
logger.info("# Graph RAG: Graph-Enhanced Retrieval-Augmented Generation")


"""
## Setting Up the MLX API Client
We initialize the MLX client to generate embeddings and responses.
"""
logger.info("## Setting Up the MLX API Client")

client = MLX(
    base_url="https://api.studio.nebius.com/v1/",
#     api_key=os.getenv("OPENAI_API_KEY")  # Retrieve the API key from environment variables
)

"""
## Document Processing Functions
"""
logger.info("## Document Processing Functions")

def extract_text_from_pdf(pdf_path):
    """
    Extract text content from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        str: Extracted text content
    """
    logger.debug(f"Extracting text from {pdf_path}...")  # Print the path of the PDF being processed
    pdf_document = fitz.open(pdf_path)  # Open the PDF file using PyMuPDF
    text = ""  # Initialize an empty string to store the extracted text

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]  # Get the page object
        text += page.get_text()  # Extract text from the page and append to the text string

    return text  # Return the extracted text content

def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Split text into overlapping chunks.

    Args:
        text (str): Input text to chunk
        chunk_size (int): Size of each chunk in characters
        overlap (int): Overlap between chunks in characters

    Returns:
        List[Dict]: List of chunks with metadata
    """
    chunks = []  # Initialize an empty list to store the chunks

    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]

        if chunk_text:
            chunks.append({
                "text": chunk_text,  # The chunk of text
                "index": len(chunks),  # The index of the chunk
                "start_pos": i,  # The starting position of the chunk in the original text
                "end_pos": i + len(chunk_text)  # The ending position of the chunk in the original text
            })

    logger.debug(f"Created {len(chunks)} text chunks")

    return chunks  # Return the list of chunks

"""
## Creating Embeddings
"""
logger.info("## Creating Embeddings")

def create_embeddings(texts, model="BAAI/bge-en-icl"):
    """
    Create embeddings for the given texts.

    Args:
        texts (List[str]): Input texts
        model (str): Embedding model name

    Returns:
        List[List[float]]: Embedding vectors
    """
    if not texts:
        return []

    batch_size = 100
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]  # Get the current batch of texts

        response = client.embeddings.create(
            model=model,
            input=batch
        )

        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)  # Add the batch embeddings to the list

    return all_embeddings  # Return all embeddings

"""
## Knowledge Graph Construction
"""
logger.info("## Knowledge Graph Construction")

def extract_concepts(text):
    """
    Extract key concepts from text using MLX's API.

    Args:
        text (str): Text to extract concepts from

    Returns:
        List[str]: List of concepts
    """
    system_message = """Extract key concepts and entities from the provided text.
Return ONLY a list of 5-10 key terms, entities, or concepts that are most important in this text.
Format your response as a JSON array of strings."""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Extract key concepts from:\n\n{text[:3000]}"}  # Limit for API
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )

    try:
        concepts_json = json.loads(response.choices[0].message.content)
        concepts = concepts_json.get("concepts", [])
        if not concepts and "concepts" not in concepts_json:
            for key, value in concepts_json.items():
                if isinstance(value, list):
                    concepts = value
                    break
        return concepts
    except (json.JSONDecodeError, AttributeError):
        content = response.choices[0].message.content
        matches = re.findall(r'\[(.*?)\]', content, re.DOTALL)
        if matches:
            items = re.findall(r'"([^"]*)"', matches[0])
            return items
        return []

def build_knowledge_graph(chunks):
    """
    Build a knowledge graph from text chunks.

    Args:
        chunks (List[Dict]): List of text chunks with metadata

    Returns:
        Tuple[nx.Graph, List[np.ndarray]]: The knowledge graph and chunk embeddings
    """
    logger.debug("Building knowledge graph...")

    graph = nx.Graph()

    texts = [chunk["text"] for chunk in chunks]

    logger.debug("Creating embeddings for chunks...")
    embeddings = create_embeddings(texts)

    logger.debug("Adding nodes to the graph...")
    for i, chunk in enumerate(chunks):
        logger.debug(f"Extracting concepts for chunk {i+1}/{len(chunks)}...")
        concepts = extract_concepts(chunk["text"])

        graph.add_node(i,
                      text=chunk["text"],
                      concepts=concepts,
                      embedding=embeddings[i])

    logger.debug("Creating edges between nodes...")
    for i in range(len(chunks)):
        node_concepts = set(graph.nodes[i]["concepts"])

        for j in range(i + 1, len(chunks)):
            other_concepts = set(graph.nodes[j]["concepts"])
            shared_concepts = node_concepts.intersection(other_concepts)

            if shared_concepts:
                similarity = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))

                concept_score = len(shared_concepts) / min(len(node_concepts), len(other_concepts))
                edge_weight = 0.7 * similarity + 0.3 * concept_score

                if edge_weight > 0.6:
                    graph.add_edge(i, j,
                                  weight=edge_weight,
                                  similarity=similarity,
                                  shared_concepts=list(shared_concepts))

    logger.debug(f"Knowledge graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    return graph, embeddings

"""
## Graph Traversal and Query Processing
"""
logger.info("## Graph Traversal and Query Processing")

def traverse_graph(query, graph, embeddings, top_k=5, max_depth=3):
    """
    Traverse the knowledge graph to find relevant information for the query.

    Args:
        query (str): The user's question
        graph (nx.Graph): The knowledge graph
        embeddings (List): List of node embeddings
        top_k (int): Number of initial nodes to consider
        max_depth (int): Maximum traversal depth

    Returns:
        List[Dict]: Relevant information from graph traversal
    """
    logger.debug(f"Traversing graph for query: {query}")

    query_embedding = create_embeddings(query)

    similarities = []
    for i, node_embedding in enumerate(embeddings):
        similarity = np.dot(query_embedding, node_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(node_embedding))
        similarities.append((i, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)

    starting_nodes = [node for node, _ in similarities[:top_k]]
    logger.debug(f"Starting traversal from {len(starting_nodes)} nodes")

    visited = set()  # Set to keep track of visited nodes
    traversal_path = []  # List to store the traversal path
    results = []  # List to store the results

    queue = []
    for node in starting_nodes:
        heapq.heappush(queue, (-similarities[node][1], node))  # Negative for max-heap

    while queue and len(results) < (top_k * 3):  # Limit results to top_k * 3
        _, node = heapq.heappop(queue)

        if node in visited:
            continue

        visited.add(node)
        traversal_path.append(node)

        results.append({
            "text": graph.nodes[node]["text"],
            "concepts": graph.nodes[node]["concepts"],
            "node_id": node
        })

        if len(traversal_path) < max_depth:
            neighbors = [(neighbor, graph[node][neighbor]["weight"])
                        for neighbor in graph.neighbors(node)
                        if neighbor not in visited]

            for neighbor, weight in sorted(neighbors, key=lambda x: x[1], reverse=True):
                heapq.heappush(queue, (-weight, neighbor))

    logger.debug(f"Graph traversal found {len(results)} relevant chunks")
    return results, traversal_path

"""
## Response Generation
"""
logger.info("## Response Generation")

def generate_response(query, context_chunks):
    """
    Generate a response using the retrieved context.

    Args:
        query (str): The user's question
        context_chunks (List[Dict]): Relevant chunks from graph traversal

    Returns:
        str: Generated response
    """
    context_texts = [chunk["text"] for chunk in context_chunks]

    combined_context = "\n\n---\n\n".join(context_texts)

    max_context = 14000

    if len(combined_context) > max_context:
        combined_context = combined_context[:max_context] + "... [truncated]"

    system_message = """You are a helpful AI assistant. Answer the user's question based on the provided context.
If the information is not in the context, say so. Refer to specific parts of the context in your answer when possible."""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # Specify the model to use
        messages=[
            {"role": "system", "content": system_message},  # System message to guide the assistant
            {"role": "user", "content": f"Context:\n{combined_context}\n\nQuestion: {query}"}  # User message with context and query
        ],
        temperature=0.2  # Set the temperature for response generation
    )

    return response.choices[0].message.content

"""
## Visualization
"""
logger.info("## Visualization")

def visualize_graph_traversal(graph, traversal_path):
    """
    Visualize the knowledge graph and the traversal path.

    Args:
        graph (nx.Graph): The knowledge graph
        traversal_path (List): List of nodes in traversal order
    """
    plt.figure(figsize=(12, 10))  # Set the figure size

    node_color = ['lightblue'] * graph.number_of_nodes()

    for node in traversal_path:
        node_color[node] = 'lightgreen'

    if traversal_path:
        node_color[traversal_path[0]] = 'green'
        node_color[traversal_path[-1]] = 'red'

    pos = nx.spring_layout(graph, k=0.5, iterations=50, seed=42)

    nx.draw_networkx_nodes(graph, pos, node_color=node_color, node_size=500, alpha=0.8)

    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', 1.0)
        nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], width=weight*2, alpha=0.6)

    traversal_edges = [(traversal_path[i], traversal_path[i+1])
                      for i in range(len(traversal_path)-1)]

    nx.draw_networkx_edges(graph, pos, edgelist=traversal_edges,
                          width=3, alpha=0.8, edge_color='red',
                          style='dashed', arrows=True)

    labels = {}
    for node in graph.nodes():
        concepts = graph.nodes[node]['concepts']
        label = concepts[0] if concepts else f"Node {node}"
        labels[node] = f"{node}: {label}"

    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)

    plt.title("Knowledge Graph with Traversal Path")  # Set the plot title
    plt.axis('off')  # Turn off the axis
    plt.tight_layout()  # Adjust layout
    plt.show()  # Display the plot

"""
## Complete Graph RAG Pipeline
"""
logger.info("## Complete Graph RAG Pipeline")

def graph_rag_pipeline(pdf_path, query, chunk_size=1000, chunk_overlap=200, top_k=3):
    """
    Complete Graph RAG pipeline from document to answer.

    Args:
        pdf_path (str): Path to the PDF document
        query (str): The user's question
        chunk_size (int): Size of text chunks
        chunk_overlap (int): Overlap between chunks
        top_k (int): Number of top nodes to consider for traversal

    Returns:
        Dict: Results including answer and graph visualization data
    """
    text = extract_text_from_pdf(pdf_path)

    chunks = chunk_text(text, chunk_size, chunk_overlap)

    graph, embeddings = build_knowledge_graph(chunks)

    relevant_chunks, traversal_path = traverse_graph(query, graph, embeddings, top_k)

    response = generate_response(query, relevant_chunks)

    visualize_graph_traversal(graph, traversal_path)

    return {
        "query": query,
        "response": response,
        "relevant_chunks": relevant_chunks,
        "traversal_path": traversal_path,
        "graph": graph
    }

"""
## Evaluation Function
"""
logger.info("## Evaluation Function")

def evaluate_graph_rag(pdf_path, test_queries, reference_answers=None):
    """
    Evaluate Graph RAG on multiple test queries.

    Args:
        pdf_path (str): Path to the PDF document
        test_queries (List[str]): List of test queries
        reference_answers (List[str], optional): Reference answers for comparison

    Returns:
        Dict: Evaluation results
    """
    text = extract_text_from_pdf(pdf_path)

    chunks = chunk_text(text)

    graph, embeddings = build_knowledge_graph(chunks)

    results = []

    for i, query in enumerate(test_queries):
        logger.debug(f"\n\n=== Evaluating Query {i+1}/{len(test_queries)} ===")
        logger.debug(f"Query: {query}")

        relevant_chunks, traversal_path = traverse_graph(query, graph, embeddings)

        response = generate_response(query, relevant_chunks)

        reference = None
        comparison = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
            comparison = compare_with_reference(response, reference, query)

        results.append({
            "query": query,
            "response": response,
            "reference_answer": reference,
            "comparison": comparison,
            "traversal_path_length": len(traversal_path),
            "relevant_chunks_count": len(relevant_chunks)
        })

        logger.debug(f"\nResponse: {response}\n")
        if comparison:
            logger.debug(f"Comparison: {comparison}\n")

    return {
        "results": results,
        "graph_stats": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "avg_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes()
        }
    }

def compare_with_reference(response, reference, query):
    """
    Compare generated response with reference answer.

    Args:
        response (str): Generated response
        reference (str): Reference answer
        query (str): Original query

    Returns:
        str: Comparison analysis
    """
    system_message = """Compare the AI-generated response with the reference answer.
Evaluate based on: correctness, completeness, and relevance to the query.
Provide a brief analysis (2-3 sentences) of how well the generated response matches the reference."""

    prompt = f"""
Query: {query}

AI-generated response:
{response}

Reference answer:
{reference}

How well does the AI response match the reference?
"""

    comparison = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_message},  # System message to guide the assistant
            {"role": "user", "content": prompt}  # User message with the prompt
        ],
        temperature=0.0  # Set the temperature for response generation
    )

    return comparison.choices[0].message.content

"""
## Evaluation of Graph RAG on a Sample PDF Document
"""
logger.info("## Evaluation of Graph RAG on a Sample PDF Document")

pdf_path = f"{GENERATED_DIR}/AI_Information.pdf"

query = "What are the key applications of transformers in natural language processing?"

results = graph_rag_pipeline(pdf_path, query)

logger.debug("\n=== ANSWER ===")
logger.debug(results["response"])

test_queries = [
    "How do transformers handle sequential data compared to RNNs?"
]

reference_answers = [
    "Transformers handle sequential data differently from RNNs by using self-attention mechanisms instead of recurrent connections. This allows transformers to process all tokens in parallel rather than sequentially, capturing long-range dependencies more efficiently and enabling better parallelization during training. Unlike RNNs, transformers don't suffer from vanishing gradient problems with long sequences."
]

evaluation = evaluate_graph_rag(pdf_path, test_queries, reference_answers)

logger.debug("\n=== EVALUATION SUMMARY ===")
logger.debug(f"Graph nodes: {evaluation['graph_stats']['nodes']}")
logger.debug(f"Graph edges: {evaluation['graph_stats']['edges']}")
for i, result in enumerate(evaluation['results']):
    logger.debug(f"\nQuery {i+1}: {result['query']}")
    logger.debug(f"Path length: {result['traversal_path_length']}")
    logger.debug(f"Chunks used: {result['relevant_chunks_count']}")

logger.info("\n\n[DONE]", bright=True)