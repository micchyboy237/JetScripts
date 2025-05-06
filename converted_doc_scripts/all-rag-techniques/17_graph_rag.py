from PIL import Image
from collections import defaultdict
from jet.llm.mlx.base import MLX
from jet.llm.utils.embeddings import get_embedding_function
from jet.logger import CustomLogger
from typing import List, Dict, Tuple, Any
import io
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pypdf
import re

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")
file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)
logger.info("Initializing MLX and embedding function")
mlx = MLX()
embed_func = get_embedding_function("mxbai-embed-large")


def extract_text_from_pdf(pdf_path):
    logger.debug(f"Extracting text from {pdf_path}...")
    all_text = ""
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text() or ""
            all_text += text
    return all_text


def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "index": len(chunks),
                "start_pos": i,
                "end_pos": i + len(chunk_text)
            })
    logger.debug(f"Created {len(chunks)} text chunks")
    return chunks


def create_embeddings(texts):
    if not texts:
        return []
    embeddings = embed_func(texts)
    return embeddings


def extract_concepts(text, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    system_message = "You are an expert in knowledge extraction. Extract key concepts from the provided text. Return a JSON object with a 'concepts' key containing a list of strings."
    response = mlx.chat(
        [
            {"role": "system", "content": system_message},
            {"role": "user",
                "content": f"Extract key concepts from:\n\n{text[:3000]}"}
        ],
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    try:
        concepts_json = json.loads(
            response["choices"][0]["message"]["content"])
        concepts = concepts_json.get("concepts", [])
        if not concepts and "concepts" not in concepts_json:
            for key, value in concepts_json.items():
                if isinstance(value, list):
                    concepts = value
                    break
        return concepts
    except (json.JSONDecodeError, AttributeError):
        content = response["choices"][0]["message"]["content"]
        matches = re.findall(r'\[(.*?)\]', content, re.DOTALL)
        if matches:
            items = re.findall(r'"([^"]*)"', matches[0])
            return items
        return []


def build_knowledge_graph(chunks):
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
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                concept_score = len(shared_concepts) / \
                    min(len(node_concepts), len(other_concepts))
                edge_weight = 0.7 * similarity + 0.3 * concept_score
                if edge_weight > 0.6:
                    graph.add_edge(i, j,
                                   weight=edge_weight,
                                   similarity=similarity,
                                   shared_concepts=list(shared_concepts))
    logger.debug(
        f"Knowledge graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    return graph, embeddings


def traverse_graph(query, graph, embeddings, top_k=5, max_depth=3):
    logger.debug(f"Traversing graph for query: {query}")
    query_embedding = create_embeddings([query])[0]
    similarities = []
    for i, node_embedding in enumerate(embeddings):
        similarity = np.dot(query_embedding, node_embedding) / \
            (np.linalg.norm(query_embedding) * np.linalg.norm(node_embedding))
        similarities.append((i, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    starting_nodes = [node for node, _ in similarities[:top_k]]
    logger.debug(f"Starting traversal from {len(starting_nodes)} nodes")
    visited = set()
    traversal_path = []
    results = []
    queue = []
    for node in starting_nodes:
        heapq.heappush(queue, (-similarities[node][1], node))
    while queue and len(results) < (top_k * 3):
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


def generate_response(query, context_chunks, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    context_texts = [chunk["text"] for chunk in context_chunks]
    combined_context = "\n\n---\n\n".join(context_texts)
    max_context = 14000
    if len(combined_context) > max_context:
        combined_context = combined_context[:max_context] + "... [truncated]"
    system_message = "You are a helpful AI assistant. Answer the query based strictly on the provided context. If the context doesn't contain the answer, state that clearly."
    response = mlx.chat(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Context:\n{combined_context}\n\nQuestion: {query}"}
        ],
        model=model,
        temperature=0.2
    )
    return response["choices"][0]["message"]["content"]


def visualize_graph_traversal(graph, traversal_path):
    plt.figure(figsize=(12, 10))
    node_color = ['lightblue'] * graph.number_of_nodes()
    for node in traversal_path:
        node_color[node] = 'lightgreen'
    if traversal_path:
        node_color[traversal_path[0]] = 'green'
        node_color[traversal_path[-1]] = 'red'
    pos = nx.spring_layout(graph, k=0.5, iterations=50, seed=42)
    nx.draw_networkx_nodes(
        graph, pos, node_color=node_color, node_size=500, alpha=0.8)
    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', 1.0)
        nx.draw_networkx_edges(graph, pos, edgelist=[
                               (u, v)], width=weight*2, alpha=0.6)
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
    plt.title("Knowledge Graph with Traversal Path")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(GENERATED_DIR, 'graph_traversal.png'))


def graph_rag_pipeline(pdf_path, query, chunk_size=1000, chunk_overlap=200, top_k=3):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    graph, embeddings = build_knowledge_graph(chunks)
    relevant_chunks, traversal_path = traverse_graph(
        query, graph, embeddings, top_k)
    response = generate_response(query, relevant_chunks)
    visualize_graph_traversal(graph, traversal_path)
    return {
        "query": query,
        "response": response,
        "relevant_chunks": relevant_chunks,
        "traversal_path": traversal_path,
        "graph": graph
    }


def evaluate_graph_rag(pdf_path, test_queries, reference_answers=None):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    graph, embeddings = build_knowledge_graph(chunks)
    results = []
    for i, query in enumerate(test_queries):
        logger.debug(f"\n\n=== Evaluating Query {i+1}/{len(test_queries)} ===")
        logger.debug(f"Query: {query}")
        relevant_chunks, traversal_path = traverse_graph(
            query, graph, embeddings)
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


def compare_with_reference(response, reference, query, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    system_message = "You are an evaluator. Compare the response to the reference answer, assessing accuracy and completeness for the query. Provide a concise comparison."
    prompt = f"Query: {query}\n\nResponse: {response}\n\nReference Answer: {reference}"
    comparison = mlx.chat(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        model=model,
        temperature=0.0
    )
    return comparison["choices"][0]["message"]["content"]


pdf_path = os.path.join(GENERATED_DIR, "AI_Information.pdf")
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
