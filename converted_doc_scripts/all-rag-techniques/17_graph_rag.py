import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import re
from typing import List, Dict, Any
from jet.file.utils import save_file
from helpers import (
    setup_config, initialize_mlx, generate_embeddings,
    load_validation_data, generate_ai_response,
    load_json_data, SearchResult, SimpleVectorStore, DATA_DIR, DOCS_PATH
)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """Chunk text into overlapping segments with metadata."""
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


def extract_concepts(text: str, mlx, model: str = "llama-3.2-3b-instruct-4bit") -> List[str]:
    """Extract key concepts from text."""
    system_message = "You are an expert in knowledge extraction. Extract key concepts from the provided text. Return a JSON object with a 'concepts' key containing a list of strings."
    response = mlx.chat(
        [
            {"role": "system", "content": system_message},
            {"role": "user",
                "content": f"Extract key concepts from:\n\n{text[:3000]}"}
        ],
        model=model,
        temperature=0.0,
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


def build_knowledge_graph(chunks: List[Dict[str, Any]], embed_func, source: str) -> tuple[nx.Graph, List[np.ndarray], SimpleVectorStore]:
    """Build a knowledge graph from chunks."""
    logger.debug("Building knowledge graph...")
    graph = nx.Graph()
    texts = [chunk["text"] for chunk in chunks]
    logger.debug("Creating embeddings for chunks...")
    embeddings = generate_embeddings(texts, embed_func, logger)
    logger.debug("Adding nodes to the graph...")
    store = SimpleVectorStore()
    for i, chunk in enumerate(chunks):
        logger.debug(f"Extracting concepts for chunk {i+1}/{len(chunks)}...")
        concepts = extract_concepts(chunk["text"], mlx)
        graph.add_node(i, text=chunk["text"],
                       concepts=concepts, embedding=embeddings[i])
        store.add_item(
            text=chunk["text"],
            embedding=embeddings[i],
            metadata={"index": i, "source": source}
        )
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
                    graph.add_edge(i, j, weight=edge_weight, similarity=similarity,
                                   shared_concepts=list(shared_concepts))
    logger.debug(
        f"Knowledge graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    return graph, embeddings, store


def traverse_graph(query: str, graph: nx.Graph, store: SimpleVectorStore, embed_func, top_k: int = 5, max_depth: int = 3) -> tuple[List[Dict[str, Any]], List[int]]:
    """Traverse graph to find relevant chunks."""
    logger.debug(f"Traversing graph for query: {query}")
    query_embedding = embed_func(query)
    initial_results = store.search(query_embedding, top_k=top_k)
    starting_nodes = [result["metadata"]["index"]
                      for result in initial_results]
    logger.debug(f"Starting traversal from {len(starting_nodes)} nodes")
    visited = set()
    traversal_path = []
    results = []
    queue = []
    for node in starting_nodes:
        similarity = next(
            (r["similarity"] for r in initial_results if r["metadata"]["index"] == node), 0.0)
        queue.append((-similarity, node))
    while queue and len(results) < (top_k * 3):
        _, node = queue.pop(0)
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
                queue.append((-weight, neighbor))
    logger.debug(f"Graph traversal found {len(results)} relevant chunks")
    return results, traversal_path


def visualize_graph_traversal(graph: nx.Graph, traversal_path: List[int]):
    """Visualize graph with traversal path."""
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
        nx.draw_networkx_edges(graph, pos, edlist=[
                               (u, v)], width=weight*2, alpha=0.6)
    traversal_edges = [(traversal_path[i], traversal_path[i+1])
                       for i in range(len(traversal_path)-1)]
    nx.draw_networkx_edges(graph, pos, edlist=traversal_edges, width=3,
                           alpha=0.8, edge_color='red', style='dashed', arrows=True)
    labels = {}
    for node in graph.nodes():
        concepts = graph.nodes[node]['concepts']
        label = concepts[0] if concepts else f"Node {node}"
        labels[node] = f"{node}: {label}"
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)
    plt.title("Knowledge Graph with Traversal Path")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'graph_traversal.png'))


def graph_rag_pipeline(chunks: List[Dict[str, Any]], query: str, embed_func, mlx, source: str = "document", chunk_size: int = 1000, chunk_overlap: int = 200, top_k: int = 3) -> Dict[str, Any]:
    """Run Graph RAG pipeline."""
    graph, embeddings, store = build_knowledge_graph(
        chunks, embed_func, source)
    relevant_chunks, traversal_path = traverse_graph(
        query, graph, store, embed_func, top_k)
    response = generate_ai_response(
        query,
        "You are a helpful AI assistant. Answer the query based strictly on the provided context. If the context doesn't contain the answer, state that clearly.",
        relevant_chunks,
        mlx,
        logger,
        model="llama-3.2-3b-instruct-4bit"
    )
    visualize_graph_traversal(graph, traversal_path)
    return {
        "query": query,
        "response": response,
        "relevant_chunks": relevant_chunks,
        "traversal_path": traversal_path,
        "graph": graph
    }


def evaluate_graph_rag(chunks: List[Dict[str, Any]], test_queries: List[str], embed_func, mlx, reference_answers: List[str] = None) -> Dict[str, Any]:
    """Evaluate Graph RAG performance."""
    graph, embeddings, store = build_knowledge_graph(
        chunks, embed_func, "document")
    results = []
    for i, query in enumerate(test_queries):
        logger.debug(f"\n\n=== Evaluating Query {i+1}/{len(test_queries)} ===")
        logger.debug(f"Query: {query}")
        relevant_chunks, traversal_path = traverse_graph(
            query, graph, store, embed_func)
        response = generate_ai_response(
            query,
            "You are a helpful AI assistant. Answer the query based strictly on the provided context. If the context doesn't contain the answer, state that clearly.",
            relevant_chunks,
            mlx,
            logger,
            model="llama-3.2-3b-instruct-4bit"
        )
        reference = None
        comparison = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
            comparison = compare_with_reference(
                response, reference, query, mlx)
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


def compare_with_reference(response: str, reference: str, query: str, mlx, model: str = "llama-3.2-3b-instruct-4bit") -> str:
    """Compare response with reference answer."""
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


script_dir, generated_dir, log_file, logger = setup_config(__file__)
mlx, embed_func = initialize_mlx(logger)
formatted_texts, original_chunks = load_json_data(DOCS_PATH, logger)
logger.info("Loaded pre-chunked data from DOCS_PATH")
query = "What are the key applications of transformers in natural language processing?"
results = graph_rag_pipeline(
    chunks=original_chunks,
    query=query,
    embed_func=embed_func,
    mlx=mlx
)
logger.debug("\n=== ANSWER ===")
logger.debug(results["response"])
test_queries = [
    "How do transformers handle sequential data compared to RNNs?"
]
reference_answers = [
    "Transformers handle sequential data differently from RNNs by using self-attention mechanisms instead of recurrent connections. This allows transformers to process all tokens in parallel rather than sequentially, capturing long-range dependencies more efficiently and enabling better parallelization during training. Unlike RNNs, transformers don't suffer from vanishing gradient problems with long sequences."
]
evaluation = evaluate_graph_rag(
    chunks=original_chunks,
    test_queries=test_queries,
    embed_func=embed_func,
    mlx=mlx,
    reference_answers=reference_answers
)
save_file(evaluation, f"{generated_dir}/evaluation_results.json")
logger.info(
    f"Saved evaluation results to {generated_dir}/evaluation_results.json")
logger.debug("\n=== EVALUATION SUMMARY ===")
logger.debug(f"Graph nodes: {evaluation['graph_stats']['nodes']}")
logger.debug(f"Graph edges: {evaluation['graph_stats']['edges']}")
for i, result in enumerate(evaluation['results']):
    logger.debug(f"\nQuery {i+1}: {result['query']}")
    logger.debug(f"Path length: {result['traversal_path_length']}")
    logger.debug(f"Chunks used: {result['relevant_chunks_count']}")
logger.info("\n\n[DONE]", bright=True)
