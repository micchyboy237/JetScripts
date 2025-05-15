from chunker import OpenAI
from openai import OpenAI
from typing import List, Dict, Tuple
import networkx as nx
import matplotlib.pyplot as plt
import heapq
from collections import defaultdict
import re
from PIL import Image
import io

class Chunker:
    def __init__(self):
        self.openai = OpenAI()

    def create_embeddings(self, texts, model="BAAI/bge-en-icl"):
        """
        Create embeddings for the given texts.

        Args:
            texts (List[str]): Input texts
            model (str): Embedding model name

        Returns:
            List[List[float]]: Embedding vectors
        """
        # Handle empty input
        if not texts:
            return []
        # Process in batches if needed (OpenAI API limits)
        batch_size = 100
        all_embeddings = []
        # Iterate over the input texts in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]  # Get the current batch of texts
            # Create embeddings for the current batch
            response = self.openai.embeddings.create(
                model=model,
                input=batch
            )
            # Extract embeddings from the response
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)  # Add the batch embeddings to the list
        return all_embeddings  # Return all embeddings

    def generate_response(self, query, context_chunks):
        """
        Generate a response using the retrieved context.

        Args:
            query (str): The user's question
            context_chunks (List[Dict]): Relevant chunks from graph traversal

        Returns:
            str: Generated response
        """
        # Extract text from each chunk in the context
        context_texts = [chunk["text"] for chunk in context_chunks]

        # Combine the extracted texts into a single context string, separated by "---"
        combined_context = "nn---nn".join(context_texts)
        # Define the maximum allowed length for the context (OpenAI limit)
        max_context = 14000
        # Truncate the combined context if it exceeds the maximum length
        if len(combined_context) > max_context:
            combined_context = combined_context[:max_context] + "... [truncated]"
        # Define the system message to guide the AI assistant
        system_message = """You are a helpful AI assistant. Answer the user's question based on the provided context.  If the information is not in the context, say so. Refer to specific parts of the context in your answer when possible."""
        # Generate the response using the OpenAI API
        response = self.openai.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",  # Specify the model to use
            messages=[
                {"role": "system", "content": system_message},  # System message to guide the assistant
                {"role": "user", "content": f"Context:n{combined_context}nnQuestion: {query}"}  # User message with context and query
            ],
            temperature=0.2  # Set the temperature for response generation
        )
        # Return the generated response content
        return response.choices[0].message.content

    def graph_rag_pipeline(self, pdf_path, query, chunk_size=1000, chunk_overlap=200, top_k=3):
        """
        Complete Graph RAG pipeline from document to answer.

        Args:
            pdf_path (str): Path to the PDF document
            query (str): The user's question
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between chunks
            top_k (int): Number of initial nodes to consider

        Returns:
            Dict: Results including answer and graph visualization data
        """
        # Extract text from the PDF document
        text = self.extract_text_from_pdf(pdf_path)
        # Split the extracted text into overlapping chunks
        chunks = self.chunk_text(text, chunk_size, chunk_overlap)
        # Build a knowledge graph from the text chunks
        graph, embeddings = self.build_knowledge_graph(chunks)
        # Traverse the knowledge graph to find relevant information for the query
        relevant_chunks, traversal_path = self.traversal(graph, top_k)
        # Return the results including answer and graph visualization data
        return {"answer": relevant_chunks, "graph": graph, "embeddings": embeddings}

    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from the given PDF document.

        Args:
            pdf_path (str): Path to the PDF document

        Returns:
            str: Extracted text
        """
        # Use PyMuPDF to extract text from the PDF document
        with open(pdf_path, "rb") as f:
            text = f.read().decode()
        return text

    def chunk_text(self, text, chunk_size, chunk_overlap):
        """
        Chunk the given text into overlapping chunks.

        Args:
            text (str): Text to be chunked
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between chunks

        Returns:
            List[List[str]]: Chunks of the given text
        """
        # Use networkx to build a graph from the text chunks
        graph = nx.Graph()
        for i in range(len(text)):
            for j in range(i + chunk_size, len(text)):
                graph.add_edge(text[i:i + chunk_size], text[j:j + chunk_overlap])
        # Build the knowledge graph from the text chunks
        chunks = list(nx.all_simple_graphs(graph, num_nodes=100))
        return chunks

    def traversal(self, graph, top_k):
        """
        Traverse the knowledge graph to find relevant information for the query.

        Args:
            graph (nx.Graph): Graph from the text chunks
            top_k (int): Number of initial nodes to consider

        Returns:
            Dict: Results including answer and graph visualization data
        """
        # Use the graph to find relevant information for the query
        relevant_chunks, traversal_path = graph.topological_sort()
        # Return the results including answer and graph visualization data
        return {"answer": relevant_chunks, "graph": graph, "embeddings": []}

    def openai(self):
        # Initialize the OpenAI API
        self.openai = OpenAI()

    def openai_chat(self):
        # Initialize the OpenAI API for chat functionality
        self.openai = OpenAI("user:YOUR_API_KEY", "user:YOUR_API_SECRET")

    def openai_graph(self):
        # Initialize the OpenAI API for graph functionality
        self.openai = OpenAI("user:YOUR_API_KEY", "user:YOUR_API_SECRET")

    def openai_graph_rag(self):
        # Initialize the OpenAI API for graph and RAG functionality
        self.openai = OpenAI("user:YOUR_API_KEY", "user:YOUR_API_SECRET")

    def __main__(self):
        # Example usage for creating embeddings
        texts = ["This is a sample text.", "Another example for the AI assistant."]
        model = "BAAI/bge-en-icl"
        all_embeddings = self.create_embeddings(texts, model)
        print("All embeddings:", all_embeddings)

        # Example usage for generating a response
        query = "What is the meaning of life?"
        context_chunks = [{"text": "The meaning of life is to find your purpose."}, {"text": "To find your passion."}]
        response = self.generate_response(query, context_chunks)
        print("Generated response:", response)

        # Example usage for completing the Graph RAG pipeline
        pdf_path = "document.pdf"
        query = "What is the meaning of life?"
        chunk_size = 1000
        chunk_overlap = 200
        top_k = 3
        graph, embeddings = self.graph_rag_pipeline(pdf_path, query, chunk_size, chunk_overlap, top_k)
        print("Graph and embeddings:", graph, embeddings)

        # Example usage for extracting text from a PDF document
        pdf_path = "document.pdf"
        text = self.extract_text_from_pdf(pdf_path)
        print("Extracted text:", text)

        # Example usage for chunking the text
        text = "This is a sample text. This is another example for the AI assistant."
        chunks = self.chunk_text(text, 1000, 200)
        print("Chunks:", chunks)

        # Example usage for traversing the knowledge graph
        graph = nx.Graph()
        chunks = self.chunk_text(text, 1000, 200)
        traversal_path = graph.topological_sort()
        print("Traversal path:", traversal_path)

if __name__ == "__main__":
    chunker = Chunker()
    chunker.__main__()