```python
import os
import numpy as np
import json
import fitz  # PyMuPDF
from openai import OpenAI
from typing import List, Dict, Tuple, Any
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
        relevant_chunks, traversal_path = self.tr