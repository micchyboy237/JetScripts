from typing import List, Optional
from jet.llm.rag.rag_preprocessor import WebDataPreprocessor
from jet.logger import logger
from jet.models.model_types import ModelType
from jet.models.utils import resolve_model_value
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from transformers import AutoTokenizer
import trafilatura
import logging
import spacy
from spacy.language import Language
import textacy.preprocessing as tprep
import numpy as np


class MLXRAGProcessor:
    """Processes preprocessed web data with MLX for RAG usage."""

    def __init__(self, model_name: ModelType = "qwen3-1.7b-4bit"):
        """Initialize with MLX model and tokenizer."""
        logger.info(f"Loading MLX model: {model_name}")
        try:
            model_path = resolve_model_value(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model, _ = load(model_path)
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {str(e)}")
            raise

    def generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks using MLX."""
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embeddings = []
        for chunk in chunks:
            inputs = self.tokenizer(
                chunk, return_tensors="np", padding=True, truncation=True, max_length=512)
            input_ids = mx.array(inputs["input_ids"]).astype(
                mx.int32)  # Ensure int32 for token IDs
            output = self.model(input_ids)
            embedding = np.array(
                mx.mean(output, axis=1).tolist(), dtype=np.float32).squeeze()
            embeddings.append(embedding)
        embeddings_array = np.stack(embeddings)
        logger.info(f"Generated embeddings shape: {embeddings_array.shape}")
        return embeddings_array

    def process_query(self, query: str, chunks: List[str], embeddings: np.ndarray) -> str:
        """Process a query using embeddings for RAG."""
        logger.info(f"Processing query: {query}")
        query_inputs = self.tokenizer(
            query, return_tensors="np", padding=True, truncation=True, max_length=512)
        logger.debug(
            f"Query input IDs shape: {query_inputs['input_ids'].shape}, dtype: {query_inputs['input_ids'].dtype}")
        query_input_ids = mx.array(query_inputs["input_ids"])
        query_output = self.model(query_input_ids)
        logger.debug(
            f"Query output shape: {query_output.shape}, dtype: {query_output.dtype}")
        mean_output = mx.mean(query_output, axis=1)
        logger.debug(
            f"Mean output shape: {mean_output.shape}, dtype: {mean_output.dtype}")
        query_embedding = np.array(
            mean_output.tolist(), dtype=np.float32).squeeze()
        logger.debug(
            f"Query embedding shape: {query_embedding.shape}, dtype: {query_embedding.dtype}")
        norm_embeddings = embeddings / \
            np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm_query = query_embedding / \
            np.linalg.norm(query_embedding, keepdims=True)
        logger.debug(
            f"Norm query shape: {norm_query.shape}, Norm embeddings shape: {norm_embeddings.shape}")
        similarities = np.dot(norm_embeddings, norm_query)
        logger.debug(f"Similarities shape: {similarities.shape}")
        top_idx = np.argmax(similarities)
        relevant_chunk = chunks[top_idx]
        response = f"Based on the context: {relevant_chunk[:100]}..., the answer is derived from the relevant information."
        logger.info("Query processed successfully")
        return response


def main():
    """Main function to demonstrate preprocessing and MLX RAG usage."""
    try:
        logger.info("Initializing WebDataPreprocessor")
        preprocessor = WebDataPreprocessor(chunk_size=500, chunk_overlap=50)

        url = "https://example.com"
        logger.info(f"Preprocessing content from {url}")
        chunks = preprocessor.preprocess(url)
        if not chunks:
            logger.error("No chunks generated, exiting")
            return

        logger.info(f"Generated {len(chunks)} chunks from {url}")

        logger.info("Initializing MLXRAGProcessor")
        mlx_processor = MLXRAGProcessor()

        logger.info("Generating embeddings for chunks")
        embeddings = mlx_processor.generate_embeddings(chunks)
        if embeddings.shape[0] != len(chunks):
            logger.error("Mismatch between chunks and embeddings, exiting")
            return

        query = "What is the main topic of the webpage?"
        logger.info(f"Processing query: {query}")
        response = mlx_processor.process_query(query, chunks, embeddings)

        print(f"Query: {query}")
        print(f"Response: {response}")
        print(f"Number of chunks processed: {len(chunks)}")
        logger.info("Main function completed successfully")

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
