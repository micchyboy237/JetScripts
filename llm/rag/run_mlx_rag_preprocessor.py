from typing import List, Iterator
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
from tqdm import tqdm


class MLXRAGProcessor:
    """Processes preprocessed web data with MLX for RAG usage."""

    def __init__(self, model_name: ModelType = "qwen3-1.7b-4bit", batch_size: int = 8, show_progress: bool = False):
        """Initialize with MLX model, tokenizer, batch size, and progress display option."""
        logger.debug(
            f"Loading MLX model: {model_name}, batch_size: {batch_size}, show_progress: {show_progress}")
        self.batch_size = batch_size
        self.show_progress = show_progress
        try:
            model_path = resolve_model_value(model_name)
            logger.debug(f"Resolved model path: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model, _ = load(model_path)
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {str(e)}")
            raise

    def generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks using MLX with batch processing."""
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embeddings = []
        num_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
        iterator = range(0, len(chunks), self.batch_size)
        if self.show_progress:
            iterator = tqdm(iterator, total=num_batches,
                            desc="Processing batches")
        for i in iterator:
            batch_chunks = chunks[i:i + self.batch_size]
            logger.debug(
                f"Processing batch {i//self.batch_size + 1} with {len(batch_chunks)} chunks")
            inputs = self.tokenizer(
                batch_chunks,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=512
            )
            input_ids = mx.array(inputs["input_ids"]).astype(mx.int32)
            logger.debug(
                f"Batch input IDs shape: {input_ids.shape}, dtype: {input_ids.dtype}")
            output = self.model(input_ids)
            logger.debug(
                f"Batch output shape: {output.shape}, dtype: {output.dtype}")
            embedding = np.array(
                mx.mean(output, axis=1).tolist(), dtype=np.float32)
            logger.debug(
                f"Batch NumPy embedding shape: {embedding.shape}, dtype: {embedding.dtype}")
            embeddings.extend(embedding)
            del input_ids, output
            mx.metal.clear_cache()
        embeddings_array = np.stack(embeddings)
        logger.info(f"Generated embeddings shape: {embeddings_array.shape}")
        return embeddings_array

    def generate(self, query: str, chunks: List[str], embeddings: np.ndarray) -> str:
        """Process a query using embeddings for RAG."""
        logger.info(f"Processing query: {query}")
        query_inputs = self.tokenizer(
            query, return_tensors="np", padding=True, truncation=True, max_length=512
        )
        logger.debug(
            f"Query input IDs shape: {query_inputs['input_ids'].shape}, dtype: {query_inputs['input_ids'].dtype}")
        query_input_ids = mx.array(query_inputs["input_ids"]).astype(mx.int32)
        query_output = self.model(query_input_ids)
        logger.debug(
            f"Query output shape: {query_output.shape}, dtype: {query_output.dtype}")
        query_embedding = np.array(
            mx.mean(query_output, axis=1).tolist(), dtype=np.float32).squeeze()
        logger.debug(
            f"Query embedding shape: {query_embedding.shape}, dtype: {query_embedding.dtype}")
        norm_embeddings = embeddings / \
            np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm_query = query_embedding / np.linalg.norm(query_embedding)
        logger.debug(
            f"Norm query shape: {norm_query.shape}, Norm embeddings shape: {norm_embeddings.shape}")
        similarities = np.dot(norm_embeddings, norm_query)
        logger.debug(f"Similarities shape: {similarities.shape}")
        top_idx = np.argmax(similarities)
        relevant_chunk = chunks[top_idx]
        response = f"Based on the context: {relevant_chunk[:100]}..., the answer is derived from the relevant information."
        logger.info("Query processed successfully")
        return response

    def stream_generate(self, query: str, chunks: List[str], embeddings: np.ndarray, top_k: int = 3) -> Iterator[str]:
        """Stream responses for a query based on top-k relevant chunks using embeddings."""
        logger.info(f"Streaming responses for query: {query}, top_k: {top_k}")
        if top_k < 1:
            logger.warning("top_k must be at least 1, setting to 1")
            top_k = 1
        if not chunks or embeddings.shape[0] != len(chunks):
            logger.error("Invalid chunks or embeddings, cannot stream")
            return

        # Process query embedding
        query_inputs = self.tokenizer(
            query, return_tensors="np", padding=True, truncation=True, max_length=512
        )
        logger.debug(
            f"Query input IDs shape: {query_inputs['input_ids'].shape}, dtype: {query_inputs['input_ids'].dtype}")
        query_input_ids = mx.array(query_inputs["input_ids"]).astype(mx.int32)
        query_output = self.model(query_input_ids)
        logger.debug(
            f"Query output shape: {query_output.shape}, dtype: {query_output.dtype}")
        query_embedding = np.array(
            mx.mean(query_output, axis=1).tolist(), dtype=np.float32).squeeze()
        logger.debug(
            f"Query embedding shape: {query_embedding.shape}, dtype: {query_embedding.dtype}")

        # Normalize embeddings and query
        norm_embeddings = embeddings / \
            np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm_query = query_embedding / np.linalg.norm(query_embedding)
        logger.debug(
            f"Norm query shape: {norm_query.shape}, Norm embeddings shape: {norm_embeddings.shape}")

        # Compute similarities and get top-k indices
        similarities = np.dot(norm_embeddings, norm_query)
        logger.debug(f"Similarities shape: {similarities.shape}")
        top_indices = np.argsort(
            similarities)[-top_k:][::-1]  # Descending order

        # Stream responses for each top-k chunk
        for idx in top_indices:
            relevant_chunk = chunks[idx]
            similarity_score = similarities[idx]
            response = f"Based on chunk (score: {similarity_score:.4f}): {relevant_chunk[:100]}..., the answer is derived from the relevant information."
            logger.debug(
                f"Streaming response for chunk index {idx}, score: {similarity_score:.4f}")
            yield response

        logger.info("Streaming completed successfully")


def main():
    """Main function to demonstrate preprocessing and MLX RAG usage with streaming."""
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
        mlx_processor = MLXRAGProcessor(show_progress=True)
        logger.info("Generating embeddings for chunks")
        embeddings = mlx_processor.generate_embeddings(chunks)
        if embeddings.shape[0] != len(chunks):
            logger.error("Mismatch between chunks and embeddings, exiting")
            return
        query = "What is the main topic of the webpage?"
        logger.info(f"Processing query with generate: {query}")
        response = mlx_processor.generate(query, chunks, embeddings)
        print(f"Query: {query}")
        print(f"Single Response: {response}")
        print(f"Number of chunks processed: {len(chunks)}")
        logger.info(f"Processing query with stream_generate: {query}")
        print("\nStreaming Responses:")
        for i, stream_response in enumerate(mlx_processor.stream_generate(query, chunks, embeddings, top_k=3), 1):
            print(f"Stream Response {i}: {stream_response}")
        logger.info("Main function completed successfully")
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
