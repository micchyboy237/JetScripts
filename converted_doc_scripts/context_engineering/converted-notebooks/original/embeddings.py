# jet_python_modules/jet/adapters/langchain/langchain/embed_llama_cpp.py
from __future__ import annotations

from typing import Any, List, Literal, Union, Iterator, Callable, Optional

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

import numpy as np
from openai import OpenAI
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

GenerateEmbeddingsReturnType = Union[List[List[float]], np.ndarray]


class EmbedLlamaCpp(BaseModel, Embeddings):
    """
    LangChain embeddings wrapper for llama.cpp server.
    Returns list or numpy embeddings with fixed batch processing.
    """
    model_config = {"arbitrary_types_allowed": True}  # ADD THIS LINE

    model: str = Field(default="embeddinggemma", description="Embedding model name")
    base_url: str = Field(
        default="http://shawn-pc.local:8081/v1",
        description="Base URL of the llama.cpp embedding server",
    )
    max_retries: int = Field(default=3, ge=0, description="Max retries on HTTP failure")
    batch_size: int = Field(default=32, ge=1, description="Batch size for embedding calls")
    return_format: Literal["list", "numpy"] = Field(
        default="numpy", description="Output format: list of lists or numpy array"
    )
    show_progress: bool = Field(default=True, description="Show progress bar")
    verbose: bool = Field(default=False, description="Enable verbose logging")
    embedder: "LlamacppEmbedding" = Field(default=None, exclude=True, repr=False)

    def __init__(
        self,
        *,
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.verbose = verbose
        self.embedder = LlamacppEmbedding(
            model=self.model,
            base_url=self.base_url,
            max_retries=self.max_retries,
            verbose=self.verbose,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = self.embedder(
            inputs=texts,
            return_format=self.return_format,
            batch_size=self.batch_size,
            show_progress=self.show_progress,
        )
        return result if self.return_format == "list" else result.tolist()

    def embed_query(self, text: str) -> List[float]:
        result = self.embedder(
            inputs=text,
            return_format=self.return_format,
            batch_size=1,
            show_progress=False,
        )
        return result[0] if self.return_format == "list" else result[0].tolist()

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


class InputTooLargeError(ValueError):
    """Raised when input exceeds max_length."""
    def __init__(self, long_input_indexes: List[int], max_length: int):
        self.long_input_indexes = long_input_indexes
        self.max_length = max_length
        super().__init__(f"Inputs at indexes {long_input_indexes} exceed max_length ({max_length} tokens).")


class LlamacppEmbedding:
    """Simple client for llama.cpp embedding server using OpenAI API."""
    
    def __init__(
        self,
        model: str = "embeddinggemma",
        base_url: str = "http://shawn-pc.local:8081/v1",
        max_retries: int = 3,
        verbose: bool = False
    ):
        self.client = OpenAI(base_url=base_url, api_key="no-key-required", max_retries=max_retries)
        self.model = model
        self.verbose = verbose

    def __call__(
        self,
        inputs: Union[str, List[str]],
        return_format: Literal["numpy", "list"] = "numpy",
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> GenerateEmbeddingsReturnType:
        return self.get_embeddings(
            inputs,
            return_format=return_format,
            batch_size=batch_size,
            show_progress=show_progress,
        )

    def get_embeddings(
        self,
        inputs: Union[str, List[str]],
        return_format: Literal["numpy", "list"] = "numpy",
        batch_size: int = 8,
        show_progress: bool = True,
        max_length: int = 2048,
    ) -> GenerateEmbeddingsReturnType:
        input_list = [inputs] if isinstance(inputs, str) else inputs
        valid_inputs = [i for i in input_list if isinstance(i, str) and i.strip()]
        if not valid_inputs:
            raise ValueError("inputs must be a non-empty string or list of non-empty strings")

        # Basic token length check (fallback to max_length)
        long_inputs = [idx for idx, txt in enumerate(valid_inputs) if len(txt.split()) > max_length]
        if long_inputs:
            raise InputTooLargeError(long_inputs, max_length)

        embeddings = []
        progress_bar = tqdm(range(0, len(valid_inputs), batch_size), desc="Embedding", disable=not show_progress)

        for i in progress_bar:
            batch = valid_inputs[i:i + batch_size]
            try:
                resp = self.client.embeddings.create(model=self.model, input=batch)
                batch_emb = [d.embedding for d in resp.data]
                if return_format == "numpy":
                    batch_emb = [np.array(e, dtype=np.float32) for e in batch_emb]
                embeddings.extend(batch_emb)
            except Exception as e:
                logger.error(f"Batch {i // batch_size + 1} failed: {e}")
                raise

        return embeddings if return_format == "list" else np.array(embeddings, dtype=np.float32)

    def get_embedding_function(
        self,
        return_format: Literal["numpy", "list"] = "numpy",
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> Callable[[Union[str, List[str]]], GenerateEmbeddingsReturnType]:
        """Return callable embedding function."""
        def embedding_function(inputs: Union[str, List[str]]) -> GenerateEmbeddingsReturnType:
            return self.get_embeddings(
                inputs,
                return_format=return_format,
                batch_size=batch_size,
                show_progress=show_progress,
            )
        return embedding_function

    def get_embeddings_stream(
        self,
        inputs: Union[str, List[str]],
        return_format: Literal["numpy", "list"] = "numpy",
        batch_size: int = 8,
        show_progress: bool = True,
        max_input_length: Optional[int] = None,
    ) -> Iterator[GenerateEmbeddingsReturnType]:
        """Stream embeddings in batches."""
        input_list = [inputs] if isinstance(inputs, str) else inputs
        valid_inputs = [i for i in input_list if isinstance(i, str) and i.strip()]
        
        if not valid_inputs:
            raise ValueError("inputs must be a non-empty string or list of non-empty strings")

        # Simplified length check (no token counting)
        max_length = max_input_length or 2048
        long_inputs = [idx for idx, txt in enumerate(valid_inputs) if len(txt.split()) > max_length]
        if long_inputs:
            raise InputTooLargeError(long_inputs, max_length)
        
        progress_bar = tqdm(range(0, len(valid_inputs), batch_size), desc="Streaming batches", disable=not show_progress)
        
        for i in progress_bar:
            batch = valid_inputs[i:i + batch_size]
            try:
                response = self.client.embeddings.create(model=self.model, input=batch)
                batch_embeddings = [d.embedding for d in response.data]
                if return_format == "numpy":
                    batch_embeddings = np.array(batch_embeddings, dtype=np.float32)
                yield batch_embeddings
            except Exception as e:
                logger.error(f"Error in stream batch {i // batch_size + 1}: {e}")
                raise
