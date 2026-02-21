#!/bin/bash

# create_structure.sh - Set up project directory and scaffold all files with the provided code.

set -e

# Create directory structure
mkdir -p tests

# Create requirements.txt
cat > requirements.txt << 'EOF'
unstructured[all-docs]
sentence-transformers
chromadb
rich
tqdm
pytest
# Optional for Windows/Mac deps: tesseract, poppler (system packages via brew/choco)
EOF

# Create retriever.py
cat > retriever.py << 'EOF'
"""Reusable, modular, local-only unstructured data processor/retriever.

Mirrors blog techniques (element partitioning + rich metadata + embeddings + metadata filtering)
but fully local with ChromaDB. Generic, configurable, no business logic.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm
from unstructured.partition.auto import partition

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console)],
)
logger = logging.getLogger(__name__)


@dataclass
class RetrieverConfig:
    """Generic configuration - user supplies values at runtime."""
    collection_name: str
    persist_directory: str
    embedding_model_name: str = "all-MiniLM-L6-v2"


class UnstructuredIngester:
    """Single responsibility: partition local files into Unstructured elements."""

    @staticmethod
    def partition_file(file_path: Path) -> List[Any]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        logger.info(f"Partitioning {file_path.name}")
        return partition(filename=str(file_path))

    @staticmethod
    def get_files_in_directory(directory: Path, pattern: str = "**/*.*") -> List[Path]:
        return [f for f in directory.glob(pattern) if f.is_file()]


class LocalVectorStore:
    """Single responsibility: persistent local Chroma storage with metadata filtering."""

    def __init__(self, persist_directory: str, collection_name: str, embedding_model_name: str):
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_elements(self, elements: List[Any], source_file: str) -> None:
        """Prepare elements (text + enriched metadata) and upsert."""
        if not elements:
            return
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []
        for idx, element in enumerate(elements):
            text = getattr(element, "text", "").strip()
            if not text:
                continue
            documents.append(text)
            meta: Dict[str, Any] = {}
            if hasattr(element, "metadata") and hasattr(element.metadata, "to_dict"):
                meta.update(element.metadata.to_dict())
            meta["element_type"] = getattr(element, "category", "Unknown")
            meta["source_file"] = source_file
            metadatas.append(meta)
            ids.append(f"{source_file}_{idx}")
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        logger.info(f"Added {len(documents)} elements from {source_file}")

    def query(
        self, query_text: str, n_results: int = 5, where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Metadata-filtered retrieval (exact blog-style 'element_type' filter support)."""
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )


class UnstructuredLocalRetriever:
    """High-level facade - easy entry point for users."""

    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.ingester = UnstructuredIngester()
        self.vector_store = LocalVectorStore(
            config.persist_directory, config.collection_name, config.embedding_model_name
        )

    def ingest_file(self, file_path: str | Path) -> None:
        path = Path(file_path)
        elements = self.ingester.partition_file(path)
        self.vector_store.add_elements(elements, source_file=path.name)
        console.print(f"[bold green]✓ Ingested {path.name}[/bold green]")

    def ingest_directory(self, directory: str | Path, pattern: str = "**/*.*") -> None:
        dir_path = Path(directory)
        files = self.ingester.get_files_in_directory(dir_path, pattern)
        for file_path in tqdm(files, desc="Ingesting local files", unit="file"):
            try:
                self.ingest_file(file_path)
            except Exception as e:  # robust per-file
                logger.error(f"Skipped {file_path.name}: {e}")

    def retrieve(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Example filter: {'element_type': 'Table'} or {'source_file': 'report.pdf'}."""
        return self.vector_store.query(query, top_k, filters)


if __name__ == "__main__":
    # Quick example usage (uncomment/adapt)
    config = RetrieverConfig(collection_name="my_local_docs", persist_directory="./chroma_db")
    retriever = UnstructuredLocalRetriever(config)
    # retriever.ingest_directory("./my_documents_folder")
    # results = retriever.retrieve("sales figures", top_k=3, filters={"element_type": "Table"})
    # console.print(results)
EOF

# Create tests/test_retriever.py
cat > tests/test_retriever.py << 'EOF'
"""pytest tests following your exact style (BDD, result/expected vars, full list asserts, tmp_path cleanup)."""
import pytest
from pathlib import Path
from retriever import UnstructuredLocalRetriever, RetrieverConfig


@pytest.fixture
def temp_retriever_dir(tmp_path: Path):
    """Clean temporary dir + DB for each test."""
    return tmp_path


class TestUnstructuredLocalRetriever:
    """Separate test class for behaviors."""

    def test_ingest_text_file_and_retrieve(self, temp_retriever_dir: Path):
        """BDD: Given a real-world quarterly sales report text file.
        When ingested.
        Then retrieve returns exact expected content with metadata.
        """
        # Given: real-world sample (like blog PDF excerpt but .txt for easy testing)
        sample_content = (
            "Q3 2025 Sales Report\n"
            "Revenue: $2.3M (up 20%).\n"
            "Key metric: tables show regional breakdown."
        )
        sample_file = temp_retriever_dir / "q3_sales_report.txt"
        sample_file.write_text(sample_content)

        config = RetrieverConfig(
            collection_name="test_sales",
            persist_directory=str(temp_retriever_dir / "chroma"),
        )
        retriever = UnstructuredLocalRetriever(config)

        # When
        retriever.ingest_file(sample_file)

        # Then
        result = retriever.retrieve("sales revenue", top_k=1)
        expected_documents = [[sample_content]]
        expected_metadata_contains = {
            "element_type": "NarrativeText",
            "source_file": "q3_sales_report.txt",
        }

        assert result["documents"] == expected_documents
        assert expected_metadata_contains["element_type"] in result["metadatas"][0][0]["element_type"]
        assert expected_metadata_contains["source_file"] == result["metadatas"][0][0]["source_file"]

    def test_ingest_directory_and_filtered_retrieve(self, temp_retriever_dir: Path):
        """BDD: Given a directory with multiple real-world doc files.
        When ingested via directory method.
        Then filtered query returns only matching elements.
        """
        # Given: two sample files (realistic content)
        (temp_retriever_dir / "tech_article.txt").write_text(
            "Advanced RAG uses element partitioning and metadata filtering."
        )
        (temp_retriever_dir / "finance_table.txt").write_text(
            "Table: Q3 Revenue by region."
        )

        config = RetrieverConfig(
            collection_name="test_dir",
            persist_directory=str(temp_retriever_dir / "chroma_dir"),
        )
        retriever = UnstructuredLocalRetriever(config)

        # When
        retriever.ingest_directory(temp_retriever_dir, pattern="**/*.txt")

        # Then: filter demo (blog-style element_type filter)
        result = retriever.retrieve(
            "revenue", top_k=2, filters={"element_type": "NarrativeText"}
        )
        result_texts = result["documents"][0]
        expected_texts = [
            "Advanced RAG uses element partitioning and metadata filtering."
        ]  # only the narrative one after filter

        assert result_texts == expected_texts  # exact list assert
        assert result["metadatas"][0][0]["source_file"] == "tech_article.txt"
EOF

echo "Project scaffold complete."