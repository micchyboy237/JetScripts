import sqlite3
from pathlib import Path
import json
from typing import Dict, List, TypedDict, Literal
import numpy as np
from datetime import datetime
import re
from openai import OpenAI  # Assuming OpenAI-compatible API (e.g., Nebius AI)


class QueryResult(TypedDict):
    query: str
    retrieved_chunks: List[str]
    response: str
    timestamp: str


class PersistentMemoryStore:
    """A persistent memory store using SQLite to consolidate query results."""

    def __init__(self, db_path: str = "rag_memory.db"):
        """Initialize SQLite database for storing query results."""
        self.db_path = Path(db_path)
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Create the database and results table if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    retrieved_chunks TEXT NOT NULL,
                    response TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.commit()

    def add_result(self, query: str, retrieved_chunks: List[str], response: str, timestamp: str) -> None:
        """Add a query result to the persistent store."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO query_results (query, retrieved_chunks, response, timestamp) VALUES (?, ?, ?, ?)",
                (query, json.dumps(retrieved_chunks), response, timestamp)
            )
            conn.commit()

    def get_consolidated_results(self, query: str) -> List[QueryResult]:
        """Retrieve all results for a query from the persistent store."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT query, retrieved_chunks, response, timestamp FROM query_results WHERE query = ?", (query,))
            results = [
                QueryResult(
                    query=row[0],
                    retrieved_chunks=json.loads(row[1]),
                    response=row[2],
                    timestamp=row[3]
                )
                for row in cursor.fetchall()
            ]
        return results

    def get_all_results(self) -> Dict[str, List[QueryResult]]:
        """Retrieve all stored results, grouped by query."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT query, retrieved_chunks, response, timestamp FROM query_results")
            results = defaultdict(list)
            for row in cursor.fetchall():
                result = QueryResult(
                    query=row[0],
                    retrieved_chunks=json.loads(row[1]),
                    response=row[2],
                    timestamp=row[3]
                )
                results[row[0]].append(result)
        return dict(results)


class VectorStore:
    """A simple vector store for managing document chunks and embeddings."""

    def __init__(self):
        self.chunks: List[Dict[str, str]] = []
        self.embeddings: np.ndarray = None

    def add_chunks(self, chunks: List[Dict[str, str]], embeddings: List[List[float]]) -> None:
        """Add chunks and their embeddings to the store."""
        self.chunks.extend(chunks)
        if self.embeddings is None:
            self.embeddings = np.array(embeddings)
        else:
            self.embeddings = np.vstack(
                [self.embeddings, np.array(embeddings)])

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, str]]:
        """Search for top_k most similar chunks using cosine similarity."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
        query_embedding = np.array(query_embedding)
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) *
            np.linalg.norm(query_embedding)
        )
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices]


def extract_headers_and_text(web_content: str) -> List[Dict[str, str]]:
    """Extract headers and associated text from web content."""
    # Simple regex-based header extraction (e.g., <h1>, <h2>, etc.)
    header_pattern = r'<h[1-6][^>]*>(.*?)</h[1-6]>'
    headers = re.findall(header_pattern, web_content, re.DOTALL)
    chunks = []
    # Skip initial content before first header
    content_sections = re.split(header_pattern, web_content)[1:]
    for i in range(0, len(content_sections), 2):
        header = content_sections[i].strip()
        text = content_sections[i + 1].strip() if i + \
            1 < len(content_sections) else ""
        # Clean HTML tags from text
        text = re.sub(r'<[^>]+>', '', text).strip()
        if header and text:
            # Limit text length
            chunks.append({"header": header, "text": text[:1000]})
    return chunks


def chunk_text_with_headers(chunks: List[Dict[str, str]], chunk_size: int = 256, overlap: int = 50) -> List[Dict[str, str]]:
    """Split text into overlapping chunks, preserving headers."""
    result = []
    for chunk in chunks:
        text = chunk["text"]
        header = chunk["header"]
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk_text = " ".join(words[i:i + chunk_size])
            if chunk_text.strip():
                result.append({"header": header, "text": chunk_text})
    return result


def generate_embeddings(texts: List[str], client: OpenAI, model: str = "text-embedding-ada-002") -> List[List[float]]:
    """Generate embeddings for a list of texts using the OpenAI client."""
    embeddings = []
    for text in texts:
        response = client.embeddings.create(input=text, model=model)
        embeddings.append(response.data[0].embedding)
    return embeddings


def process_query_with_iterative_memory(
    query: str,
    web_content: str,
    client: OpenAI,
    memory_store: PersistentMemoryStore,
    timestamp: str = None,
    max_iterations: int = 3,
    relevance_threshold: float = 0.8,
    chunk_size: int = 256,
    overlap: int = 50,
    top_k: int = 5
) -> str:
    """Process a query iteratively, leveraging headers and memory for refined retrieval."""
    # Initialize vector store
    vector_store = VectorStore()

    # Extract headers and text from web content
    raw_chunks = extract_headers_and_text(web_content)

    # Chunk text with headers
    chunks = chunk_text_with_headers(raw_chunks, chunk_size, overlap)

    # Generate embeddings for chunks (prepend header to text for context)
    texts = [f"{chunk['header']}: {chunk['text']}" for chunk in chunks]
    embeddings = generate_embeddings(texts, client)

    # Add chunks and embeddings to vector store
    vector_store.add_chunks(chunks, embeddings)

    # Iterative processing
    best_response = None
    best_score = -1.0
    best_chunks = []

    for iteration in range(max_iterations):
        # Retrieve past chunks from memory
        past_results = memory_store.get_consolidated_results(query)
        past_chunks = [
            chunk for result in past_results for chunk in result["retrieved_chunks"]]

        # Retrieve new chunks
        query_embedding = generate_embeddings([query], client)[0]
        retrieved_chunks = vector_store.search(query_embedding, top_k)
        combined_chunks = list(
            set(past_chunks + [chunk["text"] for chunk in retrieved_chunks]))

        # Generate context with headers
        context = "\n".join(
            [f"{chunk['header']}: {chunk['text']}" for chunk in retrieved_chunks])

        # Generate response
        prompt = f"Context:\n{context}\n\nQuery: {query}\nAnswer:"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Replace with Nebius AI model if needed
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        ).choices[0].message.content

        # Evaluate response relevance (cosine similarity)
        response_embedding = generate_embeddings([response], client)[0]
        query_embedding = generate_embeddings([query], client)[0]
        similarity = np.dot(response_embedding, query_embedding) / (
            np.linalg.norm(response_embedding) *
            np.linalg.norm(query_embedding)
        )

        # Update best response
        if similarity > best_score:
            best_score = similarity
            best_response = response
            best_chunks = [chunk["text"] for chunk in retrieved_chunks]

        # Break if relevance is sufficient
        if similarity >= relevance_threshold:
            break

    # Store result in memory
    timestamp = timestamp or datetime.now().isoformat()
    memory_store.add_result(query, best_chunks, best_response, timestamp)
    return best_response


# Example usage
if __name__ == "__main__":
    # Initialize OpenAI client (replace with Nebius AI credentials)
    client = OpenAI(api_key="your-api-key",
                    base_url="https://api.nebius.ai/v1")

    # Sample web content (simulated)
    web_content = """
    <h1>About France</h1>
    <p>France is a country in Western Europe known for its culture and history. Its capital is Paris.</p>
    <h2>History</h2>
    <p>France has a rich history dating back to the Roman Empire.</p>
    """

    # Initialize memory store
    memory_store = PersistentMemoryStore()

    # Process query
    query = "What is the capital of France?"
    response = process_query_with_iterative_memory(
        query, web_content, client, memory_store)
    print(f"Query: {query}\nResponse: {response}")
