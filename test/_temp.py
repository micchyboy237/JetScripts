from typing import List, TypedDict
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import torch


class NodeResult(TypedDict):
    text: str
    tag: str
    similarity: float


class HTMLNodeSearcher:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def extract_node_texts(self, html_content: str, target_tag: str = 'p') -> List[dict]:
        """
        Extract text from specified HTML nodes.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        nodes = soup.find_all(target_tag)
        return [{'text': node.get_text(strip=True), 'tag': node.name} for node in nodes if node.get_text(strip=True)]

    def chunk_text(self, text: str, max_chunk_size: int = 512) -> List[str]:
        """
        Split long text into chunks based on character length.
        """
        if len(text) <= max_chunk_size:
            return [text]

        chunks = []
        for i in range(0, len(text), max_chunk_size):
            chunks.append(text[i:i + max_chunk_size])
        return chunks

    def search_nodes(self, query: str, html_content: str, target_tag: str = 'p', top_k: int = 3) -> List[NodeResult]:
        """
        Search HTML nodes by semantic similarity to the query.
        """
        # Extract node texts
        nodes = self.extract_node_texts(html_content, target_tag)
        if not nodes:
            return []

        # Encode query
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        results: List[NodeResult] = []
        for node in nodes:
            # Chunk long node text
            chunks = self.chunk_text(node['text'])
            if not chunks:
                continue

            # Encode chunks
            chunk_embeddings = self.model.encode(
                chunks, convert_to_tensor=True)
            similarities = util.cos_sim(query_embedding, chunk_embeddings)[0]

            # Take the highest similarity score for the node
            max_score = similarities.max().item()
            results.append(
                {'text': node['text'], 'tag': node['tag'], 'similarity': max_score})

        # Sort by similarity and return top-k
        return sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_k]
