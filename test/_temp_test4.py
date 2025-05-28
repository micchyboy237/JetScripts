import urllib.parse
import requests
from bs4 import BeautifulSoup
import numpy as np
import mlx.core as mx
from sentence_transformers import SentenceTransformer


class AnimeAvailabilityTool:
    def __init__(self):
        # Load a lightweight embedding model for vector search, optimized for MPS
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device='mps')

    def get_md_header_docs(self, url):
        """Scrape HTML header documents from the given URL."""
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract headers (h1, h2, h3) and meta description
            headers = [elem.get_text().strip()
                       for elem in soup.find_all(['h1', 'h2', 'h3'])]
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            meta_content = meta_desc['content'].strip() if meta_desc else ""
            return headers + [meta_content] if meta_content else headers
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return []

    def search_docs(self, query, documents):
        """Perform vector search with MPS to find document similarity."""
        if not documents:
            return 0.0
        # Generate embeddings for query and documents
        query_embedding = self.embedder.encode([query], convert_to_numpy=False)
        doc_embeddings = self.embedder.encode(
            documents, convert_to_numpy=False)
        # Convert to MLX arrays for MPS
        query_embedding = mx.array(query_embedding)
        doc_embeddings = mx.array(doc_embeddings)
        # Compute cosine similarity
        dot_product = mx.sum(query_embedding * doc_embeddings, axis=1)
        norm_query = mx.sqrt(mx.sum(query_embedding ** 2))
        norm_docs = mx.sqrt(mx.sum(doc_embeddings ** 2, axis=1))
        similarities = dot_product / (norm_query * norm_docs + 1e-8)
        return float(mx.max(similarities).item())

    def validate_anime(self, anime_titles, threshold=0.8):
        """Validate if anime titles exist on AniWatch with high similarity."""
        results = {}
        for title in anime_titles:
            encoded_title = urllib.parse.quote(title)
            url = f"https://aniwatchtv.to/search?keyword={encoded_title}"
            documents = self.get_md_header_docs(url)
            score = self.search_docs(title, documents)
            results[title] = {
                'url': url,
                'exists': score >= threshold,
                'score': score
            }
        return results
