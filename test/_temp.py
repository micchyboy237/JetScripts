import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize
import nltk
import re
from typing import List, Tuple, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('punkt')


class VectorSearchWeb:
    def __init__(self, embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
                 max_context_size: int = 512):
        """Initialize with embedding model, cross-encoder, and context size."""
        self.embed_model = SentenceTransformer(embed_model_name)
        self.cross_encoder = CrossEncoder(cross_encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
        self.max_context_size = max_context_size
        self.index = None
        self.chunk_metadata = []  # (doc_id, chunk_text, chunk_idx, header)
        logger.info("Initialized with model %s, context size %d",
                    embed_model_name, max_context_size)

    def preprocess_web_document(self, text: str) -> List[Tuple[str, str]]:
        """Split web-scraped text into header-content pairs, with noise filtering."""
        text = re.sub(r'(<script.*?</script>|<style.*?</style>|<!--.*?-->)',
                      '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()

        header_pattern = r'(<h[1-6]>.*?</h[1-6]>|\#\#+\s+[^\n]+)'
        sections = []
        current_header = "No Header"
        current_content = []

        parts = re.split(header_pattern, text, flags=re.IGNORECASE)
        for i in range(0, len(parts), 2):
            content = parts[i].strip()
            if content:
                current_content.append(content)
            if i + 1 < len(parts):
                next_header = parts[i + 1].strip()
                if current_content:
                    sections.append(
                        (current_header, '\n'.join(current_content)))
                    current_content = []
                current_header = re.sub(
                    r'<[^>]+>|\#+', '', next_header).strip()

        if current_content:
            sections.append((current_header, '\n'.join(current_content)))

        sections = [(h, c) for h, c in sections if len(c.strip()) > 50 and not re.match(
            r'^(Navigation|Footer|Copyright|Login|Advertisement|Sidebar)', h, re.IGNORECASE)]
        return sections

    def chunk_document(self, text: str, doc_id: str, chunk_size: int, overlap: int) -> List[Tuple[str, str, int, str]]:
        """Chunk document, respecting headers and context size."""
        if chunk_size > self.max_context_size:
            logger.warning("Chunk size %d exceeds context size %d; capping at %d",
                           chunk_size, self.max_context_size, self.max_context_size)
            chunk_size = self.max_context_size

        sections = self.preprocess_web_document(text)
        chunks = []
        chunk_idx = 0

        for header, content in sections:
            section_text = f"{header}\n{content}" if header != "No Header" else content
            section_tokens = len(self.tokenizer.encode(
                section_text, add_special_tokens=False))

            if section_tokens <= chunk_size:
                chunks.append((doc_id, section_text, chunk_idx, header))
                chunk_idx += 1
            else:
                token_chunks = self._chunk_by_tokens(
                    section_text, doc_id, chunk_size, overlap, chunk_idx, header)
                chunks.extend(token_chunks)
                chunk_idx += len(token_chunks)

        return chunks

    def _chunk_by_tokens(self, text: str, doc_id: str, chunk_size: int, overlap: int,
                         start_idx: int, header: str) -> List[Tuple[str, str, int, str]]:
        """Chunk text by tokens with overlap."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        i = 0
        chunk_idx = start_idx

        while i < len(tokens):
            end_idx = min(i + chunk_size, len(tokens))
            chunk_tokens = tokens[i:end_idx]
            chunk_text = self.tokenizer.decode(
                chunk_tokens, skip_special_tokens=True)
            chunks.append((doc_id, chunk_text, chunk_idx, header))
            chunk_idx += 1
            i += chunk_size - overlap

        return chunks

    def index_documents(self, documents: List[Tuple[str, str]], chunk_sizes: List[int], overlap_ratio: float = 0.2):
        """Index documents with multiple chunk sizes."""
        all_chunks = []
        for doc_id, text in documents:
            for chunk_size in chunk_sizes:
                overlap = int(chunk_size * overlap_ratio)
                chunks = self.chunk_document(text, doc_id, chunk_size, overlap)
                all_chunks.extend(chunks)

        chunk_texts = [chunk[1] for chunk in all_chunks]
        embeddings = self.embed_model.encode(
            chunk_texts, show_progress_bar=True)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunk_metadata = all_chunks
        logger.info(f"Indexed {len(all_chunks)} chunks with dimension %d", dim)

    def search(self, query: str, k: int = 5, use_cross_encoder: bool = True, query_type: str = "short") -> List[Tuple[str, str, int, str, float]]:
        """Search with deduplication to reduce redundant neighbors."""
        chunk_size_preference = 150 if query_type == "short" else 250
        query_embedding = self.embed_model.encode(
            [query], show_progress_bar=False)[0]
        faiss.normalize_L2(query_embedding.reshape(1, -1))

        distances, indices = self.index.search(
            query_embedding.reshape(1, -1), k * 2)
        candidates = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.chunk_metadata):
                candidates.append(self.chunk_metadata[idx] + (score,))

        if use_cross_encoder:
            pairs = [[query, chunk[1]] for chunk in candidates]
            scores = self.cross_encoder.predict(pairs)
            candidates = [(c[0], c[1], c[2], c[3], s)
                          for c, s in zip(candidates, scores)]
            candidates = sorted(candidates, key=lambda x: x[4], reverse=True)

        seen_headers = {}
        deduped = []
        for candidate in candidates:
            doc_id, _, _, header, score = candidate
            key = (doc_id, header)
            if key not in seen_headers or score > seen_headers[key][4]:
                seen_headers[key] = candidate

        candidates = list(seen_headers.values())
        candidates = sorted(candidates, key=lambda x: (
            abs(len(self.tokenizer.encode(
                x[1], add_special_tokens=False)) - chunk_size_preference),
            -x[4]
        ))[:k]

        return candidates

    def evaluate_models(self, documents: List[Tuple[str, str]],
                        validation_set: List[Tuple[str, List[Tuple[str, int]]]],
                        model_names: List[str], chunk_sizes: List[int],
                        overlap_ratio: float = 0.2, k: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Evaluate embedding models using precision@k, recall@k, and MR Khrushchev@K.
        validation_set: List of (query, [(doc_id, chunk_idx), ...]) pairs.
        Returns model_name -> {'precision': float, 'recall': float, 'mrr': float}.
        """
        results = {}
        original_model = self.embed_model.model_card_data.get(
            'model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        original_tokenizer = self.tokenizer
        original_context_size = self.max_context_size

        for model_name in model_names:
            logger.info("Evaluating model: %s", model_name)
            self.embed_model = SentenceTransformer(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.max_context_size = 512 if 'MiniLM' in model_name else 512  # Adjust if known

            self.index_documents(
                documents, chunk_sizes=chunk_sizes, overlap_ratio=overlap_ratio)

            precision_sum = 0.0
            recall_sum = 0.0
            mrr_sum = 0.0
            total_queries = len(validation_set)

            for query, relevant_chunks in validation_set:
                query_type = "short" if len(self.tokenizer.encode(
                    query, add_special_tokens=False)) < 50 else "long"
                results = self.search(query, k=k, query_type=query_type)

                relevant_set = set((doc_id, chunk_idx)
                                   for doc_id, chunk_idx in relevant_chunks)
                retrieved_set = set((doc_id, chunk_idx)
                                    for doc_id, _, chunk_idx, _, _ in results)

                # Precision@k
                precision = len(relevant_set & retrieved_set) / \
                    k if k > 0 else 0.0
                precision_sum += precision

                # Recall@k
                recall = len(relevant_set & retrieved_set) / \
                    len(relevant_set) if relevant_set else 0.0
                recall_sum += recall

                # MRR
                mrr = 0.0
                for rank, (doc_id, _, chunk_idx, _, _) in enumerate(results, 1):
                    if (doc_id, chunk_idx) in relevant_set:
                        mrr = 1.0 / rank
                        break
                mrr_sum += mrr

            results[model_name] = {
                'precision': precision_sum / total_queries if total_queries else 0.0,
                'recall': recall_sum / total_queries if total_queries else 0.0,
                'mrr': mrr_sum / total_queries if total_queries else 0.0
            }
            logger.info("Model %s: Precision@%d=%.4f, Recall@%d=%.4f, MRR=%.4f",
                        model_name, k, results[model_name]['precision'],
                        k, results[model_name]['recall'], results[model_name]['mrr'])

        self.embed_model = SentenceTransformer(original_model)
        self.tokenizer = original_tokenizer
        self.max_context_size = original_context_size
        self.index = None
        self.chunk_metadata = []
        return results

    def evaluate_retrieval_examples(self, documents: List[Tuple[str, str]],
                                    example_queries: List[Tuple[str, List[Tuple[str, int]]]],
                                    chunk_sizes: List[int], overlap_ratio: float = 0.2, k: int = 5) -> Dict[str, List[Dict]]:
        """
        Evaluate retrieval performance for example queries.
        example_queries: List of (query, [(doc_id, chunk_idx), ...]) pairs.
        Returns query -> [{'doc_id': str, 'chunk_idx': int, 'header': str, 'score': float, 'text': str, 'is_relevant': bool}, ...].
        """
        self.index_documents(
            documents, chunk_sizes=chunk_sizes, overlap_ratio=overlap_ratio)
        results = {}

        for query, relevant_chunks in example_queries:
            query_type = "short" if len(self.tokenizer.encode(
                query, add_special_tokens=False)) < 50 else "long"
            search_results = self.search(query, k=k, query_type=query_type)
            relevant_set = set((doc_id, chunk_idx)
                               for doc_id, chunk_idx in relevant_chunks)

            result_list = []
            for doc_id, chunk_text, chunk_idx, header, score in search_results:
                is_relevant = (doc_id, chunk_idx) in relevant_set
                result_list.append({
                    'doc_id': doc_id,
                    'chunk_idx': chunk_idx,
                    'header': header,
                    'score': score,
                    'text': chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text,
                    'is_relevant': is_relevant
                })

            results[query] = result_list

        return results


# Example Usage
if __name__ == "__main__":
    documents = [
        ("doc1", "<h1>Introduction</h1>\nShort intro text.\n<h2>Details</h2>\nSome details here."),
        ("doc2", "<h1>Main Topic</h1>\n" + "Content about the topic. " *
         200 + "\n<h2>Subtopic</h2>\n" + "More details. " * 50),
        ("doc3", "<h1>Overview</h1>\n" + "Overview content. " *
         500 + "\n<h2>Conclusion</h2>\n" + "Final thoughts. " * 100)
    ]

    # Validation set: (query, [(doc_id, chunk_idx), ...])
    validation_set = [
        ("What is the main topic?", [("doc2", 0), ("doc2", 1)]),
        ("Explain the structure and content of the documents in detail.",
         [("doc3", 0), ("doc3", 1)])
    ]

    example_queries = [
        ("What is the main topic of AI?", [("doc2", 0), ("doc2", 1)]),
        ("Explain AI technologies in detail.", [("doc3", 0), ("doc3", 1)])
    ]

    searcher = VectorSearchWeb(max_context_size=512)
    chunk_sizes = [150, 250, 350]

    # Evaluate models
    model_names = ["sentence-transformers/all-MiniLM-L6-v2",
                   "sentence-transformers/all-mpnet-base-v2"]
    model_scores = searcher.evaluate_models(
        documents, validation_set, model_names, chunk_sizes, overlap_ratio=0.2, k=3)
    best_model = max(model_scores, key=model_scores.get)
    logger.info("Best model: %s with Precision@3 = %.4f",
                best_model, model_scores[best_model])

    # Re-initialize with best model
    searcher = VectorSearchWeb(
        embed_model_name=best_model, max_context_size=512)
    searcher.index_documents(
        documents, chunk_sizes=chunk_sizes, overlap_ratio=0.2)

    # Test queries
    short_query = "What is the main topic?"
    long_query = "Explain the structure and content of the documents in detail."

    results_short = searcher.search(short_query, k=3, query_type="short")
    results_long = searcher.search(long_query, k=3, query_type="long")

    print("Short Query Results:")
    for doc_id, chunk_text, chunk_idx, header, score in results_short:
        print(
            f"Doc: {doc_id}, Chunk: {chunk_idx}, Header: {header}, Score: {score:.4f}, Text: {chunk_text[:100]}...")

    print("\nLong Query Results:")
    for doc_id, chunk_text, chunk_idx, header, score in results_long:
        print(
            f"Doc: {doc_id}, Chunk: {chunk_idx}, Header: {header}, Score: {score:.4f}, Text: {chunk_text[:100]}...")

    # Re-initialize with best model
    searcher = VectorSearchWeb(
        embed_model_name=best_model, max_context_size=512)

    # Evaluate retrieval performance examples
    example_results = searcher.evaluate_retrieval_examples(
        documents, example_queries, chunk_sizes, overlap_ratio=0.2, k=3)

    print("\nRetrieval Performance Examples:")
    for query, results in example_results.items():
        print(f"\nQuery: {query}")
        for result in results:
            print(f"Doc: {result['doc_id']}, Chunk: {result['chunk_idx']}, Header: {result['header']}, "
                  f"Score: {result['score']:.4f}, Relevant: {result['is_relevant']}, Text: {result['text']}")
