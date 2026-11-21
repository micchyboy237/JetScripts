"""
Query-Aware Word Group Extraction using Jet Adapters
=================================
Extends the previous approach to find word groups relevant to a specific query.
Uses Semantic Search + MMR for diversity with local llama.cpp embeddings.
"""

import warnings
from collections import Counter

import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# Import Jet Adapter for Embeddings
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.embed_utils import embed

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


class QueryAwareWordGroupExtractor:
    def __init__(self, model_name=EMBED_MODEL):
        print(f"Using embedding model: {model_name}")
        self.model_name = model_name

    def preprocess_text(self, text):
        """Split text into sentences and clean."""
        from nltk.corpus import stopwords
        from nltk.tokenize import sent_tokenize, word_tokenize

        stop_words = set(stopwords.words("english"))
        stop_words.update([".", ",", "!", "?", ";", ":", '"', "'", "(", ")"])

        sentences = sent_tokenize(text)
        processed = []
        for sent in sentences:
            words = word_tokenize(sent.lower())
            filtered = [
                w for w in words if w not in stop_words and len(w) > 2 and w.isalpha()
            ]
            if filtered:
                processed.append({"original": sent, "tokens": filtered})
        return processed

    def extract_chunks(self, documents, chunk_size=3):
        """
        Extract small chunks (groups of words) from documents.
        """
        all_chunks = []
        for doc in documents:
            sentences = self.preprocess_text(doc)
            for i in range(0, len(sentences), chunk_size):
                group = sentences[i : i + chunk_size]
                if group:
                    combined_tokens = []
                    combined_text = " ".join([s["original"] for s in group])
                    for s in group:
                        combined_tokens.extend(s["tokens"])
                    all_chunks.append(
                        {
                            "text": combined_text,
                            "tokens": combined_tokens,
                            "source_sentences": [s["original"] for s in group],
                        }
                    )
        return all_chunks

    def maximal_marginal_relevance(
        self, query_embedding, chunk_embeddings, chunks, lambda_mult=0.5, top_k=5
    ):
        """
        Select chunks that are both relevant to the query and diverse among themselves.
        """
        query_similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]

        selected_indices = []
        remaining_indices = list(range(len(chunks)))

        # Pick the first item: the one most similar to the query
        best_idx = np.argmax(query_similarities)
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

        while len(selected_indices) < top_k and remaining_indices:
            best_score = -np.inf
            best_idx = None

            for idx in remaining_indices:
                # Relevance score: similarity to query
                relevance = query_similarities[idx]

                # Diversity score: max similarity to already selected items
                selected_embeds = chunk_embeddings[selected_indices]
                current_embed = chunk_embeddings[idx].reshape(1, -1)

                sims_to_selected = cosine_similarity(current_embed, selected_embeds)[0]
                max_sim_to_selected = (
                    np.max(sims_to_selected) if len(selected_indices) > 0 else 0
                )

                # MMR Score
                mmr_score = (
                    lambda_mult * relevance - (1 - lambda_mult) * max_sim_to_selected
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

        return selected_indices

    def search_and_group(self, query, documents, top_k=5, chunk_size=2):
        """
        Main method: Find word groups relevant to the query.
        """
        print(f"\nProcessing query: '{query}'")

        print("Extracting semantic chunks...")
        chunks = self.extract_chunks(documents, chunk_size=chunk_size)
        chunk_texts = [c["text"] for c in chunks]

        if not chunk_texts:
            return []

        print("Generating embeddings...")
        # Use jet adapter for embedding
        query_embedding = embed(query, model=self.model_name, return_format="numpy")

        # Normalize query embedding
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        chunk_embeddings = embed(
            chunk_texts,
            model=self.model_name,
            return_format="numpy",
            show_progress=False,
        )

        # Normalize chunk embeddings
        norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        chunk_embeddings = chunk_embeddings / norms

        print("Ranking by relevance and diversity (MMR)...")
        selected_indices = self.maximal_marginal_relevance(
            query_embedding, chunk_embeddings, chunks, top_k=top_k
        )

        results = []
        for idx in selected_indices:
            chunk = chunks[idx]
            word_freq = Counter(chunk["tokens"])
            top_words = [word for word, _ in word_freq.most_common(5)]

            # Calculate relevance score for display
            score = cosine_similarity([query_embedding], [chunk_embeddings[idx]])[0][0]

            results.append(
                {
                    "relevance_score": float(score),
                    "representative_words": top_words,
                    "full_text": chunk["text"],
                    "source_sentences": chunk["source_sentences"],
                    "all_tokens": chunk["tokens"],
                }
            )

        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results

    def display_results(self, results):
        """Display query-relevant word groups."""
        print("\n" + "=" * 80)
        print("QUERY-RELEVANT WORD GROUPS")
        print("=" * 80)

        if not results:
            print("No relevant groups found.")
            return

        for i, res in enumerate(results, 1):
            print(f"\n{'─' * 80}")
            print(f"Result {i} (Relevance: {res['relevance_score']:.4f})")
            print(f"{'─' * 80}")
            print(f"Key Words: {', '.join(res['representative_words'])}")
            print(f"\nContext:")
            for sent in res["source_sentences"]:
                print(f"  • {sent}")

        print("\n" + "=" * 80)


def main():
    # Diverse dataset covering Tech, Science, History, Finance, and Health
    documents = [
        # Tech & Cloud
        "Cloud computing allows scalable infrastructure. AWS and Azure are major providers.",
        "Serverless architecture reduces operational overhead. Functions run only when triggered by events.",
        "Containerization with Docker ensures consistent environments. Kubernetes orchestrates these containers at scale.",
        # Cybersecurity
        "Cybersecurity involves encryption and firewalls. Multi-factor authentication secures access.",
        "Zero-trust models require verification for every user and device. Data loss prevention tools monitor sensitive information.",
        "Penetration testing identifies vulnerabilities before attackers do. Regular security audits are essential for compliance.",
        # Data Science & AI
        "Machine learning algorithms use neural networks to process data. Deep learning is a subset of AI.",
        "Natural language processing enables machines to understand human text. Sentiment analysis helps businesses gauge customer opinion.",
        "Big data analytics requires distributed systems like Hadoop. Spark provides faster in-memory processing for large datasets.",
        # Finance
        "Blockchain technology underpins cryptocurrencies like Bitcoin. Decentralized ledgers ensure transparency and immutability.",
        "Algorithmic trading uses complex models to execute orders. High-frequency trading relies on low-latency network connections.",
        "Risk management involves assessing potential financial losses. Diversification spreads investment across various asset classes.",
        # Health & Biology
        "Genomics studies the complete set of DNA in organisms. CRISPR technology allows for precise gene editing.",
        "Telemedicine has expanded access to healthcare remotely. Wearable devices monitor vital signs in real-time.",
        "Vaccines stimulate the immune system to recognize pathogens. mRNA vaccines represent a new frontier in immunology.",
        # History & Literature
        "The Industrial Revolution transformed manufacturing processes. Steam engines powered factories and transportation networks.",
        "Shakespeare's plays explore universal human themes. His sonnets remain some of the most studied poetry in English.",
        "The Renaissance marked a rebirth of art and science. Figures like Da Vinci exemplified the polymath ideal.",
        # Environment
        "Climate change affects global weather patterns. Rising sea levels threaten coastal communities worldwide.",
        "Renewable energy sources include solar and wind power. Carbon capture technology aims to reduce atmospheric CO2.",
        "Biodiversity loss impacts ecosystem stability. Conservation efforts focus on protecting endangered species habitats.",
    ]

    extractor = QueryAwareWordGroupExtractor()

    # Example Query: Focus on Security/Tech
    query = "How do we secure data in the cloud?"

    results = extractor.search_and_group(
        query=query, documents=documents, top_k=4, chunk_size=2
    )

    extractor.display_results(results)


if __name__ == "__main__":
    main()
