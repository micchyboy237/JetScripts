"""
Advanced Hybrid Word Group Extraction
=====================================
Combines Semantic Search (Embeddings) with Statistical Co-occurrence
to rank results. Uses Sentence-level co-occurrence to boost chunks
where query terms appear together frequently.
"""

import warnings

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


class AdvancedHybridExtractor:
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

    def extract_chunks(self, documents, chunk_size=2):
        """Extract chunks for embedding."""
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

    def search_and_group_hybrid(
        self, query, documents, top_k=5, chunk_size=2, alpha=0.6, beta=0.4
    ):
        """
        Main method: Find word groups using a hybrid of Semantic and Co-occurrence scoring.

        Args:
            query: User search query
            documents: List of document strings
            top_k: Number of results to return
            chunk_size: Sentences per chunk
            alpha: Weight for Semantic Similarity (0.0 - 1.0)
            beta: Weight for Co-occurrence Overlap (0.0 - 1.0)
        """
        print(f"\nProcessing query: '{query}'")
        print(f"Hybrid Weights: Semantic={alpha}, Co-occurrence={beta}")

        # 1. Prepare Data
        chunks = self.extract_chunks(documents, chunk_size=chunk_size)
        chunk_texts = [c["text"] for c in chunks]

        if not chunk_texts:
            return []

        # 2. Semantic Scoring (Embeddings)
        print("Calculating semantic similarity...")
        query_embedding = embed(query, model=self.model_name, return_format="numpy")
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        chunk_embeddings = embed(
            chunk_texts,
            model=self.model_name,
            return_format="numpy",
            show_progress=False,
        )
        norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        chunk_embeddings = chunk_embeddings / norms

        semantic_scores = cosine_similarity([query_embedding], chunk_embeddings)[0]

        # 3. Co-occurrence Scoring (Sentence Mode)
        print("Calculating co-occurrence overlap...")
        query_tokens = set(nltk.word_tokenize(query.lower()))
        query_tokens = {w for w in query_tokens if w.isalpha() and len(w) > 2}

        co_occurrence_scores = []
        for chunk in chunks:
            chunk_token_set = set(chunk["tokens"])
            # Intersection: How many query words appear in this chunk?
            overlap = query_tokens.intersection(chunk_token_set)

            # Score based on coverage ratio (Jaccard-like)
            # If query has 4 words and chunk has 3 of them, score is high.
            if len(query_tokens) > 0:
                score = len(overlap) / len(query_tokens)
            else:
                score = 0
            co_occurrence_scores.append(score)

        # Normalize co-occurrence scores to 0-1 range for fair weighting
        max_co = max(co_occurrence_scores) if co_occurrence_scores else 1
        if max_co > 0:
            norm_co_scores = [s / max_co for s in co_occurrence_scores]
        else:
            norm_co_scores = co_occurrence_scores

        # 4. Hybrid Ranking
        final_scores = []
        for i in range(len(chunks)):
            hybrid_score = (alpha * semantic_scores[i]) + (beta * norm_co_scores[i])
            final_scores.append(hybrid_score)

        # Get top K indices based on hybrid score
        top_indices = np.argsort(final_scores)[::-1][:top_k]

        # 5. Format Results
        results = []
        for idx in top_indices:
            chunk = chunks[idx]
            results.append(
                {
                    "hybrid_score": float(final_scores[idx]),
                    "semantic_score": float(semantic_scores[idx]),
                    "co_occurrence_ratio": float(norm_co_scores[idx]),
                    "key_words": list(set(chunk["tokens"]))[:8],
                    "full_text": chunk["text"],
                    "source_sentences": chunk["source_sentences"],
                }
            )

        return results

    def display_results(self, results):
        print("\n" + "=" * 80)
        print("ADVANCED HYBRID WORD GROUPS")
        print("=" * 80)

        for i, res in enumerate(results, 1):
            print(f"\n{'─' * 80}")
            print(f"Group {i} (Hybrid Score: {res['hybrid_score']:.4f})")
            print(f"{'─' * 80}")
            print(f"  Semantic Match:   {res['semantic_score']:.4f}")
            print(
                f"  Keyword Overlap:  {res['co_occurrence_ratio']:.4f} (Ratio of query terms found)"
            )

            print(f"\nKey Words: {', '.join(res['key_words'])}")

            print(f"\nContext:")
            for sent in res["source_sentences"]:
                print(f"  • {sent}")


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

    extractor = AdvancedHybridExtractor()

    # Example Query: Focus on Security/Tech
    query = "How do we secure data in the cloud?"

    # Using default alpha=0.6 (Semantic) and beta=0.4 (Co-occurrence)
    results = extractor.search_and_group_hybrid(
        query=query, documents=documents, top_k=4, chunk_size=2
    )

    extractor.display_results(results)


if __name__ == "__main__":
    main()
