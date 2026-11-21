"""
Dynamic Cluster-Hybrid Word Group Extraction (Fixed Prompt)
============================================================
Uses an LLM to dynamically extract concept clusters from the query,
ensuring both 'Action' and 'Target/Context' are represented.
"""

import json
import warnings

import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# Jet Adapters
from jet.adapters.llama_cpp.config import EMBED_MODEL, LLM_MODEL
from jet.adapters.llama_cpp.embed_utils import embed
from jet.adapters.llama_cpp.factory import get_llm_client

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


class DynamicClusterExtractor:
    def __init__(self, embed_model=EMBED_MODEL, llm_model=LLM_MODEL):
        print(f"Using Embed Model: {embed_model}")
        print(f"Using LLM for Query Analysis: {llm_model}")
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.llm_client = get_llm_client()

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

    def get_dynamic_clusters(self, query):
        """
        Uses LLM to extract key concept clusters from the query.
        Forces identification of both 'Methods' and 'Targets/Context'.
        """
        prompt = f"""
        Analyze the user query and identify 2-3 distinct conceptual clusters.
        You MUST include at least one cluster for the 'Action/Method' and one for the 'Target/Context/Object'.
        
        Query: "{query}"
        
        Return ONLY a valid JSON object in this format:
        {{
            "clusters": [
                {{
                    "name": "Action/Method Cluster",
                    "keywords": ["action_kw1", "action_kw2", "synonym1"]
                }},
                {{
                    "name": "Target/Context Cluster",
                    "keywords": ["target_kw1", "context_kw2", "object1"]
                }}
            ]
        }}
        """

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            return data.get("clusters", [])
        except Exception as e:
            print(f"⚠️ LLM Cluster Extraction Failed: {e}")
            tokens = nltk.word_tokenize(query.lower())
            return [
                {
                    "name": "Query Terms",
                    "keywords": [t for t in tokens if t.isalpha() and len(t) > 2],
                }
            ]

    def search_and_group_dynamic(
        self, query, documents, top_k=5, chunk_size=2, alpha=0.6, beta=0.4
    ):
        """
        Main method: Dynamic clustering + Hybrid ranking.
        """
        print(f"\nProcessing query: '{query}'")

        # 1. Get Dynamic Clusters from LLM
        print("🧠 Extracting concept clusters via LLM...")
        clusters = self.get_dynamic_clusters(query)
        print(f"   Found {len(clusters)} clusters: {[c['name'] for c in clusters]}")

        cluster_sets = []
        for c in clusters:
            cluster_sets.append({"name": c["name"], "keywords": set(c["keywords"])})

        # 2. Prepare Documents
        chunks = self.extract_chunks(documents, chunk_size=chunk_size)
        chunk_texts = [c["text"] for c in chunks]
        if not chunk_texts:
            return []

        # 3. Semantic Scoring
        query_embedding = embed(query, model=self.embed_model, return_format="numpy")
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        chunk_embeddings = embed(
            chunk_texts,
            model=self.embed_model,
            return_format="numpy",
            show_progress=False,
        )
        norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        chunk_embeddings = chunk_embeddings / norms

        semantic_scores = cosine_similarity([query_embedding], chunk_embeddings)[0]

        # 4. Dynamic Cluster Scoring
        cluster_scores = []
        cluster_matches_log = []

        for chunk in chunks:
            chunk_tokens = set(chunk["tokens"])
            total_score = 0
            matches_found = {}

            active_clusters = 0
            for cs in cluster_sets:
                matches = chunk_tokens.intersection(cs["keywords"])
                if matches:
                    active_clusters += 1
                    matches_found[cs["name"]] = list(matches)
                    total_score += 0.5 + (0.1 * len(matches))

            # Bonus for hitting MULTIPLE clusters (The "Bridge")
            if active_clusters > 1:
                total_score += 0.5 * active_clusters

            cluster_scores.append(total_score)
            cluster_matches_log.append(matches_found)

        # Normalize cluster scores
        max_cl = max(cluster_scores) if cluster_scores else 1
        norm_cl_scores = (
            [s / max_cl for s in cluster_scores] if max_cl > 0 else cluster_scores
        )

        # 5. Hybrid Ranking
        final_scores = [
            (alpha * semantic_scores[i]) + (beta * norm_cl_scores[i])
            for i in range(len(chunks))
        ]
        top_indices = np.argsort(final_scores)[::-1][:top_k]

        # 6. Format Results
        results = []
        for idx in top_indices:
            chunk = chunks[idx]
            matches = cluster_matches_log[idx]

            bond_desc = []
            if len(matches) > 1:
                names = list(matches.keys())
                bond_desc.append(f"Bridge: {' + '.join(names)}")
            elif matches:
                name = list(matches.keys())[0]
                bond_desc.append(f"Single Concept: {name}")

            results.append(
                {
                    "hybrid_score": float(final_scores[idx]),
                    "semantic_score": float(semantic_scores[idx]),
                    "cluster_relevance": float(norm_cl_scores[idx]),
                    "key_words": list(set(chunk["tokens"]))[:8],
                    "concept_bonds": bond_desc,
                    "matched_keywords": matches,
                    "full_text": chunk["text"],
                    "source_sentences": chunk["source_sentences"],
                }
            )

        return results

    def display_results(self, results):
        print("\n" + "=" * 80)
        print("DYNAMIC CLUSTER-HYBRID WORD GROUPS")
        print("=" * 80)

        for i, res in enumerate(results, 1):
            print(f"\n{'─' * 80}")
            print(f"Group {i} (Hybrid Score: {res['hybrid_score']:.4f})")
            print(f"{'─' * 80}")
            print(f"  Semantic Match:   {res['semantic_score']:.4f}")
            print(f"  Cluster Relevance:{res['cluster_relevance']:.4f}")

            if res["concept_bonds"]:
                print(f"  🔗 {res['concept_bonds'][0]}")
                for cluster_name, kws in res["matched_keywords"].items():
                    print(f"     • {cluster_name}: {', '.join(kws)}")

            print(f"\nKey Words: {', '.join(res['key_words'])}")

            print(f"\nContext:")
            for sent in res["source_sentences"]:
                print(f"  • {sent}")


def main():
    documents = [
        "Cloud computing allows scalable infrastructure. AWS and Azure are major providers.",
        "Serverless architecture reduces operational overhead. Functions run only when triggered by events.",
        "Containerization with Docker ensures consistent environments. Kubernetes orchestrates these containers at scale.",
        "Cybersecurity involves encryption and firewalls. Multi-factor authentication secures access.",
        "Zero-trust models require verification for every user and device. Data loss prevention tools monitor sensitive information.",
        "Penetration testing identifies vulnerabilities before attackers do. Regular security audits are essential for compliance.",
        "Machine learning algorithms use neural networks to process data. Deep learning is a subset of AI.",
        "Natural language processing enables machines to understand human text. Sentiment analysis helps businesses gauge customer opinion.",
        "Big data analytics requires distributed systems like Hadoop. Spark provides faster in-memory processing for large datasets.",
        "Blockchain technology underpins cryptocurrencies like Bitcoin. Decentralized ledgers ensure transparency and immutability.",
        "Algorithmic trading uses complex models to execute orders. High-frequency trading relies on low-latency network connections.",
        "Risk management involves assessing potential financial losses. Diversification spreads investment across various asset classes.",
        "Genomics studies the complete set of DNA in organisms. CRISPR technology allows for precise gene editing.",
        "Telemedicine has expanded access to healthcare remotely. Wearable devices monitor vital signs in real-time.",
        "Vaccines stimulate the immune system to recognize pathogens. mRNA vaccines represent a new frontier in immunology.",
        "The Industrial Revolution transformed manufacturing processes. Steam engines powered factories and transportation networks.",
        "Shakespeare's plays explore universal human themes. His sonnets remain some of the most studied poetry in English.",
        "The Renaissance marked a rebirth of art and science. Figures like Da Vinci exemplified the polymath ideal.",
        "Climate change affects global weather patterns. Rising sea levels threaten coastal communities worldwide.",
        "Renewable energy sources include solar and wind power. Carbon capture technology aims to reduce atmospheric CO2.",
        "Biodiversity loss impacts ecosystem stability. Conservation efforts focus on protecting endangered species habitats.",
    ]

    extractor = DynamicClusterExtractor()
    query = "How do we secure data in the cloud?"

    results = extractor.search_and_group_dynamic(
        query=query, documents=documents, top_k=4, chunk_size=2, alpha=0.6, beta=0.4
    )

    extractor.display_results(results)


if __name__ == "__main__":
    main()
