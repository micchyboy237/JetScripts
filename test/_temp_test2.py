"""
Modern Approach: Word Grouping using Jet Adapters (Llama.cpp) + Clustering
======================================================================
This code demonstrates how to extract groups of words that commonly appear
together in sentences/paragraphs from a collection of documents using
local llama.cpp embeddings.
"""

import warnings
from collections import Counter, defaultdict

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans

warnings.filterwarnings("ignore")

# Import Jet Adapter for Embeddings
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.embed_utils import embed

try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt_tab", quiet=True)


class WordGroupExtractor:
    """
    Extracts groups of words that commonly appear together in sentences/paragraphs
    using sentence embeddings and clustering.
    """

    def __init__(self, model_name=EMBED_MODEL):
        """
        Initialize the extractor.
        Args:
            model_name: Name of the embedding model to use (from config)
        """
        print(f"Using embedding model: {model_name}")
        self.model_name = model_name
        self.stop_words = set(stopwords.words("english"))
        self.stop_words.update(
            [
                ".",
                ",",
                "!",
                "?",
                ";",
                ":",
                '"',
                "'",
                "(",
                ")",
                "[",
                "]",
                "{",
                "}",
                "-",
                "--",
                "...",
                "n't",
                "'s",
                "'re",
            ]
        )

    def preprocess_text(self, text):
        """
        Clean and tokenize text into sentences and words.
        """
        sentences = sent_tokenize(text)
        processed_sentences = []
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            filtered_words = [
                word
                for word in words
                if word not in self.stop_words and len(word) > 2 and word.isalpha()
            ]
            if filtered_words:
                processed_sentences.append(filtered_words)
        return processed_sentences

    def extract_word_contexts(self, documents, window_size=5):
        """
        Extract word contexts from documents using sliding windows.
        """
        all_contexts = []
        all_texts = []
        for doc_idx, document in enumerate(documents):
            sentences = self.preprocess_text(document)
            for sentence in sentences:
                for i in range(len(sentence)):
                    start = max(0, i - window_size // 2)
                    end = min(len(sentence), i + window_size // 2 + 1)
                    context = sentence[start:end]
                    if len(context) >= 2:
                        context_text = " ".join(context)
                        all_contexts.append(context)
                        all_texts.append(context_text)
        return all_contexts, all_texts

    def generate_embeddings(self, texts, batch_size=32):
        """
        Generate embeddings for a list of texts using Jet Adapters.
        """
        print(f"Generating embeddings for {len(texts)} texts...")

        # Use the jet adapter embed function
        # It handles batching and progress bars internally
        embeddings = embed(
            text=texts,
            model=self.model_name,
            return_format="numpy",
            show_progress=True,
            batch_size=batch_size,
        )

        # Normalize embeddings to unit length for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized_embeddings = embeddings / norms

        return normalized_embeddings

    def cluster_embeddings(self, embeddings, method="dbscan", **kwargs):
        """
        Cluster embeddings using specified method.
        """
        if method == "dbscan":
            eps = kwargs.get("eps", 0.3)
            min_samples = kwargs.get("min_samples", 5)
            print(f"Clustering with DBSCAN (eps={eps}, min_samples={min_samples})...")
            clustering = DBSCAN(
                eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1
            )
        elif method == "kmeans":
            n_clusters = kwargs.get("n_clusters", 10)
            print(f"Clustering with KMeans (n_clusters={n_clusters})...")
            clustering = KMeans(
                n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300
            )
        elif method == "hierarchical":
            n_clusters = kwargs.get("n_clusters", 10)
            print(
                f"Clustering with Agglomerative Clustering (n_clusters={n_clusters})..."
            )
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters, metric="cosine", linkage="average"
            )
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        labels = clustering.fit_predict(embeddings)
        return labels

    def extract_word_groups(
        self,
        documents,
        window_size=5,
        clustering_method="dbscan",
        min_group_size=3,
        **clustering_kwargs,
    ):
        """
        Main method to extract word groups from documents.
        """
        print("\n" + "=" * 60)
        print("Step 1: Extracting word contexts from documents...")
        contexts, context_texts = self.extract_word_contexts(documents, window_size)
        print(f"Extracted {len(contexts)} contexts")

        if len(contexts) == 0:
            print("No contexts extracted. Check your input documents.")
            return {}

        print("\n" + "=" * 60)
        embeddings = self.generate_embeddings(context_texts)

        print("\n" + "=" * 60)
        labels = self.cluster_embeddings(
            embeddings, clustering_method, **clustering_kwargs
        )

        print("\n" + "=" * 60)
        print("Step 4: Analyzing word groups...")

        cluster_groups = defaultdict(list)
        for idx, label in enumerate(labels):
            if label != -1:
                cluster_groups[label].append(
                    {"context": contexts[idx], "text": context_texts[idx]}
                )

        word_groups = {}
        for cluster_id, group in cluster_groups.items():
            if len(group) >= min_group_size:
                all_words = []
                for item in group:
                    all_words.extend(item["context"])

                word_freq = Counter(all_words)
                common_words = word_freq.most_common(10)
                representative_words = [word for word, count in common_words[:5]]

                word_groups[cluster_id] = {
                    "size": len(group),
                    "representative_words": representative_words,
                    "word_frequencies": dict(common_words),
                    "sample_contexts": [item["text"] for item in group[:5]],
                    "all_contexts": group,
                }

        sorted_groups = dict(
            sorted(word_groups.items(), key=lambda x: x[1]["size"], reverse=True)
        )

        print(f"\nFound {len(sorted_groups)} meaningful word groups")
        return sorted_groups

    def display_results(self, word_groups, top_n=10):
        """
        Display the extracted word groups in a readable format.
        """
        print("\n" + "=" * 80)
        print("EXTRACTED WORD GROUPS")
        print("=" * 80)

        for i, (cluster_id, group_info) in enumerate(list(word_groups.items())[:top_n]):
            print(f"\n{'─' * 80}")
            print(f"Group {i + 1} (Cluster ID: {cluster_id})")
            print(f"{'─' * 80}")
            print(f"Size: {group_info['size']} contexts")
            print(f"\nRepresentative Words:")
            print(f"  {', '.join(group_info['representative_words'])}")

            print(f"\nTop Word Frequencies:")
            for word, freq in list(group_info["word_frequencies"].items())[:10]:
                print(f"  {word:20s}: {freq}")

            print(f"\nSample Contexts:")
            for j, context in enumerate(group_info["sample_contexts"][:3], 1):
                print(f'  {j}. "{context}"')

        print("\n" + "=" * 80)


def main():
    """Example usage with sample documents."""
    documents = [
        """
        Machine learning is a subset of artificial intelligence that focuses on 
        building systems that learn from data. Deep learning algorithms use neural 
        networks with multiple layers to process complex patterns. Natural language 
        processing enables computers to understand human language. Computer vision 
        allows machines to interpret visual information from images and videos.
        """,
        """
        Data science combines statistics, programming, and domain expertise to extract 
        insights from data. Statistical analysis helps identify patterns and trends in 
        large datasets. Python and R are popular programming languages for data analysis. 
        Machine learning models can predict future outcomes based on historical data.
        """,
        """
        Web development involves creating websites and web applications. Frontend 
        development focuses on user interface design using HTML, CSS, and JavaScript. 
        Backend development handles server-side logic and database management. Modern 
        frameworks like React and Angular simplify frontend development processes.
        """,
        """
        Cloud computing provides on-demand access to computing resources over the internet. 
        Amazon Web Services, Microsoft Azure, and Google Cloud Platform are major cloud 
        providers. Containerization with Docker and orchestration with Kubernetes enable 
        scalable application deployment. Serverless computing eliminates infrastructure management.
        """,
        """
        Cybersecurity protects computer systems and networks from digital attacks. Encryption 
        algorithms secure data transmission and storage. Firewalls and intrusion detection 
        systems monitor network traffic for suspicious activities. Multi-factor authentication 
        adds additional security layers beyond passwords.
        """,
        """
        Software engineering applies systematic approaches to software development. Agile 
        methodologies emphasize iterative development and collaboration. Version control 
        systems like Git track changes in source code. Continuous integration and deployment 
        automate the software release process.
        """,
        """
        Database management systems store and organize data efficiently. Relational databases 
        use structured query language for data manipulation. NoSQL databases handle unstructured 
        data at scale. Indexing improves query performance by creating optimized data structures.
        """,
        """
        Mobile app development creates applications for smartphones and tablets. iOS development 
        uses Swift programming language while Android uses Java or Kotlin. Cross-platform 
        frameworks like Flutter and React Native enable code reuse across platforms. User 
        experience design is crucial for mobile application success.
        """,
    ]

    extractor = WordGroupExtractor()
    word_groups = extractor.extract_word_groups(
        documents,
        window_size=5,
        clustering_method="dbscan",
        min_group_size=3,
        eps=0.35,
        min_samples=3,
    )

    extractor.display_results(word_groups, top_n=10)
    return word_groups


if __name__ == "__main__":
    word_groups = main()
