import numpy as np
import requests
from jet.adapters.llama_cpp.config import EMBED_BASE_HOST, EMBED_MODEL


class NomicEmbedClient:
    def __init__(self, base_url=EMBED_BASE_HOST):
        self.base_url = base_url
        self.prompts = {
            "query": "search_query: ",
            "passage": "search_document: ",
            "Classification": "classification: ",
            "MultilabelClassification": "classification: ",
            "Clustering": "clustering: ",
            "PairClassification": "classification: ",
            "STS": "classification: ",
            "Summarization": "classification: ",
            "Speed": "search_document: ",
        }

    def embed(self, texts, task_type="passage"):
        if not isinstance(texts, list):
            texts = [texts]

        prefix = self.prompts.get(task_type, "search_document: ")
        prefixed_texts = [prefix + text for text in texts]

        try:
            response = requests.post(
                f"{self.base_url}/v1/embeddings",
                json={"model": EMBED_MODEL, "input": prefixed_texts},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if "data" in data:
                embeddings = [item["embedding"] for item in data["data"]]
            else:
                embeddings = [data["embedding"]]

            # CRITICAL FIX: Convert to numpy and L2 Normalize
            embs_np = np.array(embeddings)
            norms = np.linalg.norm(embs_np, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1
            normalized_embs = embs_np / norms

            return normalized_embs
        except Exception as e:
            print(f"Error embedding texts: {e}")
            return None

    def cosine_similarity(self, vec1, vec2):
        """Since vectors are already normalized, dot product == cosine similarity"""
        return np.dot(vec1, vec2)

    def get_centroid(self, texts, task_type="passage"):
        """Helper to get the normalized centroid of a list of texts"""
        embs = self.embed(texts, task_type=task_type)
        if embs is None or len(embs) == 0:
            return None
        centroid = np.mean(embs, axis=0)
        # Re-normalize the centroid
        norm = np.linalg.norm(centroid)
        if norm == 0:
            return centroid
        return centroid / norm


def run_classification():
    print("=== Single-Label Classification Example (Improved) ===\n")

    # 1. Define the items to classify
    tickets = [
        "My password isn't working and I can't log in",
        "I was charged twice for my subscription",
        "How do I export my data to CSV?",
        "The app crashes when I upload a photo",
    ]

    # 2. Define category EXAMPLES (Better than generic descriptions)
    categories = {
        "Technical Support": [
            "I can't log in to my account",
            "The app keeps crashing on startup",
            "I found a bug in the settings menu",
        ],
        "Billing": [
            "I was charged incorrectly",
            "How do I update my credit card?",
            "I want a refund for my last payment",
        ],
        "Feature Request": [
            "Can you add a dark mode?",
            "I wish I could export to PDF",
            "Please add support for multiple languages",
        ],
    }

    # 3. Get embeddings for tickets
    print("Embedding tickets...")
    ticket_embs = client.embed(tickets, task_type="Classification")

    # 4. Calculate centroids for each category
    print("Calculating category centroids...")
    cat_names = list(categories.keys())
    cat_centroids = []

    for cat_name, examples in categories.items():
        cent = client.get_centroid(examples, task_type="Classification")
        cat_centroids.append(cent)

    if ticket_embs is None or any(c is None for c in cat_centroids):
        return

    # 5. Classify each ticket by finding the closest category centroid
    print("\nResults:")
    print("-" * 50)
    for i, ticket in enumerate(tickets):
        similarities = [
            client.cosine_similarity(ticket_embs[i], cent) for cent in cat_centroids
        ]
        best_idx = np.argmax(similarities)
        best_cat = cat_names[best_idx]
        score = similarities[best_idx]

        print(f"Ticket: '{ticket}'")
        print(f"→ Category: {best_cat} (Score: {score:.4f})\n")


def run_multilabel_classification():
    print("=== Multilabel Classification Example ===\n")

    articles = [
        "New iPhone release features AI-powered camera and improved battery life",
        "Tesla announces price cuts on Model 3 and Model Y vehicles",
        "Study shows Mediterranean diet reduces heart disease risk",
    ]

    tags = {
        "Technology": [
            "New smartphone released with AI features",
            "Latest software update for laptops",
        ],
        # Added "product launch" to help bridge Tech/Business
        "Business": [
            "Stock market hits record high",
            "Company announces quarterly earnings",
            "Major tech company releases new product",
        ],
        "Health": [
            "Benefits of daily exercise",
            "New study on heart health",
            "Diet impacts disease risk",
        ],
        "Automotive": [
            "Electric vehicle sales rise",
            "New car model review",
            "Tesla releases new vehicle",
        ],
    }

    print("Embedding articles and tags...")
    article_embs = client.embed(articles, task_type="MultilabelClassification")

    tag_names = list(tags.keys())
    tag_centroids = []
    for tag_name, examples in tags.items():
        cent = client.get_centroid(examples, task_type="MultilabelClassification")
        tag_centroids.append(cent)

    if article_embs is None or any(c is None for c in tag_centroids):
        return

    THRESHOLD = 0.50

    print("\nResults:")
    print("-" * 50)
    for i, article in enumerate(articles):
        relevant_tags = []
        scores = {}

        for j, tag_name in enumerate(tag_names):
            sim = client.cosine_similarity(article_embs[i], tag_centroids[j])
            scores[tag_name] = sim
            if sim > THRESHOLD:
                relevant_tags.append(tag_name)

        print(f"Article: '{article[:60]}...'")
        print(f"Tags: {relevant_tags}")
        print(f"Scores: {', '.join([f'{k}: {v:.2f}' for k, v in scores.items()])}")
        print()


def run_clustering():
    print("=== Clustering Example ===\n")
    from sklearn.cluster import KMeans

    # 1. Define feedback items
    feedback_items = [
        "Love the new dark mode feature!",
        "The app is too slow when loading images",
        "Would be great to have offline mode",
        "Battery drain is excessive after update",
        "Please add support for multiple accounts",
        "UI is confusing on smaller screens",
        "Export feature doesn't work properly",
        "Best productivity app I've used",
        "Need better search functionality",
        "Crashes frequently on Android 14",
    ]

    # 2. Get embeddings
    print("Embedding feedback items...")
    embeddings = client.embed(feedback_items, task_type="Clustering")

    if embeddings is None:
        return

    # 3. Perform K-Means clustering
    NUM_CLUSTERS = 3
    print(f"Running K-Means with {NUM_CLUSTERS} clusters...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)

    # 4. Group and display results
    clustered_feedback = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in clustered_feedback:
            clustered_feedback[cluster_id] = []
        clustered_feedback[cluster_id].append(feedback_items[i])

    print("\nResults:")
    print("-" * 50)
    for cluster_id, items in sorted(clustered_feedback.items()):
        print(f"\n=== Cluster {cluster_id} ({len(items)} items) ===")
        for item in items:
            print(f"  • {item}")


def run_pair_classification():
    print("=== Pair Classification Example (Final Polish) ===\n")

    question_pairs = [
        ("How do I reset my password?", "What's the process to change my password?"),
        ("How do I reset my password?", "Where can I find billing information?"),
        ("Can I cancel my subscription?", "How do I stop recurring payments?"),
        ("What are your business hours?", "When is customer support available?"),
    ]

    q1_list = [q1 for q1, _ in question_pairs]
    q2_list = [q2 for _, q2 in question_pairs]

    print("Embedding questions...")
    q1_embs = client.embed(q1_list, task_type="STS")
    q2_embs = client.embed(q2_list, task_type="STS")

    if q1_embs is None or q2_embs is None:
        return

    print("\nResults:")
    print("-" * 50)
    # Lowered to 0.55 to catch semantic equivalents like "cancel" vs "stop"
    DUPLICATE_THRESHOLD = 0.55

    for i, (q1, q2) in enumerate(question_pairs):
        sim = client.cosine_similarity(q1_embs[i], q2_embs[i])
        relationship = "Duplicate" if sim > DUPLICATE_THRESHOLD else "Different"

        print(f"Q1: {q1}")
        print(f"Q2: {q2}")
        print(f"→ Relationship: {relationship} (Similarity: {sim:.4f})\n")


def run_sts():
    print("=== Semantic Textual Similarity (STS) Example ===\n")

    # 1. Define text pairs
    text_pairs = [
        ("The cat sat on the mat", "A feline was resting on the rug"),
        ("The cat sat on the mat", "The dog ran in the park"),
        ("Python is great for data science", "Data science works well with Python"),
        ("I love ice cream", "Chocolate ice cream is my favorite"),
    ]

    # 2. Embed both sentences in each pair
    print("Embedding sentence pairs...")
    sent1_list = [s1 for s1, _ in text_pairs]
    sent2_list = [s2 for _, s2 in text_pairs]

    sent1_embs = client.embed(sent1_list, task_type="STS")
    sent2_embs = client.embed(sent2_list, task_type="STS")

    if sent1_embs is None or sent2_embs is None:
        return

    # 3. Calculate similarity scores
    print("\nResults:")
    print("-" * 50)
    for i, (s1, s2) in enumerate(text_pairs):
        sim = client.cosine_similarity(sent1_embs[i], sent2_embs[i])

        # Map to STS score (0-5 scale)
        sts_score = sim * 5

        print(f"Sentence 1: {s1}")
        print(f"Sentence 2: {s2}")
        print(f"Cosine Similarity: {sim:.4f}")
        print(f"STS Score (0-5): {sts_score:.2f}")
        print()

    # 4. Bonus: Content Deduplication Example
    print("\n=== Bonus: News Deduplication ===")
    news_articles = [
        "Apple releases new iPhone with advanced camera",
        "New iPhone from Apple features improved photography",
        "Samsung launches Galaxy S25 with AI capabilities",
        "Google announces Pixel 9 with enhanced features",
    ]

    article_embs = client.embed(news_articles, task_type="STS")
    if article_embs is None:
        return

    duplicates = []
    for i in range(len(news_articles)):
        for j in range(i + 1, len(news_articles)):
            sim = client.cosine_similarity(article_embs[i], article_embs[j])
            if sim > 0.90:
                duplicates.append((i, j, sim))

    if duplicates:
        print("Potential duplicates found:")
        for i, j, sim in duplicates:
            print(f"  Articles {i} & {j}: similarity={sim:.3f}")
            print(f"    '{news_articles[i][:50]}...'")
            print(f"    '{news_articles[j][:50]}...'")
    else:
        print("No duplicates found above threshold 0.90")


def run_summarization():
    print("=== Extractive Summarization Example ===\n")

    # 1. Define a document
    document = """
    Artificial intelligence has transformed many industries in recent years. 
    Machine learning algorithms can now diagnose diseases with accuracy comparable to human doctors. 
    Self-driving cars are becoming increasingly common on our roads. 
    AI-powered chatbots handle customer service inquiries efficiently. 
    However, concerns about job displacement and privacy remain significant challenges. 
    Regulators worldwide are working to establish guidelines for ethical AI development. 
    The technology continues to evolve rapidly, with new breakthroughs announced regularly.
    """

    # 2. Split into sentences (simple split for demo)
    sentences = [s.strip() for s in document.split(".") if s.strip()]
    # Add period back for readability
    sentences = [s + "." for s in sentences]

    print(f"Document has {len(sentences)} sentences.")

    # 3. Get embeddings
    print("Embedding sentences...")
    sentence_embs = client.embed(sentences, task_type="Summarization")

    if sentence_embs is None:
        return

    # 4. Calculate centroid (average embedding represents the whole document)
    centroid = np.mean(sentence_embs, axis=0)

    # 5. Find sentences closest to centroid
    similarities = [client.cosine_similarity(centroid, emb) for emb in sentence_embs]

    # Select top 3 most representative sentences
    NUM_SUMMARY_SENTENCES = 3
    top_indices = np.argsort(similarities)[-NUM_SUMMARY_SENTENCES:][::-1]
    top_indices_sorted = sorted(top_indices)  # Keep original order

    summary_sentences = [sentences[i] for i in top_indices_sorted]

    print("\nExtractive Summary:")
    print("-" * 50)
    for sent in summary_sentences:
        print(f"  • {sent}")

    # 6. Bonus: Evaluate a generated summary
    print("\n=== Bonus: Summary Quality Evaluation ===")
    original_text = "The company reported record profits of $5 billion in Q4"
    good_summary = "Company achieves $5B profit in fourth quarter"
    bad_summary = "The weather was nice today"

    embs = client.embed(
        [original_text, good_summary, bad_summary], task_type="Summarization"
    )
    if embs is None:
        return

    gen_sim = client.cosine_similarity(embs[0], embs[1])
    bad_sim = client.cosine_similarity(embs[0], embs[2])

    print(f"Original: '{original_text}'")
    print(f"Good Summary: '{good_summary}' → Similarity: {gen_sim:.4f}")
    print(f"Bad Summary: '{bad_summary}' → Similarity: {bad_sim:.4f}")


if __name__ == "__main__":
    # Initialize client
    client = NomicEmbedClient()

    run_classification()
    run_multilabel_classification()
    run_clustering()
    run_pair_classification()
    run_sts()
    run_summarization()
