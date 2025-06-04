from tqdm import tqdm
import string
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load model
model = SentenceTransformer('intfloat/e5-base-v2')


def preprocess_texts(scraped_data):
    """Convert scraped data into passages with 'passage:' prefix."""
    return [f"passage: {item['text']}" for item in scraped_data]


def extract_keywords(texts, top_n=3):
    """Extract top keywords from a list of texts using TF-IDF."""
    stop_words = list(set(stopwords.words('english')
                          ).union(set(string.punctuation)))
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=100)
    tfidf_matrix = vectorizer.fit_transform(
        tqdm(texts, desc="Extracting keywords"))
    feature_names = vectorizer.get_feature_names_out()
    # Get top keywords per document
    keywords = []
    for doc_idx in range(tfidf_matrix.shape[0]):
        tfidf_scores = tfidf_matrix[doc_idx].toarray()[0]
        top_indices = tfidf_scores.argsort()[-top_n:][::-1]
        keywords.append([feature_names[i] for i in top_indices])
    return keywords


def dynamic_label_tagging(scraped_data, num_clusters=2, threshold=0.8, query=None):
    """Dynamically extract labels and tag documents using clustering with progress tracking."""
    # Preprocess texts
    passages = preprocess_texts(scraped_data)

    # Generate embeddings with progress bar
    embeddings = model.encode(
        passages,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,  # Built-in tqdm progress bar
        convert_to_numpy=True
    )

    # Cluster embeddings with progress indication
    print("Clustering embeddings...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    with tqdm(total=1, desc="Performing K-means clustering") as pbar:
        cluster_labels = kmeans.fit_predict(embeddings)
        pbar.update(1)

    # Group texts by cluster
    clusters = {i: [] for i in range(num_clusters)}
    for i, label in enumerate(cluster_labels):
        clusters[label].append(scraped_data[i]["text"])

    # Generate labels for each cluster using TF-IDF
    cluster_label_names = {}
    for cluster_id, texts in tqdm(clusters.items(), desc="Generating cluster labels"):
        if texts:  # Ensure cluster is not empty
            keywords = extract_keywords(texts, top_n=2)
            # Combine keywords into a label (simplified)
            cluster_label_names[cluster_id] = " ".join(
                set(sum(keywords, [])))[:30]
        else:
            # Fallback label
            cluster_label_names[cluster_id] = f"cluster_{cluster_id}"

    # Assign labels to documents
    tagged_results = []
    for i, (data, cluster_id) in enumerate(tqdm(zip(scraped_data, cluster_labels), total=len(scraped_data), desc="Assigning labels")):
        # Compute similarity to cluster centroid for multi-label potential
        sim_to_centroid = np.dot(
            embeddings[i], kmeans.cluster_centers_[cluster_id])
        labels = [{"label": cluster_label_names[cluster_id],
                   "similarity": float(sim_to_centroid)}]

        # Optional: Multi-label by checking similarity to other centroids
        for other_cluster_id in range(num_clusters):
            if other_cluster_id != cluster_id:
                sim = np.dot(embeddings[i], kmeans.cluster_centers_[
                             other_cluster_id])
                if sim >= threshold:
                    labels.append(
                        {"label": cluster_label_names[other_cluster_id], "similarity": float(sim)})

        tagged_results.append({
            "header": data["header"],
            "text": data["text"],
            "labels": labels
        })

    # Optional: Filter by query
    if query:
        query_text = f"query: {query}"
        print("Encoding query...")
        query_embedding = model.encode(
            [query_text], normalize_embeddings=True, show_progress_bar=False)[0]
        for result in tqdm(tagged_results, desc="Computing query similarities"):
            passage_embedding = embeddings[tagged_results.index(result)]
            query_sim = float(np.dot(passage_embedding, query_embedding))
            result["query_similarity"] = query_sim

    return tagged_results


# Example usage
if __name__ == "__main__":
    scraped_data = [
        {"header": "Section 1",
            "text": "The CDC recommends 46 grams of protein daily for women aged 19-70."},
        {"header": "Section 2", "text": "Carbohydrates are the body's main energy source."},
        {"header": "Section 3",
            "text": "Protein intake varies based on activity level and age, and carbs fuel exercise."}
    ]

    query = "how much protein should a female eat"
    results = dynamic_label_tagging(
        scraped_data, num_clusters=2, threshold=0.8, query=query)

    for r in results:
        print(f"Header: {r['header']}")
        print(f"Text: {r['text']}")
        print("Labels:", [
              f"{l['label']} (Similarity: {l['similarity']:.3f})" for l in r['labels']])
        if "query_similarity" in r:
            print(f"Query Similarity: {r['query_similarity']:.3f}")
        print()
