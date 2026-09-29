from jet.libs.bertopic.monkey_patches.add_check_array import init_patch

init_patch()


import json

from bertopic import BERTopic
from hdbscan import HDBSCAN
from jet.adapters.llama_cpp.chunking_utils import truncate_texts
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.embed_utils import embed
from jet.logger import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP

# Load JSON file
json_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Apps/my-jobs/generated/run_search_full_jobs/junior_ai_engineer/search_results.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 1. Dataset
documents = [f"{j['title']}\n{j['details']}" for j in data.get("results", [])]
documents = truncate_texts(documents, model=EMBED_MODEL, max_tokens=500)

print("--- Step 1: Generating Local Embeddings ---")
# BERTopic needs an embedding matrix (NumPy array). We use your native llama.cpp utility.
target_model = EMBED_MODEL
logger.info(f"Encoding {len(documents)} documents using local model: {target_model}")

embeddings = embed(
    text=documents, model=target_model, return_format="numpy", show_progress=True
)

print(f"Generated embedding matrix shape: {embeddings.shape}")


print("\n--- Step 2: Configuring BERTopic Pipeline ---")
# Step 2a: Dimension reduction (Crucial for high-dim vectors like Nomic)
# For a massive collection, tweak n_neighbors and n_components
umap_model = UMAP(
    n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42
)

# Step 2b: Density-based clustering
hdbscan_model = HDBSCAN(
    min_cluster_size=2,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)

# Step 2c: Fine-tune tokenization with TF-IDF for cleaner keywords
vectorizer_model = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True,
    min_df=1,  # Since you have a small doc set
    max_df=0.9,  # Filter near-ubiquitous terms
)


print("\n--- Step 3: Fitting Topic Model ---")
topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    calculate_probabilities=False,  # Keep False for speed on massive datasets
)

# Fit the model using pre-calculated local embeddings
topics, probs = topic_model.fit_transform(documents, embeddings)


print("\n--- Step 4: Displaying Common Shared Themes ---")
# Get overview of all discovered clusters
# Topic -1 is reserved for outliers/noise that didn't fit anywhere cleanly
topic_info = topic_model.get_topic_info()
print(topic_info[["Topic", "Count", "Name"]])


print("\n--- Step 5: Deep Dive into Discovered Themes ---")
for topic_id in set(topics):
    if topic_id == -1:
        print("\n❌ Outliers / Unclustered Docs:")
    else:
        print(f"\n⚡ Theme/Topic Cluster {topic_id}:")

    # Print top keywords representing this cluster
    keywords = [word for word, score in topic_model.get_topic(topic_id)]
    print(f"   Keywords: {', '.join(keywords)}")

    # Print the documents that fell into this cluster
    cluster_docs = [doc for doc, t in zip(documents, topics) if t == topic_id]
    for doc in cluster_docs:
        print(f"   - {doc}")
