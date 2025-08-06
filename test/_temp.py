from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
import pandas as pd
from typing import List, Tuple

# Sample documents
print("Using sample documents...")
docs = [
    "The stock market crashed today as tech stocks took a hit.",
    "A new study shows the health benefits of a Mediterranean diet.",
    "NASA plans to launch a new satellite to monitor climate change.",
    "Python is a popular programming language for data science.",
    "The local team won the championship after a thrilling final."
]

# Configure UMAP with a smaller number of components
umap_model = UMAP(n_components=2, n_neighbors=3, min_dist=0.1, random_state=42)

# Configure HDBSCAN for small datasets
hdbscan_model = HDBSCAN(min_cluster_size=2, min_samples=1)

# Fit BERTopic model with custom UMAP and HDBSCAN
print("Fitting BERTopic model...")
topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    embedding_model="all-MiniLM-L12-v2",  # More robust for small datasets
    min_topic_size=2
)
topics, probs = topic_model.fit_transform(docs)

# Get topic info as a DataFrame
print("Getting topic info...")
topic_info = topic_model.get_topic_info()

# Display the first few topics and their details
print("Top Topics:")
print(topic_info[['Topic', 'Name', 'Count']].head(10).to_string(index=False))

# Show a sample of documents with their assigned topics and probabilities
sample_df = pd.DataFrame({
    "Document": docs,
    "Assigned Topic": topics,
    "Topic Probability": [round(prob.max() if prob.size > 0 else 0.0, 4) for prob in probs]
})
print("\nSample Documents, Their Assigned Topics, and Probabilities:")
print(sample_df.to_string(index=False, max_colwidth=60))
