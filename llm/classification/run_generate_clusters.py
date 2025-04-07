from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

docs = [
    "Build mobile apps using React Native.",
    "Use Redux or Zustand for state management.",
    "Send push notifications with Firebase.",
    "Use Tailwind CSS for styling.",
    "Deploy apps on Google Play.",
]

embed_model = "all-MiniLM-L12-v2"

model = SentenceTransformer(embed_model)
embeddings = model.encode(docs)

# Cluster
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(embeddings)

# View groups
for label in set(labels):
    print(f"\nLabel: Cluster {label}")
    for i, doc in enumerate(docs):
        if labels[i] == label:
            print(" -", doc)
