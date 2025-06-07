import os
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

# Initialize summarization model
summarizer = pipeline(
    "summarization", model="sshleifer/distilbart-cnn-12-6", device=0)

# Load SearxNG-scraped data
documents = []
for file in os.listdir("searxng_data"):
    if file.endswith(".txt"):
        with open(os.path.join("searxng_data", file), "r", encoding="utf-8") as f:
            documents.append(f.read())

# Summarize short documents (<200 tokens)


def process_doc(text):
    token_count = len(text.split())
    if token_count < 200:
        try:
            summary = summarizer(text, max_length=200, min_length=50, do_sample=False)[
                0]["summary_text"]
            return summary
        except:
            return text
    return text


processed_docs = [process_doc(doc) for doc in documents]

# Split long documents


def split_document(text, chunk_size=600, overlap=150):
    chunks = []
    lines = text.split("\n")
    current_chunk = ""
    current_len = 0
    for line in lines:
        line_len = len(line.split())
        if line.startswith(("#", "##", "###")) or current_len + line_len > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = line
            current_len = line_len
        else:
            current_chunk += "\n" + line
            current_len += line_len
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


chunks = []
for doc in processed_docs:
    chunks.extend(split_document(doc))

# Cluster redundant chunks with DBSCAN
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
chunk_embeddings = embedder.encode(chunks, convert_to_numpy=True)
dbscan = DBSCAN(eps=0.5, min_samples=2, metric="cosine")
labels = dbscan.fit_predict(chunk_embeddings)
unique_chunks = []
for label in set(labels):
    if label != -1:
        cluster_chunks = [chunks[i]
                          for i in range(len(chunks)) if labels[i] == label]
        unique_chunks.append(cluster_chunks[0])
    else:
        unique_chunks.extend([chunks[i]
                             for i in range(len(chunks)) if labels[i] == -1])

# Output sample chunks
for i, chunk in enumerate(unique_chunks[:3]):
    print(f"Chunk {i+1}: {chunk[:200]}...")
