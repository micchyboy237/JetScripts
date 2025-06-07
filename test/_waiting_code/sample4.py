import os
from transformers import pipeline
import re

# Initialize summarization model
summarizer = pipeline(
    "summarization", model="sshleifer/distilbart-cnn-12-6", device=0)

# Load SearxNG-scraped data
documents = []
for file in os.listdir("searxng_data"):
    if file.endswith(".txt"):
        with open(os.path.join("searxng_data", file), "r", encoding="utf-8") as f:
            documents.append(f.read())

# Summarize long documents (>2000 tokens)


def summarize_long_doc(text):
    if len(text.split()) > 2000:
        try:
            summary = summarizer(text, max_length=500, min_length=100, do_sample=False)[
                0]["summary_text"]
            return summary
        except:
            return text  # Fallback if summarization fails
    return text


summarized_docs = [summarize_long_doc(doc) for doc in documents]

# Recursive chunking with header preservation


def recursive_chunk(text, chunk_size=600, overlap=150):
    chunks = []
    lines = text.split("\n")
    current_chunk = ""
    current_len = 0
    headers = []

    for line in lines:
        if line.startswith(("#", "##", "###")):
            headers.append(line)
        line_len = len(line.split())
        if current_len + line_len > chunk_size or line.startswith(("#", "##", "###")) and current_chunk:
            if current_chunk:
                chunks.append({"text": current_chunk.strip(),
                              "headers": headers.copy()})
            if line.startswith(("#", "##", "###")):
                current_chunk = line
                current_len = line_len
                headers = [line]
            else:
                current_chunk = line
                current_len = line_len
        else:
            current_chunk += "\n" + line
            current_len += line_len

    if current_chunk:
        chunks.append({"text": current_chunk.strip(), "headers": headers})

    # Add overlap
    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        headers = chunk["headers"]
        if i > 0:
            prev_chunk = chunks[i-1]["text"]
            overlap_text = " ".join(prev_chunk.split()[-overlap//4:])
            text = overlap_text + "\n" + text
        overlapped_chunks.append({"text": text, "headers": headers})

    return overlapped_chunks


chunks = []
for doc in summarized_docs:
    chunks.extend(recursive_chunk(doc))

# Output sample chunks
for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i+1}: {chunk['text'][:200]}...")
    print(f"Headers: {chunk['headers']}")
