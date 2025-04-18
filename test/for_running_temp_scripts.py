import os
from jet.scrapers.utils import clean_text
from jet.wordnet.sentence import split_sentences
import torch
import numpy as np
from jet.file.utils import load_file, save_file
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Load the tokenizer and model
model_name = "microsoft/MiniLM-L12-H384-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


# Function to get normalized text embeddings using mean pooling
def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt",
                       padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(last_hidden.size()).float()
    summed = (last_hidden * input_mask_expanded).sum(1)
    counts = input_mask_expanded.sum(1)
    mean_pooled = summed / counts
    embedding = mean_pooled.numpy()
    return normalize(embedding)


# Load data
data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_server/generated/search/top_anime_romantic_comedy_reddit_2024-2025/top_context_nodes.json"
data = load_file(data_file)

query = data["query"]
texts = [sentence.strip() for d in data["results"]
         for sentence in clean_text(d["text"]).splitlines()]
texts = [sentence for text in texts for sentence in split_sentences(text)]

# Compute normalized embeddings
query_embedding = get_text_embedding(query)
text_embeddings = [get_text_embedding(text) for text in texts]

# Calculate cosine similarity
results = []
for i, text in enumerate(texts):
    score = cosine_similarity(query_embedding, text_embeddings[i])[0][0]
    results.append({"score": score, "text": text})

# Sort results by similarity score
sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

# Print results
for result in sorted_results:
    print(f"Score: {result['score']:.4f}\nText: {result['text']}\n")

# Save results
output_dir = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
os.makedirs(output_dir, exist_ok=True)

save_file({
    "query": query,
    "all_count": len(sorted_results),
    "all_results": sorted_results,
}, os.path.join(output_dir, "all_results.json"))
