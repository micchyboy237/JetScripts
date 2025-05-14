# Load and clean JSON data
import json


def load_json_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    all_text = []
    for item in data:
        text = f"{item.get('header', '')}\n{item.get('parent_header', '')}\n{clean_text(item.get('content', ''))}".strip()
        all_text.append(text)
    return data, all_text


def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
    return text.strip()


json_path = f"{DATA_DIR}/web_scraped_data.json"
json_data, extracted_text = load_json_data(json_path)

# Use JSON entries as chunks directly
text_chunks = extracted_text
logger.debug(f"Number of text chunks: {len(text_chunks)}")

# Generate embeddings
response = create_embeddings(text_chunks, batch_size=32)

# Modified context-enriched search


def context_enriched_search(query, json_data, embeddings, k=1, context_size=1):
    query_embedding = embed_func(query)
    similarity_scores = []
    for i, (chunk_embedding, item) in enumerate(zip(embeddings, json_data)):
        similarity_score = cosine_similarity(
            np.array(query_embedding), np.array(chunk_embedding))
        if item.get('parent_header') and item['parent_header'].lower() in query.lower():
            similarity_score *= 1.1
        similarity_scores.append((i, similarity_score))
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_index = similarity_scores[0][0]
    start = max(0, top_index - context_size)
    end = min(len(json_data), top_index + context_size + 1)
    return [f"{json_data[i].get('header', '')}\n{json_data[i].get('parent_header', '')}\n{json_data[i].get('content', '')}".strip() for i in range(start, end)]


# Rest of the code remains the same
top_chunks = context_enriched_search(
    query, json_data, response, k=1, context_size=1)
ai_response = generate_response(query, system_prompt, top_chunks)
evaluation_score = evaluate_response(query, ai_response, true_answer)
