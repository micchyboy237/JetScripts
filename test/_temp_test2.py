import re
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Download NLTK data (run once)
nltk.download('punkt', quiet=True)

# Mean Pooling - Take attention mask into account for correct averaging


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Cosine similarity function


def cosine_similarity(embeddings1, embeddings2):
    return F.cosine_similarity(embeddings1, embeddings2, dim=1)

# Function to compute embeddings with batching


def get_embeddings(texts, model, tokenizer, use_mean_pooling=True, batch_size=32):
    """
    Compute embeddings for texts in batches.
    Args:
        texts (list): List of text strings.
        model: Transformer model.
        tokenizer: Tokenizer for the model.
        use_mean_pooling (bool): Whether to use mean pooling or CLS token.
        batch_size (int): Number of texts per batch.
    Returns:
        torch.Tensor: Embeddings for all texts.
    """
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded_input = tokenizer(
            batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            model_output = model(**encoded_input)
        if use_mean_pooling:
            batch_embeddings = mean_pooling(
                model_output, encoded_input['attention_mask'])
        else:
            batch_embeddings = model_output[0][:, 0, :]  # CLS token embedding
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)

# Preprocessing function


def preprocess_text(content, min_length=20, max_length=300, overlap=100, debug=False):
    """
    Preprocesses raw text for similarity search, ensuring complete sentences in all segments.
    Args:
        content (str): Raw input text.
        min_length (int): Minimum length for valid text segments.
        max_length (int): Maximum length for text segments.
        overlap (int): Number of characters to overlap between split segments.
        debug (bool): Whether to print preprocessed texts for debugging.
    Returns:
        list: List of preprocessed text segments with complete sentences.
    """
    # Validate input
    if not isinstance(content, str) or not content.strip():
        if debug:
            print("Warning: Empty or invalid content provided.")
        return []

    # Validate overlap
    if overlap >= max_length or overlap < 0:
        overlap = 0
        if debug:
            print(f"Warning: Invalid overlap {overlap}, setting to 0.")

    # Step 1: Clean text
    content = re.sub(r'^#{1,6}\s*', '', content, flags=re.MULTILINE)
    content = re.sub(r'\s+', ' ', content.strip())  # Normalize whitespace
    content = re.sub(r'\n+', '\n', content)  # Normalize newlines

    # Step 2: Split into lines for metadata detection
    lines = content.split('\n')
    processed_texts = []
    metadata_buffer = []
    is_metadata = False

    # Step 3: Process lines
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect metadata
        if len(line.split()) < 5 or any(keyword in line.lower() for keyword in ['studio', 'source', 'theme', 'demographic']) or re.match(r'^\d+\.\d+$|^[0-9K]+$', line):
            metadata_buffer.append(line)
            is_metadata = True
        else:
            # Process metadata buffer if any
            if metadata_buffer:
                metadata_text = ' '.join(metadata_buffer)
                if len(metadata_text) >= min_length:
                    # Split metadata if too long, but avoid partial sentences
                    sentences = sent_tokenize(metadata_text)
                    current_segment = ""
                    for sent in sentences:
                        if len(current_segment) + len(sent) + 1 <= max_length:
                            current_segment += (" " +
                                                sent) if current_segment else sent
                        else:
                            if current_segment:
                                processed_texts.append(current_segment)
                            current_segment = sent if len(
                                sent) <= max_length else ""
                            # Handle long sentences
                            while len(current_segment) > max_length:
                                sub_sentences = sent_tokenize(
                                    current_segment[:max_length])
                                if sub_sentences:
                                    last_sub = sub_sentences[-1]
                                    split_point = current_segment.find(
                                        last_sub) + len(last_sub)
                                    processed_texts.append(
                                        current_segment[:split_point].strip())
                                    overlap_start = max(
                                        0, split_point - overlap)
                                    current_segment = current_segment[overlap_start:].strip(
                                    )
                                else:
                                    split_point = current_segment.rfind(
                                        ' ', 0, max_length)
                                    if split_point == -1:
                                        split_point = max_length
                                    processed_texts.append(
                                        current_segment[:split_point].strip())
                                    overlap_start = max(
                                        0, split_point - overlap)
                                    current_segment = current_segment[overlap_start:].strip(
                                    )
                    if current_segment and len(current_segment) >= min_length:
                        processed_texts.append(current_segment)
                metadata_buffer = []
                is_metadata = False

            # Process narrative text
            sentences = sent_tokenize(line)
            current_segment = ""
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < min_length or re.match(r'^\[.*\]$', sent):
                    continue

                # Build segment with complete sentences
                if len(current_segment) + len(sent) + 1 <= max_length:
                    current_segment += (" " +
                                        sent) if current_segment else sent
                else:
                    if current_segment:
                        processed_texts.append(current_segment)
                        # Start new segment with overlap
                        sub_sentences = sent_tokenize(current_segment)
                        if sub_sentences:
                            last_complete = sub_sentences[-1]
                            split_point = current_segment.rfind(
                                last_complete) + len(last_complete)
                            overlap_start = max(0, split_point - overlap)
                            overlap_text = current_segment[overlap_start:].strip(
                            )
                            current_segment = overlap_text if len(
                                overlap_text) >= min_length else ""
                        else:
                            current_segment = ""

                    # Handle long sentence
                    while len(sent) > max_length:
                        sub_sentences = sent_tokenize(sent[:max_length])
                        if sub_sentences:
                            last_sub = sub_sentences[-1]
                            split_point = sent.find(last_sub) + len(last_sub)
                            processed_texts.append(sent[:split_point].strip())
                            overlap_start = max(0, split_point - overlap)
                            sent = sent[overlap_start:].strip()
                        else:
                            split_point = sent.rfind(' ', 0, max_length)
                            if split_point == -1:
                                split_point = max_length
                            processed_texts.append(sent[:split_point].strip())
                            overlap_start = max(0, split_point - overlap)
                            sent = sent[overlap_start:].strip()

                    if len(sent) >= min_length:
                        current_segment = sent

            # Add remaining segment
            if current_segment and len(current_segment) >= min_length:
                processed_texts.append(current_segment)

    # Process remaining metadata
    if metadata_buffer:
        metadata_text = ' '.join(metadata_buffer)
        if len(metadata_text) >= min_length:
            sentences = sent_tokenize(metadata_text)
            current_segment = ""
            for sent in sentences:
                if len(current_segment) + len(sent) + 1 <= max_length:
                    current_segment += (" " +
                                        sent) if current_segment else sent
                else:
                    if current_segment:
                        processed_texts.append(current_segment)
                    current_segment = sent if len(sent) <= max_length else ""
                    while len(current_segment) > max_length:
                        sub_sentences = sent_tokenize(
                            current_segment[:max_length])
                        if sub_sentences:
                            last_sub = sub_sentences[-1]
                            split_point = current_segment.find(
                                last_sub) + len(last_sub)
                            processed_texts.append(
                                current_segment[:split_point].strip())
                            overlap_start = max(0, split_point - overlap)
                            current_segment = current_segment[overlap_start:].strip(
                            )
                        else:
                            split_point = current_segment.rfind(
                                ' ', 0, max_length)
                            if split_point == -1:
                                split_point = max_length
                            processed_texts.append(
                                current_segment[:split_point].strip())
                            overlap_start = max(0, split_point - overlap)
                            current_segment = current_segment[overlap_start:].strip(
                            )
            if current_segment and len(current_segment) >= min_length:
                processed_texts.append(current_segment)

    # Step 4: Deduplicate
    seen = set()
    unique_texts = []
    for text in processed_texts:
        if text not in seen:
            unique_texts.append(text)
            seen.add(text)

    # Step 5: Debug output
    if debug:
        print("Preprocessed Texts:")
        for i, text in enumerate(unique_texts):
            print(f"{i+1}. ({len(text)} chars) {text}")

    return unique_texts
# Function to preprocess query


def preprocess_query(query, max_length=300):
    """
    Preprocesses the query to match corpus preprocessing.
    Args:
        query (str): Raw query text.
        max_length (int): Maximum length for the query.
    Returns:
        str: Preprocessed query.
    """
    if not isinstance(query, str) or not query.strip():
        return ""
    query = re.sub(r'\s+', ' ', query.strip())  # Normalize whitespace
    query = re.sub(r'[^\w\s.,!?]', '', query)  # Remove special characters
    if len(query) > max_length:
        split_point = query.rfind(' ', 0, max_length)
        if split_point == -1:
            split_point = max_length
        query = query[:split_point].strip()
    return query

# Function to perform similarity search


def similarity_search(query, texts, model, tokenizer, use_mean_pooling=True, top_k=5):
    query_embedding = get_embeddings(
        [query], model, tokenizer, use_mean_pooling)
    text_embeddings = get_embeddings(texts, model, tokenizer, use_mean_pooling)
    similarities = cosine_similarity(query_embedding, text_embeddings)
    similarities = similarities.cpu().numpy()
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    top_k_scores = similarities[top_k_indices]
    results = [(texts[idx], top_k_scores[i])
               for i, idx in enumerate(top_k_indices)]
    return results


def main():
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    # Preprocessing parameters
    min_length = 20
    max_length = 300
    overlap = 100

    # Example content
    content = """## Naruto: Shippuuden Movie 6 - Road to Ninja
Movie, 2012 Finished 1 ep, 109 min
Action Adventure Fantasy
Naruto: Shippuuden Movie 6 - Road to Ninja
Returning home to Konohagakure, the young ninja celebrate defeating a group of supposed Akatsuki members. Naruto Uzumaki and Sakura Haruno, however, feel differently. Naruto is jealous of his comrades' congratulatory families, wishing for the presence of his own parents. Sakura, on the other hand, is angry at her embarrassing parents, and wishes for no parents at all. The two clash over their opposing ideals, but are faced with a more pressing matter when the masked Madara Uchiha suddenly appears and transports them to an alternate world. In this world, Sakura's parents are considered heroes--for they gave their lives to protect Konohagakure from the Nine-Tailed Fox attack 10 years ago. Consequently, Naruto's parents, Minato Namikaze and Kushina Uzumaki, are alive and well. Unable to return home or find the masked Madara, Naruto and Sakura stay in this new world and enjoy the changes they have always longed for. All seems well for the two ninja, until an unexpected threat emerges that pushes Naruto and Sakura to not only fight for the Konohagakure of the alternate world, but also to find a way back to their own. [Written by MAL Rewrite]
Studio Pierrot
Source Manga
Theme Isekai
Demographic Shounen
7.68
366K
Add to My List"""

    # Preprocess content with debug mode
    texts = preprocess_text(content, min_length=min_length,
                            max_length=max_length, overlap=overlap, debug=True)

    # Example query
    query = "Naruto and Sakura in an alternate world"
    query = preprocess_query(query, max_length=max_length)

    # Test with mean pooling
    print("\n=== Similarity Search with Mean Pooling ===")
    print(f"Query: {query}")
    print(f"Min Length: {min_length}")
    print(f"Max Length: {max_length}")
    print(f"Overlap: {overlap}")
    print()
    results_mean = similarity_search(
        query, texts, model, tokenizer, use_mean_pooling=True, top_k=3)
    for i, (text, score) in enumerate(results_mean, 1):
        print(f"{i}. Score: {score:.4f}\nText: {text}\n")

    # Test with CLS token (no mean pooling) - Note: CLS token is less reliable for this model
    print("\n=== Similarity Search with CLS Token ===")
    print(f"Query: {query}")
    print(f"Min Length: {min_length}")
    print(f"Max Length: {max_length}")
    print(f"Overlap: {overlap}")
    print()
    results_cls = similarity_search(
        query, texts, model, tokenizer, use_mean_pooling=False, top_k=3)
    for i, (text, score) in enumerate(results_cls, 1):
        print(f"{i}. Score: {score:.4f}\nText: {text}\n")


if __name__ == "__main__":
    main()
