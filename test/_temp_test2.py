from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.nn.functional import cosine_similarity
from typing import List
from transformers import PreTrainedTokenizer
from typing import List, Optional
import re
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, TypedDict, Set, Union
from transformers import PreTrainedTokenizer, PreTrainedModel

# Download NLTK data (run once)
nltk.download('punkt', quiet=True)

# Define typed dictionaries for structured return types


class SearchResult(TypedDict):
    score: float
    doc_index: int
    text: str

# Mean Pooling - Take attention mask into account for correct averaging


def mean_pooling(model_output: Tuple[torch.Tensor, ...], attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings: torch.Tensor = model_output[0]
    input_mask_expanded: torch.Tensor = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Cosine similarity function


def cosine_similarity(embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(embeddings1, embeddings2, dim=1)

# Function to compute embeddings with batching


def get_embeddings(
    texts: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    use_mean_pooling: bool = True,
    batch_size: int = 32
) -> torch.Tensor:
    """
    Compute embeddings for texts in batches.
    Args:
        texts: List of text strings.
        model: Transformer model.
        tokenizer: Tokenizer for the model.
        use_mean_pooling: Whether to use mean pooling or CLS token.
        batch_size: Number of texts per batch.
    Returns:
        Embeddings for all texts.
    """
    embeddings: List[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        batch_texts: List[str] = texts[i:i + batch_size]
        encoded_input = tokenizer(
            batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512
        )
        with torch.no_grad():
            model_output = model(**encoded_input)
        if use_mean_pooling:
            batch_embeddings: torch.Tensor = mean_pooling(
                model_output, encoded_input['attention_mask'])
        else:
            batch_embeddings = model_output[0][:, 0, :]  # CLS token embedding
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)

# Preprocessing function


def preprocess_text(
    content: str,
    tokenizer: PreTrainedTokenizer,
    header: Optional[str] = None,
    min_length: int = 20,
    max_length: int = 300,
    overlap: int = 100,
    debug: bool = False
) -> List[str]:
    """
    Preprocesses raw text for similarity search, ensuring complete sentences in all segments.
    Uses actual encoded input_ids length for max_length comparison.
    """
    # Validate input
    if not isinstance(content, str) or not content.strip():
        if debug:
            print("Warning: Empty or invalid content provided.")
        return []

    if debug:
        print(f"Input content length: {len(content)} characters")

    # Validate overlap
    if overlap >= max_length or overlap < 0:
        overlap = 0
        if debug:
            print(f"Warning: Invalid overlap {overlap}, setting to 0.")

    # Step 1: Clean text
    content = re.sub(r'^#{1,6}\s*', '', content, flags=re.MULTILINE)
    content = re.sub(r'\s+', ' ', content.strip())  # Normalize whitespace
    content = re.sub(r'\n+', '\n', content)  # Normalize newlines
    if header:
        header = re.sub(r'^#{1,6}\s*', '', header, flags=re.MULTILINE)

    if debug:
        print(f"Cleaned content length: {len(content)} characters")
        if header:
            print(f"Cleaned header length: {len(header)} characters")

    # Step 2: Split into lines for metadata detection
    lines: List[str] = content.split('\n')
    processed_texts: List[str] = []
    metadata_buffer: List[str] = []
    is_metadata: bool = False

    # Step 3: Process lines
    for line in lines:
        line = line.strip()
        if not line:
            if debug:
                print("Skipping empty line")
            continue

        if debug:
            print(f"Processing line: {line[:50]}...")

        # Detect metadata
        if (len(line.split()) < 5 or
            any(keyword in line.lower() for keyword in ['studio', 'source', 'theme', 'demographic']) or
                re.match(r'^\d+\.\d+$|^[0-9K]+$', line)):
            metadata_buffer.append(line)
            is_metadata = True
            if debug:
                print(f"Classified as metadata: {line}")
        else:
            # Process metadata buffer if any
            if metadata_buffer:
                metadata_text: str = ' '.join(metadata_buffer)
                if debug:
                    print(f"Processing metadata: {metadata_text[:50]}...")
                if len(metadata_text) >= min_length:
                    sentences: List[str] = sent_tokenize(metadata_text)
                    current_segment: str = ""
                    for sent in sentences:
                        temp_segment = (current_segment + " " +
                                        sent).strip() if current_segment else sent
                        segment_with_header = f"{header}\n{temp_segment}".strip(
                        ) if header else temp_segment
                        encoded = tokenizer(
                            segment_with_header, add_special_tokens=True, return_tensors='pt')
                        token_length = encoded['input_ids'].shape[1]
                        if token_length <= max_length:
                            current_segment = temp_segment
                        else:
                            if current_segment:
                                segment_with_header = f"{header}\n{current_segment}".strip(
                                ) if header else current_segment
                                encoded = tokenizer(
                                    segment_with_header, add_special_tokens=True, return_tensors='pt')
                                if encoded['input_ids'].shape[1] <= max_length:
                                    processed_texts.append(segment_with_header)
                                    if debug:
                                        print(
                                            f"Added metadata segment: {segment_with_header[:50]}...")
                                else:
                                    if debug:
                                        print(
                                            f"Discarded metadata segment (too long): {segment_with_header[:50]}...")
                            current_segment = sent if len(
                                sent) <= max_length else ""
                    if current_segment and len(current_segment) >= min_length:
                        segment_with_header = f"{header}\n{current_segment}".strip(
                        ) if header else current_segment
                        encoded = tokenizer(
                            segment_with_header, add_special_tokens=True, return_tensors='pt')
                        if encoded['input_ids'].shape[1] <= max_length:
                            processed_texts.append(segment_with_header)
                            if debug:
                                print(
                                    f"Added metadata segment: {segment_with_header[:50]}...")
                        else:
                            if debug:
                                print(
                                    f"Discarded metadata segment (too long): {segment_with_header[:50]}...")
                else:
                    if debug:
                        print(
                            f"Discarded metadata (too short): {metadata_text}")
                metadata_buffer = []
                is_metadata = False

            # Process narrative text
            sentences = sent_tokenize(line)
            current_segment = ""
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < min_length or re.match(r'^\[.*\]$', sent):
                    if debug:
                        print(
                            f"Discarded sentence (too short or bracketed): {sent}")
                    continue

                temp_segment = (current_segment + " " +
                                sent).strip() if current_segment else sent
                segment_with_header = f"{header}\n{temp_segment}".strip(
                ) if header else temp_segment
                encoded = tokenizer(
                    segment_with_header, add_special_tokens=True, return_tensors='pt')
                token_length = encoded['input_ids'].shape[1]
                if token_length <= max_length:
                    current_segment = temp_segment
                else:
                    if current_segment:
                        segment_with_header = f"{header}\n{current_segment}".strip(
                        ) if header else current_segment
                        encoded = tokenizer(
                            segment_with_header, add_special_tokens=True, return_tensors='pt')
                        if encoded['input_ids'].shape[1] <= max_length:
                            processed_texts.append(segment_with_header)
                            if debug:
                                print(
                                    f"Added narrative segment: {segment_with_header[:50]}...")
                        else:
                            if debug:
                                print(
                                    f"Discarded narrative segment (too long): {segment_with_header[:50]}...")
                        current_segment = ""
                    current_segment = sent if len(sent) <= max_length else ""

            if current_segment and len(current_segment) >= min_length:
                segment_with_header = f"{header}\n{current_segment}".strip(
                ) if header else current_segment
                encoded = tokenizer(
                    segment_with_header, add_special_tokens=True, return_tensors='pt')
                if encoded['input_ids'].shape[1] <= max_length:
                    processed_texts.append(segment_with_header)
                    if debug:
                        print(
                            f"Added narrative segment: {segment_with_header[:50]}...")
                else:
                    if debug:
                        print(
                            f"Discarded narrative segment (too long): {segment_with_header[:50]}...")

    # Process remaining metadata
    if metadata_buffer:
        metadata_text = ' '.join(metadata_buffer)
        if len(metadata_text) >= min_length:
            sentences = sent_tokenize(metadata_text)
            current_segment = ""
            for sent in sentences:
                temp_segment = (current_segment + " " +
                                sent).strip() if current_segment else sent
                segment_with_header = f"{header}\n{temp_segment}".strip(
                ) if header else temp_segment
                encoded = tokenizer(
                    segment_with_header, add_special_tokens=True, return_tensors='pt')
                token_length = encoded['input_ids'].shape[1]
                if token_length <= max_length:
                    current_segment = temp_segment
                else:
                    if current_segment:
                        segment_with_header = f"{header}\n{current_segment}".strip(
                        ) if header else current_segment
                        encoded = tokenizer(
                            segment_with_header, add_special_tokens=True, return_tensors='pt')
                        if encoded['input_ids'].shape[1] <= max_length:
                            processed_texts.append(segment_with_header)
                            if debug:
                                print(
                                    f"Added metadata segment: {segment_with_header[:50]}...")
                        else:
                            if debug:
                                print(
                                    f"Discarded metadata segment (too long): {segment_with_header[:50]}...")
                    current_segment = sent if len(sent) <= max_length else ""
            if current_segment and len(current_segment) >= min_length:
                segment_with_header = f"{header}\n{current_segment}".strip(
                ) if header else current_segment
                encoded = tokenizer(
                    segment_with_header, add_special_tokens=True, return_tensors='pt')
                if encoded['input_ids'].shape[1] <= max_length:
                    processed_texts.append(segment_with_header)
                    if debug:
                        print(
                            f"Added metadata segment: {segment_with_header[:50]}...")
                else:
                    if debug:
                        print(
                            f"Discarded metadata segment (too long): {segment_with_header[:50]}...")
        else:
            if debug:
                print(f"Discarded metadata (too short): {metadata_text}")

    # Step 4: Deduplicate
    seen: Set[str] = set()
    unique_texts: List[str] = []
    for text in processed_texts:
        if text not in seen:
            unique_texts.append(text)
            seen.add(text)

    # Step 5: Debug output
    if debug:
        print("Final Preprocessed Texts:")
        for i, text in enumerate(unique_texts):
            encoded = tokenizer(
                text, add_special_tokens=True, return_tensors='pt')
            token_count = encoded['input_ids'].shape[1]
            print(f"{i+1}. ({len(text)} chars, {token_count} tokens) {text}")

    if not unique_texts and debug:
        print("Warning: No valid segments produced after preprocessing.")

    return unique_texts

# Function to preprocess query


def preprocess_query(
    query: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 300
) -> str:
    """
    Preprocesses the query to match corpus preprocessing.
    Uses actual encoded input_ids length for max_length comparison.
    Args:
        query: Raw query text.
        tokenizer: Tokenizer for the model to compute token length.
        max_length: Maximum length for the query (in tokens).
    Returns:
        Preprocessed query.
    """
    if not isinstance(query, str) or not query.strip():
        return ""
    query = re.sub(r'\s+', ' ', query.strip())  # Normalize whitespace
    query = re.sub(r'[^\w\s.,!?]', '', query)  # Remove special characters
    encoded = tokenizer(query, add_special_tokens=True, return_tensors='pt')
    if encoded['input_ids'].shape[1] > max_length:
        # Truncate to max_length tokens
        truncated_ids = encoded['input_ids'][0, :max_length]
        query = tokenizer.decode(truncated_ids, skip_special_tokens=True)
        # Ensure we don't cut off in the middle of a word
        split_point = query.rfind(' ')
        if split_point != -1:
            query = query[:split_point].strip()
    return query

# Function to perform similarity search


def similarity_search(
    query: str,
    texts: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    use_mean_pooling: bool = True,
    top_k: int = 5,
    threshold: float = 0.5,
    debug: bool = False
) -> List[SearchResult]:
    """
    Performs similarity search on texts using transformer embeddings.
    Args:
        query: Query text.
        texts: List of text segments to search.
        model: Transformer model.
        tokenizer: Tokenizer for the model.
        use_mean_pooling: Whether to use mean pooling or CLS token.
        top_k: Number of top results to return.
        threshold: Minimum similarity score for results to be included.
        debug: Whether to print top results for debugging.
    Returns:
        List of dictionaries containing score, doc_index, and text.
    """
    # Validate inputs
    if not query or not texts:
        if debug:
            print("Warning: Empty query or texts provided.")
        return []

    # Get embeddings
    query_embedding: torch.Tensor = get_embeddings(
        [query], model, tokenizer, use_mean_pooling)
    text_embeddings: torch.Tensor = get_embeddings(
        texts, model, tokenizer, use_mean_pooling)

    # Check if embeddings are empty
    if query_embedding.numel() == 0 or text_embeddings.numel() == 0:
        if debug:
            print("Warning: No valid embeddings generated.")
        return []

    # Compute similarities
    similarities: torch.Tensor = cosine_similarity(
        query_embedding, text_embeddings)
    similarities_np = similarities.cpu().numpy()

    # Get top-k results
    top_k_indices = similarities_np.argsort()[-top_k:][::-1]
    top_k_scores = similarities_np[top_k_indices]

    # Debug output: Print top results regardless of threshold
    if debug:
        print("Top Similarity Search Results:")
        for i, (idx, score) in enumerate(zip(top_k_indices[:5], top_k_scores[:5]), 1):
            print(
                f"{i}. Score: {score:.4f}\n"
                f"Index: {idx}\n"
                f"Text: {texts[idx]}\n"
            )
        if not top_k_scores.size:
            print("No results available.")

    # Build results list: Apply threshold for returned results
    results: List[dict] = [
        {
            'score': float(top_k_scores[i]),
            'doc_index': int(idx),
            'text': texts[idx]
        }
        for i, idx in enumerate(top_k_indices)
        if top_k_scores[i] >= threshold
    ]

    return results


def main() -> None:
    model_path = 'sentence-transformers/all-MiniLM-L12-v2'

    # Load model and tokenizer
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_path)
    model: PreTrainedModel = AutoModel.from_pretrained(model_path)

    # Preprocessing parameters
    min_length: int = 20
    max_length: int = 150
    overlap: int = 20
    top_k: int = 3
    threshold: float = 0.7

    # Example query
    query: str = 'List upcoming isekai anime this year (2024-2025).'
    query = preprocess_query(query, tokenizer, max_length=max_length)

    print(f"Query: {query}")
    print(f"Model: {model_path}")
    print(f"Min Length: {min_length}")
    print(f"Max Length: {max_length} tokens")
    print(f"Overlap: {overlap}")
    print(f"Top K: {top_k}")
    print(f"Threshold: {threshold}")
    print()

    # Example content
    content: str = """## Naruto: Shippuuden Movie 6 - Road to Ninja
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
    header = content.split('\n')[0]  # Get first line as header
    content = '\n'.join(content.split('\n')[1:])  # Remove header from content

    # Preprocess content with debug mode
    texts: List[str] = preprocess_text(
        content, tokenizer, header=header, min_length=min_length, max_length=max_length, overlap=overlap, debug=True
    )

    # Test with mean pooling
    print("\n=== Similarity Search with Mean Pooling ===\n")
    results_mean: List[SearchResult] = similarity_search(
        query, texts, model, tokenizer, use_mean_pooling=True,
        top_k=top_k, threshold=threshold, debug=True
    )
    for i, result in enumerate(results_mean, 1):
        print(
            f"{i}. Score: {result['score']:.4f}\nIndex: {result['doc_index']}\nText: {result['text']}\n")

    # Test with CLS token
    print("\n=== Similarity Search with CLS Token ===\n")
    results_cls: List[SearchResult] = similarity_search(
        query, texts, model, tokenizer, use_mean_pooling=False,
        top_k=top_k, threshold=threshold, debug=True
    )
    for i, result in enumerate(results_cls, 1):
        print(
            f"{i}. Score: {result['score']:.4f}\nIndex: {result['doc_index']}\nText: {result['text']}\n")


if __name__ == "__main__":
    main()
