import json
from jet.logger import logger
import spacy
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag, WordNetLemmatizer
from typing import List, Optional, Set, Tuple
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
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Define typed dictionaries for structured return types


class SearchResult(TypedDict):
    rank: int
    score: float
    doc_index: int
    text: str
    tokens: int

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
    max_length: int = 150,  # Adjusted to match error context
    overlap: int = 20,     # Adjusted to match error context
    debug: bool = False
) -> List[Tuple[str, List[str]]]:
    """
    Preprocesses raw text for similarity search, ensuring complete sentences in all segments.
    Generates tags using NLTK (POS) and SpaCy (NER, noun chunks) to improve vector search context.
    Returns a list of tuples (segment, tags).
    Uses actual encoded input_ids length for max_length comparison.
    """
    # Initialize NLP tools
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    # Enable parser by removing disable=['parser']
    nlp = spacy.load('en_core_web_sm')

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

    # Step 2: Split into lines
    lines: List[str] = content.split('\n')
    processed_segments: List[Tuple[str, List[str]]] = []

    def generate_tags(text: str) -> List[str]:
        """Generate tags for a text segment using NLTK and SpaCy."""
        tags = set()

        # NLTK: POS tagging
        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)
        for word, pos in pos_tags:
            if pos.startswith(('NN', 'VB', 'JJ')) and word not in stop_words and len(word) > 2:
                lemma = lemmatizer.lemmatize(
                    word, pos='v' if pos.startswith('VB') else 'n')
                tags.add(lemma)

        # SpaCy: NER and noun chunks
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ('PERSON', 'ORG', 'GPE', 'PRODUCT'):
                tags.add(ent.text.lower())
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower().strip()
            if len(chunk_text) > 2 and chunk_text not in stop_words:
                tags.add(chunk_text)

        # Filter and sort tags
        return sorted([tag for tag in tags if len(tag) > 2 and tag not in stop_words])

    # Step 3: Process lines
    for line in lines:
        line = line.strip()
        if not line:
            if debug:
                print("Skipping empty line")
            continue

        if debug:
            print(f"Processing line: {line[:50]}...")

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
            encoded = tokenizer(segment_with_header,
                                add_special_tokens=True, return_tensors='pt')
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
                        tags = generate_tags(segment_with_header)
                        processed_segments.append((segment_with_header, tags))
                        if debug:
                            print(
                                f"Added segment: {segment_with_header[:50]}... with tags: {tags}")
                    else:
                        if debug:
                            print(
                                f"Discarded segment (too long): {segment_with_header[:50]}...")
                    current_segment = ""
                current_segment = sent if len(sent) <= max_length else ""

        if current_segment and len(current_segment) >= min_length:
            segment_with_header = f"{header}\n{current_segment}".strip(
            ) if header else current_segment
            encoded = tokenizer(segment_with_header,
                                add_special_tokens=True, return_tensors='pt')
            if encoded['input_ids'].shape[1] <= max_length:
                tags = generate_tags(segment_with_header)
                processed_segments.append((segment_with_header, tags))
                if debug:
                    print(
                        f"Added segment: {segment_with_header[:50]}... with tags: {tags}")
            else:
                if debug:
                    print(
                        f"Discarded segment (too long): {segment_with_header[:50]}...")

    # Step 4: Deduplicate
    seen: Set[str] = set()
    unique_segments: List[Tuple[str, List[str]]] = []
    for segment, tags in processed_segments:
        if segment not in seen:
            unique_segments.append((segment, tags))
            seen.add(segment)

    # Step 5: Debug output
    if debug:
        print("Final Preprocessed Segments with Tags:")
        for i, (segment, tags) in enumerate(unique_segments):
            encoded = tokenizer(
                segment, add_special_tokens=True, return_tensors='pt')
            token_count = encoded['input_ids'].shape[1]
            print(f"{i+1}. ({len(segment)} chars, {token_count} tokens) {segment}")
            print(f"   Tags: {tags}")

    if not unique_segments and debug:
        print("Warning: No valid segments produced after preprocessing.")

    return unique_segments

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
        List of SearchResult dictionaries containing rank, score, doc_index, text, and tokens.
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
        logger.gray(f"[DEBUG] Top {top_k} Similarity Search Results:")
        for i, (idx, score) in enumerate(zip(top_k_indices[:top_k], top_k_scores[:top_k]), 1):
            logger.log(
                f"Rank {i}:",
                f"Doc: {idx}, Tokens: {len(tokenizer.encode(texts[idx], add_special_tokens=True))}",
                f"\nScore: {score:.3f}",
                f"\n{texts[idx]}",
                colors=["ORANGE", "DEBUG", "SUCCESS", "WHITE"],
            )

    # Build results list: Apply threshold for returned results
    results: List[SearchResult] = [
        {
            'rank': i + 1,
            'score': float(top_k_scores[i]),
            'doc_index': int(idx),
            'text': texts[idx],
            'tokens': len(tokenizer.encode(texts[idx], add_special_tokens=True))
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
    text_keywords_tuples: List[Tuple[str, List[str]]] = preprocess_text(
        content, tokenizer, header=header, min_length=min_length, max_length=max_length, overlap=overlap, debug=True
    )

    # Extract just the text segments from the tuples
    texts: List[str] = [segment for segment, _ in text_keywords_tuples]

    # Test with mean pooling
    logger.info("\n=== Similarity Search with Mean Pooling ===\n")
    results_mean: List[SearchResult] = similarity_search(
        query, texts, model, tokenizer, use_mean_pooling=True,
        top_k=top_k, threshold=threshold, debug=True
    )
    for i, result in enumerate(results_mean, 1):
        logger.log(
            f"Rank {result['rank']}:",
            f"Doc: {result['doc_index']}, Tokens: {result['tokens']}",
            f"\nScore: {result['score']:.3f}",
            f"\n{json.dumps(result['text'])[:100]}...",
            colors=["ORANGE", "DEBUG", "SUCCESS", "WHITE"],
        )

    # Test with CLS token
    logger.info("\n=== Similarity Search with CLS Token ===\n")
    results_cls: List[SearchResult] = similarity_search(
        query, texts, model, tokenizer, use_mean_pooling=False,
        top_k=top_k, threshold=threshold, debug=True
    )
    for i, result in enumerate(results_cls, 1):
        logger.log(
            f"Rank {result['rank']}:",
            f"Doc: {result['doc_index']}, Tokens: {result['tokens']}",
            f"\nScore: {result['score']:.3f}",
            f"\n{json.dumps(result['text'])[:100]}...",
            colors=["ORANGE", "DEBUG", "SUCCESS", "WHITE"],
        )


if __name__ == "__main__":
    main()
