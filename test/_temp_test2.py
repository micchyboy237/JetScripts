import json
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag, WordNetLemmatizer
from typing import List, Optional, Set, Tuple, TypedDict
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from jet.logger import logger
from collections import Counter

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Define typed dictionaries


class SearchResult(TypedDict):
    rank: int
    score: float
    doc_index: int
    text: str
    tokens: int

# Mean Pooling


def mean_pooling(model_output: Tuple[torch.Tensor, ...], attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings: torch.Tensor = model_output[0]
    input_mask_expanded: torch.Tensor = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Cosine similarity


def cosine_similarity(embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(embeddings1, embeddings2, dim=1)

# Compute embeddings with batching


def get_embeddings(
    texts: List[Tuple[str, List[str]]],  # Accept (text, tags) tuples
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    use_mean_pooling: bool = True,
    batch_size: int = 32
) -> torch.Tensor:
    """
    Compute embeddings for texts with tags in batches.
    """
    combined_texts = [f"{text} {' '.join(tags)}" for text, tags in texts]
    embeddings: List[torch.Tensor] = []
    for i in range(0, len(combined_texts), batch_size):
        batch_texts: List[str] = combined_texts[i:i + batch_size]
        try:
            encoded_input = tokenizer(
                batch_texts, padding=True, truncation=True, return_tensors='pt',     max_length=512
            )
            with torch.no_grad():
                model_output = model(**encoded_input)
            if use_mean_pooling:
                batch_embeddings: torch.Tensor = mean_pooling(
                    model_output, encoded_input['attention_mask'])
            else:
                batch_embeddings = model_output[0][:, 0, :]  # CLS token
            embeddings.append(batch_embeddings)
        except Exception as e:
            logger.error(f"Embedding error in batch {i//batch_size + 1}: {e}")
    return torch.cat(embeddings, dim=0) if embeddings else torch.tensor([])

# Preprocessing function


def preprocess_text(
    content: str,
    tokenizer: PreTrainedTokenizer,
    header: Optional[str] = None,
    min_length: int = 50,
    max_length: int = 150,
    overlap: int = 20,
    debug: bool = False
) -> List[Tuple[str, List[str]]]:
    """
    Preprocesses raw text for similarity search, ensuring complete sentences in segments.
    Generates tags using NLTK and SpaCy with frequency-based filtering.
    """
    # Initialize NLP tools
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    nlp = spacy.load('en_core_web_sm')

    # Validate input
    if not isinstance(content, str) or not content.strip():
        if debug:
            logger.warning("Empty or invalid content provided.")
        return []

    if debug:
        logger.debug(f"Input content length: {len(content)} characters")

    # Validate overlap
    if overlap >= max_length or overlap < 0:
        overlap = 0
        if debug:
            logger.warning(f"Invalid overlap {overlap}, setting to 0.")

    # Step 1: Clean text
    content = re.sub(r'\[.*?\]', '', content)  # Remove bracketed text
    content = re.sub(r'^#{1,6}\s*', '', content, flags=re.MULTILINE)
    content = re.sub(r'\s+', ' ', content.strip())
    content = re.sub(r'\n+', '\n', content)
    if header:
        header = re.sub(r'^#{1,6}\s*', '', header, flags=re.MULTILINE)
        header = re.sub(r'\[.*?\]', '', header)
        # Limit header length
        header_tokens = tokenizer(header, add_special_tokens=True, return_tensors='pt')[
            'input_ids'][0]
        if len(header_tokens) > max_length // 2:
            header = tokenizer.decode(
                header_tokens[:max_length // 2], skip_special_tokens=True)
            if debug:
                logger.debug(f"Truncated header to {len(header)} characters")

    if debug:
        logger.debug(f"Cleaned content length: {len(content)} characters")
        if header:
            logger.debug(f"Cleaned header length: {len(header)} characters")

    # Step 2: Split into sentences
    sentences = sent_tokenize(content)
    processed_segments: List[Tuple[str, List[str]]] = []

    def generate_tags(texts: List[str]) -> List[List[str]]:
        """Generate tags for multiple texts with frequency-based filtering."""
        all_tags: List[Set[str]] = []
        for text in texts:
            tags = set()
            tokens = word_tokenize(text.lower())
            pos_tags = pos_tag(tokens)
            for word, pos in pos_tags:
                if pos.startswith(('NN', 'VB', 'JJ')) and word not in stop_words and len(word) > 2:
                    lemma = lemmatizer.lemmatize(
                        word, pos='v' if pos.startswith('VB') else 'n')
                    tags.add(lemma)
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ('PERSON', 'ORG', 'GPE', 'PRODUCT'):
                    tags.add(ent.text.lower())
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.lower().strip()
                if (len(chunk_text.split()) <= 2 and
                    len(chunk_text) > 2 and
                    not any(w in stop_words for w in chunk_text.split()) and
                        chunk_text not in stop_words):
                    tags.add(chunk_text)
            all_tags.append(tags)

        # Frequency-based filtering
        tag_counts = Counter(tag for tags in all_tags for tag in tags)
        filtered_tags = []
        for tags in all_tags:
            sorted_tags = sorted(
                [tag for tag in tags if tag_counts[tag] >= 1 and len(tag) > 2])
            # Limit to top 10 tags per segment
            filtered_tags.append(sorted_tags[:10])
        return filtered_tags

    def get_tokens(text: str) -> List[int]:
        """Encode text to tokens."""
        try:
            return tokenizer(text, add_special_tokens=True, return_tensors='pt')['input_ids'][0].tolist()
        except Exception as e:
            if debug:
                logger.error(f"Tokenization error: {e}")
            return []

    def decode_tokens(tokens: List[int]) -> str:
        """Decode tokens back to text."""
        try:
            return tokenizer.decode(tokens, skip_special_tokens=True).strip()
        except Exception as e:
            if debug:
                logger.error(f"Decoding error: {e}")
            return ""

    def get_sentence_aligned_overlap(tokens: List[int], sentences: List[str], sent_index: int) -> Tuple[str, List[int]]:
        """Get overlap text starting from the last complete sentence within overlap tokens."""
        if not tokens or overlap == 0:
            return "", []
        for i in range(len(tokens) - 1, max(0, len(tokens) - overlap - 1), -1):
            partial_tokens = tokens[i:]
            text = decode_tokens(partial_tokens)
            if text and sent_tokenize(text):
                if debug:
                    logger.debug(f"Overlap text: {text[:50]}...")
                return text, partial_tokens
        return "", []

    # Step 3: Process sentences into overlapping segments
    current_segment = ""
    current_tokens: List[int] = []
    sentence_index = 0

    while sentence_index < len(sentences):
        sent = sentences[sentence_index].strip()
        if len(sent) < min_length:
            if debug:
                logger.debug(f"Discarded sentence (too short): {sent[:50]}...")
            sentence_index += 1
            continue

        # Try adding the sentence to the current segment
        temp_segment = (current_segment + " " +
                        sent).strip() if current_segment else sent
        segment_with_header = f"{header}\n{temp_segment}".strip(
        ) if header else temp_segment
        temp_tokens = get_tokens(segment_with_header)

        if len(temp_tokens) <= max_length:
            current_segment = temp_segment
            current_tokens = temp_tokens
            # Save segment if it forms a complete unit (1–2 sentences)
            if len(sent_tokenize(current_segment)) >= 1:
                segment_with_header = f"{header}\n{current_segment}".strip(
                ) if header else current_segment
                if len(current_tokens) >= min_length:
                    processed_segments.append(
                        (segment_with_header, []))  # Tags added later
                    if debug:
                        logger.debug(
                            f"Added segment: {segment_with_header[:50]}...")
                    overlap_text, overlap_tokens = get_sentence_aligned_overlap(
                        current_tokens, sentences, sentence_index)
                    current_segment = overlap_text
                    current_tokens = overlap_tokens
            sentence_index += 1
        else:
            # Save current segment if it fits
            if current_segment and len(current_tokens) >= min_length:
                segment_with_header = f"{header}\n{current_segment}".strip(
                ) if header else current_segment
                processed_segments.append(
                    (segment_with_header, []))  # Tags added later
                if debug:
                    logger.debug(
                        f"Added segment: {segment_with_header[:50]}...")
                overlap_text, overlap_tokens = get_sentence_aligned_overlap(
                    current_tokens, sentences, sentence_index)
                current_segment = overlap_text
                current_tokens = overlap_tokens
            else:
                # Try the sentence alone
                segment_with_header = f"{header}\n{sent}".strip(
                ) if header else sent
                temp_tokens = get_tokens(segment_with_header)
                if len(temp_tokens) <= max_length and len(sent) >= min_length:
                    processed_segments.append(
                        (segment_with_header, []))  # Tags added later
                    if debug:
                        logger.debug(
                            f"Added segment: {segment_with_header[:50]}...")
                    overlap_text, overlap_tokens = get_sentence_aligned_overlap(
                        temp_tokens, sentences, sentence_index)
                    current_segment = overlap_text
                    current_tokens = overlap_tokens
                    sentence_index += 1
                else:
                    # Split long sentence
                    words = sent.split()
                    partial_segment = ""
                    partial_tokens: List[int] = []
                    for word in words:
                        temp_partial = (partial_segment + " " +
                                        word).strip() if partial_segment else word
                        temp_with_header = f"{header}\n{temp_partial}".strip(
                        ) if header else temp_partial
                        temp_partial_tokens = get_tokens(temp_with_header)
                        if len(temp_partial_tokens) <= max_length:
                            partial_segment = temp_partial
                            partial_tokens = temp_partial_tokens
                        else:
                            if len(partial_tokens) >= min_length:
                                partial_with_header = f"{header}\n{partial_segment}".strip(
                                ) if header else partial_segment
                                processed_segments.append(
                                    # Tags added later
                                    (partial_with_header, []))
                                if debug:
                                    logger.debug(
                                        f"Added partial segment: {partial_with_header[:50]}...")
                                overlap_text, overlap_tokens = get_sentence_aligned_overlap(
                                    partial_tokens, [partial_segment], 0)
                                partial_segment = overlap_text
                                partial_tokens = overlap_tokens
                            else:
                                partial_segment = ""
                                partial_tokens = []
                    if partial_segment and len(partial_tokens) >= min_length and len(partial_tokens) <= max_length:
                        partial_with_header = f"{header}\n{partial_segment}".strip(
                        ) if header else partial_segment
                        processed_segments.append(
                            (partial_with_header, []))  # Tags added later
                        if debug:
                            logger.debug(
                                f"Added partial segment: {partial_with_header[:50]}...")
                    sentence_index += 1
                    current_segment = ""
                    current_tokens = []

    # Save the last segment
    if current_segment and len(current_tokens) >= min_length and len(current_tokens) <= max_length:
        segment_with_header = f"{header}\n{current_segment}".strip(
        ) if header else current_segment
        processed_segments.append(
            (segment_with_header, []))  # Tags added later
        if debug:
            logger.debug(f"Added segment: {segment_with_header[:50]}...")

    # Step 4: Generate tags for all segments
    segment_texts = [segment for segment, _ in processed_segments]
    if segment_texts:
        tags_list = generate_tags(segment_texts)
        processed_segments = [(segment, tags) for (
            segment, _), tags in zip(processed_segments, tags_list)]

    # Step 5: Deduplicate
    seen: Set[str] = set()
    unique_segments: List[Tuple[str, List[str]]] = []
    for segment, tags in processed_segments:
        if segment not in seen:
            unique_segments.append((segment, tags))
            seen.add(segment)

    # Step 6: Debug output
    if debug:
        logger.debug("\nFinal Preprocessed Segments with Tags:")
        for i, (segment, tags) in enumerate(unique_segments):
            encoded = tokenizer(
                segment, add_special_tokens=True, return_tensors='pt')
            token_count = encoded['input_ids'].shape[1]
            logger.debug(
                f"{i+1}. ({len(segment)} chars, {token_count} tokens) {segment}")
            logger.debug(f"   Tags: {tags}")

    if not unique_segments and debug:
        logger.warning("No valid segments produced after preprocessing.")

    return unique_segments

# Query preprocessing


def preprocess_query(
    query: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 300
) -> str:
    """
    Preprocesses the query to match corpus preprocessing.
    """
    if not isinstance(query, str) or not query.strip():
        return ""
    query = re.sub(r'\[.*?\]', '', query)  # Remove bracketed text
    query = re.sub(r'\s+', ' ', query.strip())
    query = re.sub(r'[^\w\s.,!?]', '', query)
    encoded = tokenizer(query, add_special_tokens=True, return_tensors='pt')
    if encoded['input_ids'].shape[1] > max_length:
        truncated_ids = encoded['input_ids'][0, :max_length]
        query = tokenizer.decode(truncated_ids, skip_special_tokens=True)
        split_point = query.rfind(' ')
        if split_point != -1:
            query = query[:split_point].strip()
    return query

# Similarity search


def similarity_search(
    query: str,
    text_tuples: List[Tuple[str, List[str]]],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    use_mean_pooling: bool = True,
    top_k: int = 5,
    threshold: float = 0.5,
    debug: bool = False
) -> List[SearchResult]:
    """
    Performs similarity search using transformer embeddings with tags.
    """
    if not query or not text_tuples:
        if debug:
            logger.warning("Empty query or texts provided.")
        return []

    query_embedding: torch.Tensor = get_embeddings(
        [(query, [])], model, tokenizer, use_mean_pooling)
    text_embeddings: torch.Tensor = get_embeddings(
        text_tuples, model, tokenizer, use_mean_pooling)

    if query_embedding.numel() == 0 or text_embeddings.numel() == 0:
        if debug:
            logger.warning("No valid embeddings generated.")
        return []

    similarities: torch.Tensor = cosine_similarity(
        query_embedding, text_embeddings)
    similarities_np = similarities.cpu().numpy()

    top_k_indices = similarities_np.argsort()[-top_k:][::-1]
    top_k_scores = similarities_np[top_k_indices]

    if debug:
        logger.gray(f"[DEBUG] Top {top_k} Similarity Search Results:")
        for i, (idx, score) in enumerate(zip(top_k_indices[:top_k], top_k_scores[:top_k]), 1):
            text, tags = text_tuples[idx]
            logger.log(
                f"Rank {i}:",
                f"Doc: {idx}, Tokens: {len(tokenizer.encode(text, add_special_tokens=True))}",
                f"\nScore: {score:.3f}",
                f"\n{text[:100]}...",
                f"\nTags: {tags}",
                colors=["ORANGE", "DEBUG", "SUCCESS", "WHITE", "DEBUG"],
            )

    results: List[SearchResult] = [
        {
            'rank': i + 1,
            'score': float(top_k_scores[i]),
            'doc_index': int(idx),
            'text': text_tuples[idx][0],
            'tokens': len(tokenizer.encode(text_tuples[idx][0], add_special_tokens=True))
        }
        for i, idx in enumerate(top_k_indices)
        if top_k_scores[i] >= threshold
    ]

    return results


def main() -> None:
    model_path = 'sentence-transformers/all-MiniLM-L12-v2'
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_path)
    model: PreTrainedModel = AutoModel.from_pretrained(model_path)

    min_length: int = 50
    max_length: int = 150
    overlap: int = 20
    top_k: int = 3
    threshold: float = 0.7

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
    header = content.split('\n')[0]
    content = '\n'.join(content.split('\n')[1:])

    text_keywords_tuples: List[Tuple[str, List[str]]] = preprocess_text(
        content, tokenizer, header=header, min_length=min_length, max_length=max_length, overlap=overlap, debug=True
    )

    logger.info("\n=== Similarity Search with Mean Pooling ===\n")
    results_mean: List[SearchResult] = similarity_search(
        query, text_keywords_tuples, model, tokenizer, use_mean_pooling=True,
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

    logger.info("\n=== Similarity Search with CLS Token ===\n")
    results_cls: List[SearchResult] = similarity_search(
        query, text_keywords_tuples, model, tokenizer, use_mean_pooling=False,
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
