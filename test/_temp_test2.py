import json
import re
from typing import List, Optional, Set, Tuple, TypedDict
from uuid import uuid4
import nltk
import spacy
import torch
import torch.nn.functional as F
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag, WordNetLemmatizer
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoTokenizer, AutoModel
from jet.logger import logger

# Initialize NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)


class TokenBreakdown(TypedDict):
    header_tokens: int
    text_tokens: int
    tags_tokens: int
    overlapped_tokens: int


class SearchResult(TypedDict):
    rank: int
    score: float
    doc_index: int
    text: str
    header: Optional[str]
    tags: List[str]
    tokens: TokenBreakdown
    overlapped_text: Optional[str]


class TextProcessor:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: Optional[PreTrainedModel] = None,
        min_length: int = 50,
        max_length: int = 150,
        overlap: int = 20,
        debug: bool = False
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.min_length = min_length
        self.max_length = max_length
        self.overlap = max(0, min(overlap, max_length - 1))
        self.debug = debug
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.nlp = spacy.load('en_core_web_sm')

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'\n+', '\n', text)
        return text

    def truncate_header(self, header: str) -> str:
        header_tokens = self.tokenizer(
            header, add_special_tokens=True, return_tensors='pt')['input_ids'][0]
        if len(header_tokens) > self.max_length // 2:
            header = self.tokenizer.decode(
                header_tokens[:self.max_length // 2], skip_special_tokens=True)
            if self.debug:
                logger.debug(f"Truncated header to {len(header)} characters")
        return header

    def get_tokens(self, text: str) -> List[int]:
        try:
            return self.tokenizer(text, add_special_tokens=True, return_tensors='pt')['input_ids'][0].tolist()
        except Exception as e:
            if self.debug:
                logger.error(f"Tokenization error: {e}")
            return []

    def decode_tokens(self, tokens: List[int]) -> str:
        try:
            text = self.tokenizer.decode(
                tokens, skip_special_tokens=True).strip()
            return re.sub(r'^\W+|\W+$', '', text)
        except Exception as e:
            if self.debug:
                logger.error(f"Decoding error: {e}")
            return ""

    def generate_tags(self, texts: List[str]) -> List[List[str]]:
        all_tags: List[Set[str]] = []
        for text in texts:
            tags = set()
            tokens = word_tokenize(text.lower())
            pos_tags = pos_tag(tokens)
            for word, pos in pos_tags:
                if pos.startswith(('NN', 'VB', 'JJ')) and word not in self.stop_words and len(word) > 2:
                    lemma = self.lemmatizer.lemmatize(
                        word, pos='v' if pos.startswith('VB') else 'n')
                    tags.add(lemma)
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ('PERSON', 'ORG', 'GPE', 'PRODUCT'):
                    tags.add(ent.text.lower())
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.lower().strip()
                if (len(chunk_text.split()) <= 2 and
                    len(chunk_text) > 2 and
                    not any(w in self.stop_words for w in chunk_text.split()) and
                        chunk_text not in self.stop_words):
                    tags.add(chunk_text)
            all_tags.append(tags)

        tag_counts = Counter(tag for tags in all_tags for tag in tags)
        candidate_tags = {tag for tag in tag_counts if tag_counts[tag] >= 2}

        filtered_tags = []
        if self.model and candidate_tags:
            for i, (text, tags) in enumerate(zip(texts, all_tags)):
                valid_tags = [tag for tag in tags if tag in candidate_tags]
                if not valid_tags:
                    filtered_tags.append([])
                    continue
                text_embedding = self.get_embeddings([(text, [])])
                tag_embeddings = self.get_embeddings(
                    [(tag, []) for tag in valid_tags])
                similarities = F.cosine_similarity(
                    text_embedding, tag_embeddings).cpu().numpy()
                tag_scores = sorted(
                    zip(valid_tags, similarities), key=lambda x: x[1], reverse=True)[:8]
                sorted_tags = [tag for tag, _ in tag_scores]
                if self.debug:
                    logger.debug(
                        f"Segment {i+1} tag scores: {[(tag, float(score)) for tag, score in tag_scores]}")
                filtered_tags.append(sorted_tags)
        else:
            for tags in all_tags:
                sorted_tags = sorted(
                    [tag for tag in tags if tag in candidate_tags])[:8]
                filtered_tags.append(sorted_tags)

        return filtered_tags

    def get_sentence_aligned_overlap(self, sentences: List[str], start_index: int, prev_segment_text: str) -> Tuple[str, List[int]]:
        """Get overlap text from the last n tokens of the previous segment, ensuring whole tokens and sentences."""
        if self.overlap == 0 or not prev_segment_text:
            if self.debug:
                logger.debug("No overlap: overlap=0 or no previous segment")
            return "", []

        # Get the last n tokens from the previous segment
        prev_tokens = self.get_tokens(prev_segment_text)
        if len(prev_tokens) <= self.overlap:
            overlap_tokens = prev_tokens
            overlap_text = self.decode_tokens(overlap_tokens)
        else:
            overlap_tokens = prev_tokens[-self.overlap:]
            overlap_text = self.decode_tokens(overlap_tokens)

            # Check if the decoded text starts with a partial token
            if overlap_text and overlap_text[0] != ' ':
                first_space = overlap_text.find(' ')
                if first_space != -1:
                    overlap_text = overlap_text[first_space:].strip()
                    overlap_tokens = self.get_tokens(overlap_text)
                else:
                    overlap_text = ""
                    overlap_tokens = []

        if not overlap_text:
            if self.debug:
                logger.debug(
                    "No valid overlap text after partial token removal")
            return "", []

        # Ensure overlap text consists of whole sentences
        overlap_sentences = sent_tokenize(overlap_text)
        if not overlap_sentences:
            if self.debug:
                logger.debug("No valid sentences in overlap text")
            return "", []

        # Rebuild overlap text with whole sentences, respecting token limit
        overlap_text = ""
        overlap_tokens = []
        current_token_count = 0

        for sentence in overlap_sentences:
            temp_text = (overlap_text + " " +
                         sentence).strip() if overlap_text else sentence
            temp_tokens = self.get_tokens(temp_text)
            if len(temp_tokens) <= self.overlap:
                overlap_text = temp_text
                overlap_tokens = temp_tokens
                current_token_count = len(temp_tokens)
            else:
                break

        if not overlap_text:
            if self.debug:
                logger.debug("No valid overlap sentences within token limit")
            return "", []

        if self.debug:
            logger.debug(
                f"Overlap text ({current_token_count} tokens): {overlap_text[:50]}...")
        return overlap_text, overlap_tokens

    def preprocess_text(self, content: str, header: Optional[str] = None) -> List[Tuple[str, List[str], Optional[str]]]:
        if not isinstance(content, str) or not content.strip():
            if self.debug:
                logger.warning("Empty or invalid content provided.")
            return []

        if self.debug:
            logger.debug(f"Input content length: {len(content)} characters")

        content = self.clean_text(content)
        if header:
            header = self.clean_text(header)
            header = self.truncate_header(header)

        sentences = sent_tokenize(content)
        processed_segments: List[Tuple[str, List[str], Optional[str]]] = []
        current_sentences: List[str] = []
        sentence_index = 0
        prev_segment_text = ""

        while sentence_index < len(sentences):
            sent = sentences[sentence_index].strip()
            if len(sent) < self.min_length and not current_sentences:
                if self.debug:
                    logger.debug(
                        f"Discarded sentence (too short): {sent[:50]}...")
                sentence_index += 1
                continue

            current_sentences.append(sent)
            segment_with_header = f"{header}\n{' '.join(current_sentences)}".strip(
            ) if header else ' '.join(current_sentences)
            current_tokens = self.get_tokens(segment_with_header)

            if len(current_tokens) <= self.max_length:
                if len(current_tokens) >= self.min_length:
                    # For the first segment, no overlap; for others, use last overlap tokens from prev_segment_text
                    overlap_text, overlap_tokens = ("", []) if not prev_segment_text else self.get_sentence_aligned_overlap(
                        sentences, 0, prev_segment_text=prev_segment_text)
                    processed_segments.append(
                        (segment_with_header, [], overlap_text if overlap_text else None))
                    if self.debug:
                        logger.debug(
                            f"Adding segment: {segment_with_header[:50]}...")
                        logger.debug(
                            f"Previous segment text: {prev_segment_text[:50] if prev_segment_text else 'None'}...")
                        logger.debug(
                            f"Computed overlap: {overlap_text[:50] if overlap_text else 'None'}...")
                    prev_segment_text = segment_with_header
                    sentence_index += 1
                else:
                    sentence_index += 1
            else:
                # Segment is too long, finalize the current segment
                if len(current_sentences) > 1:
                    # Save the segment without the last sentence if it fits
                    current_sentences.pop()
                    segment_with_header = f"{header}\n{' '.join(current_sentences)}".strip(
                    ) if header else ' '.join(current_sentences)
                    segment_tokens = self.get_tokens(segment_with_header)
                    if len(segment_tokens) >= self.min_length:
                        # Use overlap from prev_segment_text (if any)
                        overlap_text, overlap_tokens = ("", []) if not prev_segment_text else self.get_sentence_aligned_overlap(
                            sentences, 0, prev_segment_text=prev_segment_text)
                        processed_segments.append(
                            (segment_with_header, [], overlap_text if overlap_text else None))
                        if self.debug:
                            logger.debug(
                                f"Adding segment: {segment_with_header[:50]}...")
                            logger.debug(
                                f"Previous segment text: {prev_segment_text[:50] if prev_segment_text else 'None'}...")
                            logger.debug(
                                f"Computed overlap: {overlap_text[:50] if overlap_text else 'None'}...")
                        prev_segment_text = segment_with_header
                    # Start new segment, compute overlap from the just-added segment
                    overlap_text, overlap_tokens = self.get_sentence_aligned_overlap(
                        sentences, 0, prev_segment_text=prev_segment_text)
                    overlap_sentences = sent_tokenize(
                        overlap_text) if overlap_text else []
                    current_sentences = overlap_sentences + \
                        [sent] if overlap_text else [sent]
                    sentence_index += 1
                else:
                    # Single sentence too long, split by words
                    words = sent.split()
                    partial_segment = ""
                    for word in words:
                        temp_partial = (partial_segment + " " +
                                        word).strip() if partial_segment else word
                        temp_with_header = f"{header}\n{temp_partial}".strip(
                        ) if header else temp_partial
                        temp_tokens = self.get_tokens(temp_with_header)
                        if len(temp_tokens) <= self.max_length:
                            partial_segment = temp_partial
                        else:
                            if partial_segment and len(self.get_tokens(f"{header}\n{partial_segment}".strip() if header else partial_segment)) >= self.min_length:
                                partial_with_header = f"{header}\n{partial_segment}".strip(
                                ) if header else partial_segment
                                overlap_text, overlap_tokens = ("", []) if not prev_segment_text else self.get_sentence_aligned_overlap(
                                    sentences, 0, prev_segment_text=prev_segment_text)
                                processed_segments.append(
                                    (partial_with_header, [], overlap_text if overlap_text else None))
                                if self.debug:
                                    logger.debug(
                                        f"Adding partial segment: {partial_with_header[:50]}...")
                                    logger.debug(
                                        f"Previous segment text: {prev_segment_text[:50] if prev_segment_text else 'None'}...")
                                    logger.debug(
                                        f"Computed overlap: {overlap_text[:50] if overlap_text else 'None'}...")
                                prev_segment_text = partial_with_header
                            partial_segment = ""
                            break
                    sentence_index += 1
                    current_sentences = []

        # Save remaining sentences if valid
        if current_sentences:
            segment_with_header = f"{header}\n{' '.join(current_sentences)}".strip(
            ) if header else ' '.join(current_sentences)
            if len(self.get_tokens(segment_with_header)) >= self.min_length:
                overlap_text, overlap_tokens = ("", []) if not prev_segment_text else self.get_sentence_aligned_overlap(
                    sentences, 0, prev_segment_text=prev_segment_text)
                processed_segments.append(
                    (segment_with_header, [], overlap_text if overlap_text else None))
                if self.debug:
                    logger.debug(
                        f"Adding segment: {segment_with_header[:50]}...")
                    logger.debug(
                        f"Previous segment text: {prev_segment_text[:50] if prev_segment_text else 'None'}...")
                    logger.debug(
                        f"Computed overlap: {overlap_text[:50] if overlap_text else 'None'}...")
                prev_segment_text = segment_with_header

        # Generate tags
        segment_texts = [segment for segment, _, _ in processed_segments]
        if segment_texts:
            tags_list = self.generate_tags(segment_texts)
            processed_segments = [(segment, tags, overlap_text) for (
                segment, _, overlap_text), tags in zip(processed_segments, tags_list)]

        # Deduplicate segments based on similarity
        unique_segments: List[Tuple[str, List[str], Optional[str]]] = []
        seen: Set[str] = set()
        for i, (segment, tags, overlap_text) in enumerate(processed_segments):
            segment_content = '\n'.join(segment.split(
                '\n')[1:]).strip() if '\n' in segment else segment
            if segment_content in seen:
                continue
            is_unique = True
            for j, (unique_segment, _, _) in enumerate(unique_segments):
                unique_content = '\n'.join(unique_segment.split('\n')[1:]).strip(
                ) if '\n' in unique_segment else unique_segment
                similarity = self.compute_text_similarity(
                    segment_content, unique_content)
                if similarity > 0.95:
                    is_unique = False
                    if self.debug:
                        logger.debug(
                            f"Discarded near-duplicate segment {i+1} (similarity {similarity:.3f}): {segment[:50]}...")
                    break
            if is_unique:
                unique_segments.append((segment, tags, overlap_text))
                seen.add(segment_content)

        if self.debug:
            logger.debug(
                "\nFinal Preprocessed Segments with Tags and Overlap:")
            for i, (segment, tags, overlap_text) in enumerate(unique_segments):
                combined_text = f"{segment} {' '.join(tags)}"
                encoded = self.tokenizer(
                    combined_text, add_special_tokens=True, return_tensors='pt')
                token_count = encoded['input_ids'].shape[1]
                logger.debug(
                    f"{i+1}. ({len(segment)} chars, {token_count} tokens) {segment}")
                logger.debug(f"   Tags: {tags}")
                logger.debug(f"   Overlap Text: {overlap_text or 'None'}")

        if not unique_segments and self.debug:
            logger.warning("No valid segments produced after preprocessing.")

        return unique_segments

    def preprocess_query(self, query: str, max_length: int = 300) -> str:
        if not isinstance(query, str) or not query.strip():
            return ""
        query = re.sub(r'\[.*?\]', '', query)
        query = re.sub(r'\s+', ' ', query.strip())
        query = re.sub(r'[^\w\s.,!?]', '', query)
        encoded = self.tokenizer(
            query, add_special_tokens=True, return_tensors='pt')
        if encoded['input_ids'].shape[1] > max_length:
            truncated_ids = encoded['input_ids'][0, :max_length]
            query = self.tokenizer.decode(
                truncated_ids, skip_special_tokens=True)
            split_point = query.rfind(' ')
            if split_point != -1:
                query = query[:split_point].strip()
        return query

    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts using embeddings."""
        if not self.model:
            return 0.0
        embeddings = self.get_embeddings(
            [(text1, []), (text2, [])], use_mean_pooling=True)
        if embeddings.shape[0] < 2:
            return 0.0
        similarity = F.cosine_similarity(
            embeddings[0:1], embeddings[1:2]).item()
        return similarity

    def get_embeddings(self, texts: List[Tuple[str, List[str]]], batch_size: int = 32, use_mean_pooling: bool = True) -> torch.Tensor:
        if not self.model:
            raise ValueError("Model is required for embeddings computation")
        combined_texts = [f"{text} {' '.join(tags)}" for text, tags in texts]
        embeddings: List[torch.Tensor] = []
        for i in range(0, len(combined_texts), batch_size):
            batch_texts = combined_texts[i:i + batch_size]
            try:
                encoded_input = self.tokenizer(
                    batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512
                )
                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                if use_mean_pooling:
                    token_embeddings = model_output[0]
                    input_mask_expanded = encoded_input['attention_mask'].unsqueeze(
                        -1).expand(token_embeddings.size()).float()
                    batch_embeddings = torch.sum(
                        token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                else:
                    batch_embeddings = model_output[0][:, 0, :]  # CLS token
                embeddings.append(batch_embeddings)
            except Exception as e:
                logger.error(
                    f"Embedding error in batch {i//batch_size + 1}: {e}")
        return torch.cat(embeddings, dim=0) if embeddings else torch.tensor([])


class SimilaritySearch:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = TextProcessor(tokenizer, model)

    def compute_text_similarity(self, text1: str, text2: str, use_mean_pooling: bool = True) -> float:
        """Compute cosine similarity between two texts using embeddings."""
        embeddings = self.processor.get_embeddings(
            [(text1, []), (text2, [])], use_mean_pooling=use_mean_pooling)
        if embeddings.shape[0] < 2:
            return 0.0
        similarity = F.cosine_similarity(
            embeddings[0:1], embeddings[1:2]).item()
        return similarity

    def search(
        self,
        query: str,
        text_tuples: List[Tuple[str, List[str], Optional[str]]],
        use_mean_pooling: bool = True,
        top_k: Optional[int] = 5,
        threshold: float = 0.5,
        debug: bool = False
    ) -> List[SearchResult]:
        if not query or not text_tuples:
            if debug:
                logger.warning("Empty query or texts provided.")
            return []

        query_embedding = self.processor.get_embeddings(
            [(query, [])], use_mean_pooling=use_mean_pooling)
        text_embeddings = self.processor.get_embeddings(
            [(text, tags) for text, tags, _ in text_tuples], use_mean_pooling=use_mean_pooling)

        if query_embedding.numel() == 0 or text_embeddings.numel() == 0:
            if debug:
                logger.warning("No valid embeddings generated.")
            return []

        similarities = F.cosine_similarity(query_embedding, text_embeddings)
        similarities_np = similarities.cpu().numpy()

        if not top_k:
            top_k = len(text_tuples)

        top_k_indices = similarities_np.argsort()[-top_k:][::-1]
        top_k_scores = similarities_np[top_k_indices]

        if debug:
            logger.gray(f"[DEBUG] Top {top_k} Similarity Search Results:")
            # Create a list of tuples with rank, index, and score
            ranked_results = list(
                zip(range(1, top_k + 1), top_k_indices[:top_k], top_k_scores[:top_k]))
            # Sort by doc index while preserving rank
            ranked_results.sort(key=lambda x: x[1])

            for rank, idx, score in ranked_results:
                text, tags, overlapped_text = text_tuples[idx]
                header = text.split('\n')[0] if '\n' in text else None
                content = '\n'.join(text.split(
                    '\n')[1:]).strip() if header else text
                header_tokens = len(self.tokenizer.encode(
                    header, add_special_tokens=True)) if header else 0
                content_tokens = len(self.tokenizer.encode(
                    content, add_special_tokens=True))
                tags_tokens = len(self.tokenizer.encode(
                    ' '.join(tags), add_special_tokens=True)) if tags else 0
                overlapped_tokens = len(self.tokenizer.encode(
                    overlapped_text, add_special_tokens=True)) if overlapped_text else 0
                token_breakdown = {
                    'header_tokens': header_tokens,
                    'text_tokens': content_tokens,
                    'tags_tokens': tags_tokens,
                    'overlapped_tokens': overlapped_tokens
                }
                logger.log(
                    f"Rank {rank}:",
                    f"Doc: {idx}, Tokens: {sum(token_breakdown.values())} (Header: {header_tokens}, Text: {content_tokens}, Tags: {tags_tokens}, Overlap: {overlapped_tokens})",
                    f"\nScore: {score:.3f}",
                    f"\nHeader: {header or 'None'}",
                    f"\nText: {content}",
                    f"\nTags: {tags}",
                    f"\nOverlapped Text: {overlapped_text or 'None'}",
                    colors=["ORANGE", "DEBUG", "SUCCESS",
                            "WHITE", "WHITE", "DEBUG", "WHITE"],
                )

        results: List[SearchResult] = []
        seen_texts: Set[str] = set()
        for i, idx in enumerate(top_k_indices):
            if top_k_scores[i] < threshold:
                continue
            text, tags, overlapped_text = text_tuples[idx]
            header = text.split('\n')[0] if '\n' in text else None
            content = '\n'.join(text.split(
                '\n')[1:]).strip() if header else text
            # Check for near-duplicate results
            is_unique = True
            for seen_text in seen_texts:
                similarity = self.compute_text_similarity(
                    content, seen_text, use_mean_pooling=use_mean_pooling)
                if similarity > 0.95:
                    is_unique = False
                    if debug:
                        logger.debug(
                            f"Discarded near-duplicate result (Doc {idx}, similarity {similarity:.3f}): {content[:50]}...")
                    break
            if not is_unique:
                continue
            header_tokens = len(self.tokenizer.encode(
                header, add_special_tokens=True)) if header else 0
            content_tokens = len(self.tokenizer.encode(
                content, add_special_tokens=True))
            tags_tokens = len(self.tokenizer.encode(
                ' '.join(tags), add_special_tokens=True)) if tags else 0
            overlapped_tokens = len(self.tokenizer.encode(
                overlapped_text, add_special_tokens=True)) if overlapped_text else 0
            token_breakdown = {
                'header_tokens': header_tokens,
                'text_tokens': content_tokens,
                'tags_tokens': tags_tokens,
                'overlapped_tokens': overlapped_tokens
            }
            results.append({
                'rank': i + 1,
                'score': float(top_k_scores[i]),
                'doc_index': int(idx),
                'text': content,
                'header': header,
                'tags': tags,
                'tokens': token_breakdown,
                'overlapped_text': overlapped_text
            })
            seen_texts.add(content)

        return results


def main() -> None:
    model_path = 'sentence-transformers/all-MiniLM-L12-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    search = SimilaritySearch(model, tokenizer)
    processor = search.processor

    sample_text = "Movie, 2012 Finished 1 ep, 109 min Action Adventure Fantasy Naruto: Shippuuden Movie 6 - Road to Ninja Returning home to Konohagakure, the young ninja celebrate defeating a group of supposed Akatsuki members."
    encoded_text = tokenizer.encode(sample_text, add_special_tokens=False)
    last_20_tokens = encoded_text[-20:]
    decoded_last_20 = tokenizer.decode(last_20_tokens)

    # Check if the decoded text starts with a partial token
    if decoded_last_20 and decoded_last_20[0] != ' ':
        # Find the first space in the decoded text
        first_space = decoded_last_20.find(' ')
        if first_space != -1:
            # Remove the partial token at the start
            decoded_last_20 = decoded_last_20[first_space:].strip()

    logger.orange(
        f"[DEBUG] Sample text tokens count: {len(encoded_text)}\nDecoded last 20: {decoded_last_20}")

    min_length = 50
    max_length = 150
    overlap = 20
    top_k = None
    threshold = 0.2
    max_result_tokens = 300

    query = 'List upcoming isekai anime this year (2024-2025).'
    query = processor.preprocess_query(query, max_length=max_length)

    print(f"Query: {query}")
    print(f"Model: {model_path}")
    print(f"Min Length: {min_length}")
    print(f"Max Length: {max_length} tokens")
    print(f"Overlap: {overlap}")
    print(f"Top K: {top_k}")
    print(f"Threshold: {threshold}")
    print(f"Max Result Tokens: {max_result_tokens}")
    print()

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
    header = content.split('\n')[0]
    content = '\n'.join(content.split('\n')[1:])

    text_keywords_tuples = processor.preprocess_text(content, header=header)

    logger.info("\n=== Similarity Search with Mean Pooling ===\n")
    results_mean = search.search(
        query, text_keywords_tuples, use_mean_pooling=True, top_k=top_k, threshold=threshold, debug=True)

    if results_mean:
        # Post-process results_mean: sort by doc_index
        results_mean.sort(key=lambda x: x['doc_index'])
        mean_result_text = ""
        current_tokens = 0
        for result in results_mean:
            # Tokenize the text to be added
            text_to_add = f"{result['text']}\n"
            tokens_to_add = result['tokens']['text_tokens']
            # Check if adding this text would exceed max_result_tokens
            if current_tokens + tokens_to_add <= max_result_tokens:
                mean_result_text += text_to_add
                current_tokens += tokens_to_add
            else:
                logger.warning(
                    f"Stopped adding results for Mean Pooling at {current_tokens} tokens to respect max_result_tokens={max_result_tokens}."
                )
                break
        mean_result_text = mean_result_text.strip()
        if mean_result_text:
            logger.info(
                "\n=== Mean Pooling Search Results (Sorted by Doc Index) ===\n")
            logger.success(mean_result_text, colors=["WHITE"])
            logger.teal(f"Total tokens in results (mean): {current_tokens}")
        else:
            logger.info(
                "\n=== Mean Pooling Search Results (Sorted by Doc Index) ===\n")
            logger.warning(
                "No results could be included within max_result_tokens for Mean Pooling search.")
    else:
        logger.info(
            "\n=== Mean Pooling Search Results (Sorted by Doc Index) ===\n")
        logger.warning(
            "No results passed the threshold for Mean Pooling search.")

    logger.info("\n=== Similarity Search with CLS Token ===\n")
    results_cls = search.search(
        query, text_keywords_tuples, use_mean_pooling=False, top_k=top_k, threshold=threshold, debug=True)

    if results_cls:
        # Post-process results_cls: sort by doc_index
        results_cls.sort(key=lambda x: x['doc_index'])
        cls_result_text = ""
        current_tokens = 0
        for result in results_cls:
            # Tokenize the text to be added
            text_to_add = f"{result['text']}\n"
            tokens_to_add = result['tokens']['text_tokens']
            # Check if adding this text would exceed max_result_tokens
            if current_tokens + tokens_to_add <= max_result_tokens:
                cls_result_text += text_to_add
                current_tokens += tokens_to_add
            else:
                logger.warning(
                    f"Stopped adding results for CLS Token at {current_tokens} tokens to respect max_result_tokens={max_result_tokens}."
                )
                break
        cls_result_text = cls_result_text.strip()
        if cls_result_text:
            logger.info(
                "\n=== CLS Token Search Results (Sorted by Doc Index) ===\n")
            logger.success(cls_result_text, colors=["WHITE"])
            logger.teal(f"Total tokens in results (cls): {current_tokens}")
        else:
            logger.info(
                "\n=== CLS Token Search Results (Sorted by Doc Index) ===\n")
            logger.warning(
                "No results could be included within max_result_tokens for CLS Token search.")
    else:
        logger.info(
            "\n=== CLS Token Search Results (Sorted by Doc Index) ===\n")
        logger.warning(
            "No results passed the threshold for CLS Token search.")


if __name__ == "__main__":
    main()
