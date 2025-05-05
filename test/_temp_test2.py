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


class SearchResult(TypedDict):
    rank: int
    score: float
    doc_index: int
    text: str
    tokens: int


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

    def get_sentence_aligned_overlap(self, sentences: List[str], start_index: int) -> Tuple[str, List[int]]:
        """Get overlap text starting from the first complete sentence within the overlap range."""
        if self.overlap == 0 or start_index >= len(sentences):
            if self.debug:
                logger.debug("No overlap: overlap=0 or no sentences")
            return "", []

        overlap_text = ""
        for i in range(start_index, min(start_index + self.overlap, len(sentences))):
            if len(overlap_text) + len(sentences[i]) <= self.max_length:
                overlap_text += sentences[i] + " "
            else:
                break

        overlap_text = overlap_text.strip()
        if not overlap_text:
            if self.debug:
                logger.debug("No valid overlap sentence found")
            return "", []

        overlap_tokens = self.get_tokens(overlap_text)
        if len(overlap_tokens) > self.max_length:
            # Truncate to max_length
            overlap_text = self.decode_tokens(overlap_tokens[:self.max_length])
            overlap_tokens = self.get_tokens(overlap_text)

        if self.debug:
            logger.debug(f"Overlap text: {overlap_text[:50]}...")
        return overlap_text, overlap_tokens

    def preprocess_text(self, content: str, header: Optional[str] = None) -> List[Tuple[str, List[str]]]:
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
        processed_segments: List[Tuple[str, List[str]]] = []
        current_sentences: List[str] = []
        sentence_index = 0

        while sentence_index < len(sentences):
            sent = sentences[sentence_index].strip()
            if len(sent) < self.min_length and len(current_sentences) == 0:
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
                    processed_segments.append((segment_with_header, []))
                    if self.debug:
                        logger.debug(
                            f"Added segment: {segment_with_header[:50]}...")
                sentence_index += 1
            else:
                # Segment is too long, try to split
                if len(current_sentences) > 1:
                    # Save the current segment without the last sentence
                    current_sentences.pop()
                    segment_with_header = f"{header}\n{' '.join(current_sentences)}".strip(
                    ) if header else ' '.join(current_sentences)
                    if len(self.get_tokens(segment_with_header)) >= self.min_length:
                        processed_segments.append((segment_with_header, []))
                        if self.debug:
                            logger.debug(
                                f"Added segment: {segment_with_header[:50]}...")
                    # Prepare overlap
                    overlap_text, overlap_tokens = self.get_sentence_aligned_overlap(
                        sentences, sentence_index - len(current_sentences))
                    current_sentences = sent_tokenize(
                        overlap_text) if overlap_text else []
                else:
                    # Single sentence too long
                    words = sent.split()
                    partial_segment = ""
                    partial_sentences: List[str] = []
                    for word in words:
                        temp_partial = (partial_segment + " " +
                                        word).strip() if partial_segment else word
                        temp_with_header = f"{header}\n{temp_partial}".strip(
                        ) if header else temp_partial
                        temp_tokens = self.get_tokens(temp_with_header)
                        if len(temp_tokens) <= self.max_length:
                            partial_segment = temp_partial
                            if word.endswith(('.', '!', '?')):
                                partial_sentences.append(partial_segment)
                                partial_segment = ""
                        else:
                            if partial_segment and len(self.get_tokens(f"{header}\n{partial_segment}".strip() if header else partial_segment)) >= self.min_length:
                                partial_with_header = f"{header}\n{partial_segment}".strip(
                                ) if header else partial_segment
                                processed_segments.append(
                                    (partial_with_header, []))
                                if self.debug:
                                    logger.debug(
                                        f"Added partial segment: {partial_with_header[:50]}...")
                            partial_segment = ""
                            partial_sentences = []
                    if partial_segment and len(self.get_tokens(f"{header}\n{partial_segment}".strip() if header else partial_segment)) >= self.min_length:
                        partial_with_header = f"{header}\n{partial_segment}".strip(
                        ) if header else partial_segment
                        processed_segments.append((partial_with_header, []))
                        if self.debug:
                            logger.debug(
                                f"Added partial segment: {partial_with_header[:50]}...")
                    sentence_index += 1
                    current_sentences = []

        # Save remaining sentences if valid
        if current_sentences:
            segment_with_header = f"{header}\n{' '.join(current_sentences)}".strip(
            ) if header else ' '.join(current_sentences)
            if len(self.get_tokens(segment_with_header)) >= self.min_length:
                processed_segments.append((segment_with_header, []))
                if self.debug:
                    logger.debug(
                        f"Added segment: {segment_with_header[:50]}...")

        # Generate tags
        segment_texts = [segment for segment, _ in processed_segments]
        if segment_texts:
            tags_list = self.generate_tags(segment_texts)
            processed_segments = [(segment, tags) for (
                segment, _), tags in zip(processed_segments, tags_list)]

        # Deduplicate and compute token counts including tags
        seen: Set[str] = set()
        unique_segments: List[Tuple[str, List[str]]] = []
        for segment, tags in processed_segments:
            if segment not in seen:
                unique_segments.append((segment, tags))
                seen.add(segment)

        if self.debug:
            logger.debug("\nFinal Preprocessed Segments with Tags:")
            for i, (segment, tags) in enumerate(unique_segments):
                combined_text = f"{segment} {' '.join(tags)}"
                encoded = self.tokenizer(
                    combined_text, add_special_tokens=True, return_tensors='pt')
                token_count = encoded['input_ids'].shape[1]
                logger.debug(
                    f"{i+1}. ({len(segment)} chars, {token_count} tokens) {segment}")
                logger.debug(f"   Tags: {tags}")

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

    def search(
        self,
        query: str,
        text_tuples: List[Tuple[str, List[str]]],
        use_mean_pooling: bool = True,
        top_k: int = 5,
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
            text_tuples, use_mean_pooling=use_mean_pooling)

        if query_embedding.numel() == 0 or text_embeddings.numel() == 0:
            if debug:
                logger.warning("No valid embeddings generated.")
            return []

        similarities = F.cosine_similarity(query_embedding, text_embeddings)
        similarities_np = similarities.cpu().numpy()

        top_k_indices = similarities_np.argsort()[-top_k:][::-1]
        top_k_scores = similarities_np[top_k_indices]

        if debug:
            logger.gray(f"[DEBUG] Top {top_k} Similarity Search Results:")
            for i, (idx, score) in enumerate(zip(top_k_indices[:top_k], top_k_scores[:top_k]), 1):
                text, tags = text_tuples[idx]
                combined_text = f"{text} {' '.join(tags)}"
                token_count = len(self.tokenizer.encode(
                    combined_text, add_special_tokens=True))
                logger.log(
                    f"Rank {i}:",
                    f"Doc: {idx}, Tokens: {token_count}",
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
                'tokens': len(self.tokenizer.encode(f"{text_tuples[idx][0]} {' '.join(text_tuples[idx][1])}", add_special_tokens=True))
            }
            for i, idx in enumerate(top_k_indices)
            if top_k_scores[i] >= threshold
        ]

        return results


def main() -> None:
    model_path = 'sentence-transformers/all-MiniLM-L12-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    search = SimilaritySearch(model, tokenizer)
    processor = search.processor

    min_length = 50
    max_length = 150
    overlap = 20
    top_k = 3
    threshold = 0.7

    query = 'List upcoming isekai anime this year (2024-2025).'
    query = processor.preprocess_query(query, max_length=max_length)

    print(f"Query: {query}")
    print(f"Model: {model_path}")
    print(f"Min Length: {min_length}")
    print(f"Max Length: {max_length} tokens")
    print(f"Overlap: {overlap}")
    print(f"Top K: {top_k}")
    print(f"Threshold: {threshold}")
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
        for result in results_mean:
            mean_result_text += f"Document {result['doc_index'] + 1}\n"
            mean_result_text += f"Rank: {result['rank']}\n"
            mean_result_text += f"Score: {result['score']:.3f}\n"
            mean_result_text += f"Tokens: {result['tokens']}\n"
            mean_result_text += f"Text: {json.dumps(result['text'])[:200]}...\n"
            mean_result_text += "\n"
        logger.info(
            "\n=== Mean Pooling Search Results (Sorted by Doc Index) ===\n")
        logger.log(mean_result_text, colors=["WHITE"])
    else:
        logger.info(
            "\n=== Mean Pooling Search Results (Sorted by Doc Index) ===\n")
        logger.warning(
            f"No results passed the threshold ({threshold}) for Mean Pooling search.")

    logger.info("\n=== Similarity Search with CLS Token ===\n")
    results_cls = search.search(query, text_keywords_tuples,
                                use_mean_pooling=False, top_k=top_k, threshold=threshold, debug=True)

    if results_cls:
        # Post-process results_cls: sort by doc_index
        results_cls.sort(key=lambda x: x['doc_index'])
        cls_result_text = ""
        for result in results_cls:
            cls_result_text += f"Document {result['doc_index'] + 1}\n"
            cls_result_text += f"Rank: {result['rank']}\n"
            cls_result_text += f"Score: {result['score']:.3f}\n"
            cls_result_text += f"Tokens: {result['tokens']}\n"
            cls_result_text += f"Text: {json.dumps(result['text'])[:200]}...\n"
            cls_result_text += "\n"
        logger.info(
            "\n=== CLS Token Search Results (Sorted by Doc Index) ===\n")
        logger.log(cls_result_text, colors=["WHITE"])
    else:
        logger.info(
            "\n=== CLS Token Search Results (Sorted by Doc Index) ===\n")
        logger.warning(
            f"No results passed the threshold ({threshold}) for CLS Token search.")


if __name__ == "__main__":
    main()
