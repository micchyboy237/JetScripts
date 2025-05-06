import json
import re
from typing import List, Optional, Tuple, TypedDict
from uuid import uuid4
import nltk
import spacy
import torch
import torch.nn.functional as F
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


class SearchResult(TypedDict):
    rank: int
    score: float
    doc_index: int
    text: str
    header: Optional[str]
    tags: List[str]
    tokens: TokenBreakdown


class TextProcessor:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: Optional[PreTrainedModel] = None,
        min_length: int = 50,
        max_length: int = 150,
        debug: bool = False
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.min_length = min_length
        self.max_length = max_length
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

    def generate_tags(self, texts: List[str]) -> List[List[str]]:
        all_tags = []
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
            all_tags.append(sorted(tags)[:8])
        return all_tags

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
        current_segment = ""
        current_sentences: List[str] = []

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            # Try adding the sentence to the current segment
            temp_sentences = current_sentences + [sent]
            temp_segment = f"{header}\n{' '.join(temp_sentences)}".strip(
            ) if header else ' '.join(temp_sentences)
            temp_tokens = self.get_tokens(temp_segment)

            if len(temp_tokens) <= self.max_length:
                current_sentences.append(sent)
                current_segment = temp_segment
            else:
                # If adding the sentence exceeds max_length, save the current segment if it meets min_length
                if current_segment and len(self.get_tokens(current_segment)) >= self.min_length:
                    processed_segments.append((current_segment, []))
                    if self.debug:
                        logger.debug(
                            f"Adding segment: {current_segment[:50]}...")
                # Start a new segment with the current sentence
                current_sentences = [sent]
                current_segment = f"{header}\n{sent}".strip(
                ) if header else sent

        # Save the final segment if it meets min_length
        if current_segment and len(self.get_tokens(current_segment)) >= self.min_length:
            processed_segments.append((current_segment, []))
            if self.debug:
                logger.debug(
                    f"Adding final segment: {current_segment[:50]}...")

        # Generate tags for all segments
        segment_texts = [segment for segment, _ in processed_segments]
        if segment_texts:
            tags_list = self.generate_tags(segment_texts)
            processed_segments = [(segment, tags) for (
                segment, _), tags in zip(processed_segments, tags_list)]

        if self.debug:
            logger.debug("\nFinal Preprocessed Segments with Tags:")
            for i, (segment, tags) in enumerate(processed_segments):
                combined_text = f"{segment} {' '.join(tags)}"
                encoded = self.tokenizer(
                    combined_text, add_special_tokens=True, return_tensors='pt')
                token_count = encoded['input_ids'].shape[1]
                logger.debug(
                    f"{i+1}. ({len(segment)} chars, {token_count} tokens) {segment}")
                logger.debug(f"   Tags: {tags}")

        if not processed_segments and self.debug:
            logger.warning("No valid segments produced after preprocessing.")

        return processed_segments

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
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, max_length: int = 150):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = TextProcessor(tokenizer, model, max_length=max_length)

    def search(
        self,
        query: str,
        text_tuples: List[Tuple[str, List[str]]],
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
            [(text, tags) for text, tags in text_tuples], use_mean_pooling=use_mean_pooling)

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
            ranked_results = list(
                zip(range(1, top_k + 1), top_k_indices[:top_k], top_k_scores[:top_k]))
            ranked_results.sort(key=lambda x: x[1])

            for rank, idx, score in ranked_results:
                text, tags = text_tuples[idx]
                header = text.split('\n')[0] if '\n' in text else None
                content = '\n'.join(text.split(
                    '\n')[1:]).strip() if header else text
                header_tokens = len(self.tokenizer.encode(
                    header, add_special_tokens=True)) if header else 0
                content_tokens = len(self.tokenizer.encode(
                    content, add_special_tokens=True))
                tags_tokens = len(self.tokenizer.encode(
                    ' '.join(tags), add_special_tokens=True)) if tags else 0
                token_breakdown = {
                    'header_tokens': header_tokens,
                    'text_tokens': content_tokens,
                    'tags_tokens': tags_tokens
                }
                logger.log(
                    f"Rank {rank}:",
                    f"Doc: {idx}, Tokens: {sum(token_breakdown.values())} (Header: {header_tokens}, Text: {content_tokens}, Tags: {tags_tokens})",
                    f"\nScore: {score:.3f}",
                    f"\nHeader: {header or 'None'}",
                    f"\nText: {content}",
                    f"\nTags: {tags}",
                    colors=["ORANGE", "DEBUG", "SUCCESS",
                            "WHITE", "WHITE", "DEBUG"],
                )

        results: List[SearchResult] = []
        for i, idx in enumerate(top_k_indices):
            if top_k_scores[i] < threshold:
                continue
            text, tags = text_tuples[idx]
            header = text.split('\n')[0] if '\n' in text else None
            content = '\n'.join(text.split(
                '\n')[1:]).strip() if header else text
            header_tokens = len(self.tokenizer.encode(
                header, add_special_tokens=True)) if header else 0
            content_tokens = len(self.tokenizer.encode(
                content, add_special_tokens=True))
            tags_tokens = len(self.tokenizer.encode(
                ' '.join(tags), add_special_tokens=True)) if tags else 0
            token_breakdown = {
                'header_tokens': header_tokens,
                'text_tokens': content_tokens,
                'tags_tokens': tags_tokens
            }
            results.append({
                'rank': i + 1,
                'score': float(top_k_scores[i]),
                'doc_index': int(idx),
                'text': content,
                'header': header,
                'tags': tags,
                'tokens': token_breakdown
            })

        return results


def main() -> None:
    model_path = 'sentence-transformers/all-MiniLM-L12-v2'
    min_length = 50
    max_length = 100
    top_k = None
    threshold = 0.2
    max_result_tokens = max_length * 3

    query = 'List upcoming isekai anime this year (2024-2025).'

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    search = SimilaritySearch(model, tokenizer, max_length=max_length)
    processor = search.processor

    query = processor.preprocess_query(query, max_length=max_length)

    print(f"Query: {query}")
    print(f"Model: {model_path}")
    print(f"Min Length: {min_length}")
    print(f"Max Length: {max_length} tokens")
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
        results_mean.sort(key=lambda x: x['doc_index'])
        mean_result_text = ""
        current_tokens = 0
        for result in results_mean:
            text_to_add = f"{result['text']}\n"
            tokens_to_add = result['tokens']['text_tokens']
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
        results_cls.sort(key=lambda x: x['doc_index'])
        cls_result_text = ""
        current_tokens = 0
        for result in results_cls:
            text_to_add = f"{result['text']}\n"
            tokens_to_add = result['tokens']['text_tokens']
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
