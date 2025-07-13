import argparse
from collections import defaultdict
import shutil
import string
from jet.code.html_utils import clean_html, preprocess_html
from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.code.markdown_utils._markdown_analyzer import analyze_markdown, link_to_text_ratio
from jet.code.markdown_utils._markdown_parser import derive_by_header_hierarchy, parse_markdown
from jet.code.markdown_utils._preprocessors import clean_markdown_links
from jet.code.splitter_markdown_utils import extract_markdown_links
from jet.llm.mlx.templates.generate_labels import generate_labels
from jet.llm.utils.mmr_diversity import sort_by_mmr_diversity
from jet.logger import logger
from jet.models.embeddings.base import generate_embeddings
from jet.models.model_registry.transformers.cross_encoder_model_registry import CrossEncoderRegistry
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import ModelType, EmbedModelType, LLMModelType
from jet.models.tokenizer.base import count_tokens, get_string_detokenizer_fn, get_string_tokenizer_fn, get_tokenizer_fn
from jet.utils.text_constants import TEXT_CONTRACTIONS_EN
from jet.wordnet.similarity import group_similar_texts
import asyncio
import os
from typing import DefaultDict, List, Dict, Literal, Optional, Set
from jet.data.utils import generate_unique_id
from jet.file.utils import save_file
from jet.scrapers.hrequests_utils import scrape_urls
from jet.scrapers.utils import scrape_links, search_data
from jet.wordnet.analyzers.text_analysis import analyze_readability
from jet.models.embeddings.chunking import chunk_headers_by_hierarchy
import justext
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
import json
from tqdm import tqdm
import re
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import uuid

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

PROMPT_TEMPLATE = """\
Context information is below.
---------------------
{context}
---------------------

Given the context information, answer the query.

Query: {query}
"""


def format_sub_dir(text: str) -> str:
    return text.lower().strip('.,!?').replace(' ', '_').replace('.', '_').replace(',', '_').replace('!', '_').replace('?', '_').strip()


def format_sub_url_dir(url: str) -> str:
    clean_url = re.sub(r'^(https?://|www\.)|(\?.*)', '', url)
    trans_table = str.maketrans({p: '_' for p in string.punctuation})
    formatted = clean_url.translate(trans_table).lower()
    formatted = re.sub(r'_+', '_', formatted)
    return formatted.strip('_')


def is_valid_header(header: Optional[str]) -> bool:
    if not header or not header.strip():
        return False
    generic_keywords = {'planet', 'articles', 'tutorials',
                        'jobs', 'series details', 'about us', 'conclusion'}
    date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b'
    anime_keywords = {'anime', 'isekai', 'series', 'episode', 'character'}
    if any(keyword in header.lower() for keyword in generic_keywords) or re.match(date_pattern, header):
        return False
    if any(keyword in header.lower() for keyword in anime_keywords):
        return True
    return len(word_tokenize(header)) > 3


def separate_by_headers(paragraphs: List) -> List[Dict]:
    sections = []
    current_section = {
        "header": None,
        "content": [],
        "xpath": None,
        "parent_header": None,
        "parent_level": None,
        "level": 0
    }
    header_stack = []
    header_tags = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

    for paragraph in paragraphs:
        if paragraph.is_heading:
            xpath = paragraph.xpath.lower()
            header_tag_match = re.search(r'/(h[1-6])(?:\[|$)', xpath)
            header_tag = header_tag_match.group(
                1) if header_tag_match else None
            if header_tag and header_tag in header_tags:
                if (current_section["content"] or current_section["header"]) and is_valid_header(current_section["header"]):
                    sections.append(current_section)
                level = header_tags[header_tag]
                header_stack = [h for h in header_stack if h["level"] < level]
                parent = next(
                    (h for h in header_stack[::-1] if h["level"] < level), None)
                current_section = {
                    "header": f"{'#' * level} {paragraph.text}" if paragraph.text else None,
                    "content": [],
                    "xpath": paragraph.xpath,
                    "parent_header": f"{'#' * parent['level']} {parent['content']}" if parent else None,
                    "parent_level": parent["level"] if parent else None,
                    "level": level
                }
                header_stack.append(
                    {"level": level, "content": paragraph.text})
            else:
                current_section["content"].append(paragraph.text)
        else:
            current_section["content"].append(paragraph.text)
    if (current_section["content"] or current_section["header"]) and is_valid_header(current_section["header"]):
        sections.append(current_section)
    return sections


def merge_similar_docs(embeddings: List[Dict], similarity_threshold: float = 0.8) -> tuple[List[Dict], List[Dict]]:
    texts = [f"{doc['header'] or ''}\n{doc['content']}" for doc in embeddings]
    embedding_matrix = [doc["embedding"] for doc in embeddings]
    clusters = group_similar_texts(
        texts=texts,
        threshold=similarity_threshold,
        embeddings=embedding_matrix,
    )
    merged_docs = []
    merge_info = []
    for cluster_texts in clusters:
        cluster_docs = [embeddings[i]
                        for i, text in enumerate(texts) if text in cluster_texts]
        if len(cluster_docs) > 1:
            best_doc = cluster_docs[0]
            merged_doc_id = generate_unique_id()
            merged_doc = best_doc.copy()
            merged_doc["doc_id"] = merged_doc_id
            merged_docs.append(merged_doc)
            merged_text = "\n".join([doc["content"] for doc in cluster_docs])
            merged_num_tokens = sum(doc["num_tokens"] for doc in cluster_docs)
            merge_info.append({
                "merged_doc_id": merged_doc_id,
                "original_doc_ids": [doc["doc_id"] for doc in cluster_docs],
                "original_chunk_ids": [doc["chunk_id"] for doc in cluster_docs],
                "content": merged_text,
                "num_tokens": merged_num_tokens,
                "header": best_doc["header"],
                "url": best_doc["url"],
                "score": best_doc.get("score", None),
                "mtld": best_doc["mtld"],
                "mtld_category": best_doc["mtld_category"],
                "word_count": best_doc["word_count"],
                "link_to_text_ratio": best_doc["link_to_text_ratio"],
                "parent_header": best_doc.get("parent_header", None),
                "parent_level": best_doc.get("parent_level", None),
                "level": best_doc.get("level", None),
                "chunk_index": best_doc.get("chunk_index", None),
                "doc_index": best_doc.get("doc_index", None)
            })
        else:
            doc = cluster_docs[0]
            merged_docs.append(doc)
            merge_info.append({
                "merged_doc_id": doc["doc_id"],
                "original_doc_ids": [doc["doc_id"]],
                "original_chunk_ids": [doc["chunk_id"]],
                "content": doc["content"],
                "num_tokens": doc["num_tokens"],
                "header": doc["header"],
                "url": doc["url"],
                "score": doc.get("score", None),
                "mtld": doc["mtld"],
                "mtld_category": doc["mtld_category"],
                "word_count": doc["word_count"],
                "link_to_text_ratio": doc["link_to_text_ratio"],
                "parent_header": doc.get("parent_header", None),
                "parent_level": doc.get("parent_level", None),
                "level": doc.get("level", None),
                "chunk_index": doc.get("chunk_index", None),
                "doc_index": doc.get("doc_index", None)
            })
    save_file(merge_info, f"{OUTPUT_DIR}/merged_docs.json")
    return merged_docs, merge_info


def preprocess_text(
    text: str,
    preserve_chars: Optional[Set[str]] = None,
    remove_stopwords: bool = False,
    apply_lemmatization: bool = False
) -> str:
    if not text or not text.strip():
        logger.debug(f"Empty or whitespace-only input text: '{text}'")
        return ""
    logger.debug(f"Preprocessing text: '{text}'")
    text = re.sub(r'\s+', ' ', text.strip())
    for contraction, expanded in TEXT_CONTRACTIONS_EN.items():
        text = re.sub(r'\b' + contraction + r'\b',
                      expanded, text, flags=re.IGNORECASE)
    text = text.lower()
    preserve_chars = preserve_chars or {'-', '_'}
    pattern = r'[^a-z0-9\s' + ''.join(map(re.escape, preserve_chars)) + r']'
    text = re.sub(pattern, '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    if remove_stopwords:
        logger.warning("Stopword removal not implemented in this version")
    if apply_lemmatization:
        logger.warning("Lemmatization not implemented in this version")
    logger.debug(f"Preprocessed text: '{text}'")
    return text


async def prepare_for_rag(urls: List[str], embed_model: EmbedModelType = 'all-MiniLM-L6-v2', urls_limit: Optional[int] = None, chunk_size: int = 200, chunk_overlap: int = 20, query: str = "", tokenizer_model: Optional[ModelType] = None) -> tuple:
    model = SentenceTransformerRegistry.load_model(embed_model)
    all_orig_documents = []
    all_documents = []
    all_links = []
    all_results = []
    seen_texts = set()
    total_tokens = 0
    total_high_score_tokens = 0
    total_mtld_high_score_average = 0
    HIGH_QUALITY_SCORE = 0.6
    TARGET_HIGH_SCORE_TOKENS = 2000
    TARGET_TOKENS = 10000

    async for url, status, html in scrape_urls(urls, show_progress=True, limit=urls_limit):
        if status == "completed" and html:
            sub_url_dir = format_sub_url_dir(url)
            sub_output_dir = os.path.join(OUTPUT_DIR, "pages", sub_url_dir)
            os.makedirs(sub_output_dir, exist_ok=True)

            links = set(scrape_links(html, url))
            links = [link for link in links if (
                link != url if isinstance(link, str) else link["url"] != url)]
            save_file(links, os.path.join(sub_output_dir, "links.json"))
            all_links.extend(links)

            html = preprocess_html(html)
            save_file(html, f"{sub_output_dir}/page.html")
            doc_markdown = convert_html_to_markdown(html)
            save_file(doc_markdown, f"{sub_output_dir}/page.md")

            doc_analysis = analyze_markdown(doc_markdown)
            save_file(doc_analysis, f"{sub_output_dir}/analysis.json")
            doc_markdown_tokens = parse_markdown(doc_markdown)
            save_file(doc_markdown_tokens,
                      f"{sub_output_dir}/markdown_tokens.json")

            original_docs = derive_by_header_hierarchy(doc_markdown)
            save_file(original_docs, f"{sub_output_dir}/docs.json")
            all_orig_documents.extend(original_docs)

            _tokenizer = get_string_tokenizer_fn(
                tokenizer_model) if tokenizer_model else None
            _detokenizer_fn = get_string_detokenizer_fn(
                tokenizer_model) if tokenizer_model else None

            sections = chunk_headers_by_hierarchy(
                doc_markdown,
                chunk_size=chunk_size,
                tokenizer=_tokenizer,
            )
            save_file(sections, f"{sub_output_dir}/chunks.json")

            documents = []
            for section in tqdm(sections, desc=f"Chunking sections for {url}", leave=False):
                section = section.copy()

                section["doc_id"] = generate_unique_id()
                section["chunk_id"] = generate_unique_id()
                section["url"] = url
                section["parent_header"] = section.get(
                    "parent_header")
                section["parent_level"] = section.get("parent_level")
                section["level"] = section.get("level")
                text_key = section["content"].strip().replace(
                    "\n", " ").replace("\r", " ")
                text_key = re.sub(r"\s+", " ", text_key)
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)
                if section["level"] is None:
                    logger.debug(f"Chunk with None level: {section}")
                documents.append(section)
            if not documents:
                logger.warning(f"No documents collected for {url}.")
                continue
            filtered_documents = []
            token_counts: List[int] = count_tokens(
                tokenizer_model, [clean_markdown_links(f"{doc["header"]}\n{doc["content"]}") for doc in documents], prevent_total=True)
            for doc, num_tokens in zip(documents, token_counts):
                link_to_text_ratio_result = link_to_text_ratio(
                    f"{doc["header"].lstrip('#').strip()}\n{doc["content"]}", threshold=0.25)
                doc["link_to_text_ratio"] = link_to_text_ratio_result["ratio"]
                doc["num_tokens"] = num_tokens

                if doc["parent_header"]:
                    doc["parent_header"] = clean_markdown_links(
                        doc["parent_header"])
                doc["header"] = clean_markdown_links(doc["header"])
                doc["content"] = clean_markdown_links(doc["content"])

                readability = analyze_readability(doc["content"])
                doc["mtld"] = readability["mtld"]
                doc["mtld_category"] = readability["mtld_category"]
                doc["doc_index"] = len(all_documents) + len(documents)
                doc["word_count"] = len(word_tokenize(doc["content"]))

                if doc["header"].strip() and doc["word_count"] >= 8 and readability["mtld"] > 0.0 and not link_to_text_ratio_result["is_link_heavy"]:
                    filtered_documents.append(doc)
            documents = filtered_documents
            if not documents:
                continue

            texts = []
            prev_content_tokens = []
            for i, doc in enumerate(documents):
                content_tokens = _tokenizer(
                    doc["content"]) if _tokenizer else word_tokenize(doc["content"])
                if i > 0 and chunk_overlap > 0 and doc["chunk_index"] > 0:
                    prev_overlap = prev_content_tokens[-chunk_overlap:] if len(
                        prev_content_tokens) >= chunk_overlap else prev_content_tokens
                    prev_content = _detokenizer_fn(prev_overlap) + " "
                else:
                    prev_content = ""
                text = f"{doc['parent_header'] or ''}\n{doc['header']}\n{prev_content}{doc['content']}"
                texts.append(text)
                prev_content_tokens = content_tokens
            texts = [preprocess_text(text)
                     for text in texts]
            save_file(
                {
                    "preprocessed_query": preprocess_text(query),
                    "count": len(documents),
                    "preprocessed_texts": [
                        {
                            "text": text,
                            "doc_index": doc["doc_index"],
                            "chunk_index": doc["chunk_index"],
                            "header": doc["header"],
                        }
                        for text, doc in zip(texts, documents)
                    ]
                },
                f"{sub_output_dir}/texts_for_embeddings.json"
            )

            generated_embeddings = generate_embeddings(
                texts, model, show_progress=True)
            embeddings = []
            for i, (doc, embedding) in enumerate(zip(documents, generated_embeddings)):
                doc["embedding"] = embedding
                doc["doc_index"] = len(all_documents) + i
                embeddings.append(doc)
            embedding_matrix = np.array(
                [doc["embedding"] for doc in embeddings]).astype('float32')
            index = faiss.IndexFlatIP(embedding_matrix.shape[1])
            index.add(embedding_matrix)

            results = query_rag(index, embeddings, model, [], query,
                                k=None, threshold=-0.0, use_reranking=False)
            total_tokens += sum(result["num_tokens"] for result in results)
            high_score_tokens = sum(
                result["num_tokens"]
                for result in results
                if (
                    result["score"] >= HIGH_QUALITY_SCORE
                    and result.get("mtld_category") != "very_low"
                )
            )
            mtld_high_score_values = [
                result["mtld"]
                for result in results
                if (
                    result["score"] >= HIGH_QUALITY_SCORE
                    and result.get("mtld_category") != "very_low"
                )
            ]
            mtld_high_score_average = (
                sum(mtld_high_score_values) / len(mtld_high_score_values)
                if mtld_high_score_values else 0
            )
            save_file({
                "query": query,
                "count": len(results),
                "total_tokens": total_tokens,
                "high_score_tokens": high_score_tokens,
                "mtld_high_score_average": mtld_high_score_average,
                "results": results
            }, f"{sub_output_dir}/rag_results.json")

            save_file(
                {
                    "query": query,
                    "count": len(documents),
                    "total_tokens": sum(doc["num_tokens"] for doc in documents),
                    "high_score_tokens": high_score_tokens,
                    "mtld_high_score_average": mtld_high_score_average,
                    "documents": [
                        {k: v for k, v in doc.items() if k != "embedding"}
                        for doc in documents if "embedding" in doc
                    ]
                },
                f"{sub_output_dir}/docs_with_scores.json"
            )

            # if high_score_tokens:
            all_results.extend(results)
            all_documents.extend(embeddings)

            total_tokens += sum(result["num_tokens"] for result in results)
            total_high_score_tokens += high_score_tokens
            total_mtld_high_score_average += round(mtld_high_score_average, 2)

        if total_high_score_tokens >= TARGET_HIGH_SCORE_TOKENS or total_tokens >= TARGET_TOKENS:
            logger.info(
                f"Stopping scrape: {total_tokens} tokens collected.")
            break
    if not all_documents:
        logger.warning("No documents collected after scraping.")
        return None, [], model, [], []

    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    for i, result in enumerate(all_results, 1):
        result["rank"] = i

    save_file(all_links, os.path.join(OUTPUT_DIR, "links.json"))
    save_file({
        "count": len(all_documents),
        "total_tokens": sum(doc.get("num_tokens", 0) for doc in all_documents),
        "total_high_score_tokens": total_high_score_tokens,
        "total_mtld_high_score_average": total_mtld_high_score_average,
        "documents": [
            {
                "doc_id": doc.get("doc_id"),
                "chunk_id": doc.get("chunk_id"),
                "url": doc.get("url"),
                "parent_level": doc.get("parent_level"),
                "level": doc.get("level"),
                "parent_header": doc.get("parent_header"),
                "header": doc.get("header"),
                "content": doc.get("content"),
                "num_tokens": doc.get("num_tokens"),
                "score": result.get("score"),
                "mtld": doc.get("mtld"),
                "mtld_category": doc.get("mtld_category"),
                "word_count": doc.get("word_count"),
                "link_to_text_ratio": doc.get("link_to_text_ratio"),
                "doc_index": doc.get("doc_index"),
                "chunk_index": doc.get("chunk_index", 0)
            } for doc in all_documents
        ]
    }, f"{OUTPUT_DIR}/docs.json")
    save_file({
        "query": query,
        "count": len(all_results),
        "total_tokens": sum(result.get("num_tokens", 0) for result in all_results),
        "total_high_score_tokens": total_high_score_tokens,
        "total_mtld_high_score_average": total_mtld_high_score_average,
        "results": [
            {
                "merged_doc_id": result.get("merged_doc_id"),
                "chunk_id": result.get("chunk_id"),
                "doc_index": result.get("doc_index"),
                "chunk_index": result.get("chunk_index", 0),
                "header": result.get("header"),
                "content": result.get("content"),
                "url": result.get("url"),
                "similarity_scores": result.get("similarity_scores"),
                "score": result.get("score"),
                "mtld": result.get("mtld"),
                "mtld_category": result.get("mtld_category"),
                "word_count": result.get("word_count"),
                "link_to_text_ratio": result.get("link_to_text_ratio"),
                "num_tokens": result.get("num_tokens"),
                "parent_header": result.get("parent_header"),
                "parent_level": result.get("parent_level"),
                "level": result.get("level"),
                "selected_doc_ids": result.get("selected_doc_ids")
            } for result in all_results
        ]
    }, f"{OUTPUT_DIR}/rag_results.json")
    score_ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    url_summary = defaultdict(
        lambda: {"scores": [], "num_tokens": 0, "score_categories": defaultdict(int)})
    for result in all_results:
        url = result.get("url", "Unknown Source")
        score = result.get("score", 0.0)
        num_tokens = result.get("num_tokens", 0)
        url_summary[url]["scores"].append(float(score))
        url_summary[url]["num_tokens"] += num_tokens
        for lower, upper in score_ranges:
            if lower <= score < upper:
                url_summary[url]["score_categories"][f"{upper}"] += 1
                break
    header_insights = {}
    for result in all_results:
        url = result.get("url", "Unknown Source")
        header = result.get("header", "No Header")
        score = result.get("score", 0.0)
        if url not in header_insights:
            header_insights[url] = {}
        if header not in header_insights[url]:
            header_insights[url][header] = {
                "count": 0,
                "scores": [],
                "avg_score": 0.0,
                "max_score": 0.0,
                "min_score": 0.0,
            }
        header_insights[url][header]["count"] += 1
        header_insights[url][header]["scores"].append(float(score))

    for url in header_insights:
        for header in header_insights[url]:
            scores = header_insights[url][header]["scores"]
            if scores:
                header_insights[url][header]["avg_score"] = sum(
                    scores) / len(scores)
                header_insights[url][header]["max_score"] = max(scores)
                header_insights[url][header]["min_score"] = min(scores)
            else:
                header_insights[url][header]["avg_score"] = 0.0
                header_insights[url][header]["max_score"] = 0.0
                header_insights[url][header]["min_score"] = 0.0

    summary_data = [
        {
            "url": url,
            "count": len(data["scores"]),
            "max_score": max(data["scores"]) if data["scores"] else 0.0,
            "avg_score": sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0.0,
            "min_score": min(data["scores"]) if data["scores"] else 0.0,
            "total_tokens": data["num_tokens"],
            "score_categories": {
                f"{upper}": data["score_categories"][f"{upper}"]
                for lower, upper in score_ranges
            },
            "header_insights": [
                {
                    "header": header,
                    "count": header_insights[url][header]["count"],
                    "avg_score": header_insights[url][header]["avg_score"],
                    "max_score": header_insights[url][header]["max_score"],
                    "min_score": header_insights[url][header]["min_score"],
                }
                for header in header_insights.get(url, {})
            ]
        } for url, data in url_summary.items()
    ]
    save_file(summary_data, f"{OUTPUT_DIR}/summary.json")
    logger.info(
        f"Saved combined docs.json with {len(all_documents)} documents to {OUTPUT_DIR}/docs.json")
    logger.info(
        f"Saved combined rag_results.json with {len(all_results)} results to {OUTPUT_DIR}/rag_results.json")
    logger.info(
        f"Saved summary.json with insights for {len(url_summary)} URLs to {OUTPUT_DIR}/summary.json")
    logger.info(f"Clustering {len(all_documents)} documents...")
    return all_orig_documents, all_documents, all_results


def compute_similarity(
    query_embedding: np.ndarray,
    header_text: str,
    content_text: str,
    parent_header: Optional[str],
    model: SentenceTransformer,
    mode: Literal["average", "max"] = "average"
) -> tuple[float, Dict[str, float]]:
    """Compute similarity score and individual similarities between query, combined headers, and content.

    Returns score based on specified mode: 'average' or 'max'.
    """
    def strip_hashtags(text: str) -> str:
        return text.lstrip('#').strip() if text else ""

    header_clean = strip_hashtags(header_text) if header_text else ""
    parent_header_clean = strip_hashtags(
        parent_header) if parent_header else ""
    content_clean = preprocess_text(content_text) if content_text else ""

    combined_header = f"{parent_header_clean}\n{header_clean}".strip()

    texts_to_embed = [text for text in [
        combined_header, content_clean] if text]
    similarities = {
        "query_vs_combined_header": 0.0,
        "query_vs_content": 0.0
    }

    if not texts_to_embed:
        return 0.0, similarities

    text_embeddings = model.encode(
        texts_to_embed,
        convert_to_tensor=False,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    computed_similarities = []

    if combined_header:
        similarities["query_vs_combined_header"] = cosine_similarity(
            [query_embedding], [text_embeddings[0]])[0][0]
        computed_similarities.append(similarities["query_vs_combined_header"])

    if content_clean:
        content_idx = 1 if combined_header else 0
        similarities["query_vs_content"] = cosine_similarity(
            [query_embedding], [text_embeddings[content_idx]])[0][0]
        computed_similarities.append(similarities["query_vs_content"])

    if mode == "max":
        score = float(np.max(computed_similarities)
                      ) if computed_similarities else 0.0
    else:
        score = float(np.mean(computed_similarities)
                      ) if computed_similarities else 0.0

    return score, similarities


def query_rag(
    index,
    embeddings: List[Dict],
    model: SentenceTransformer,
    merge_info: List[Dict],
    query: str,
    k: Optional[int] = None,
    threshold: float = 0.0,
    cross_encoder_model: EmbedModelType = 'cross-encoder/ms-marco-MiniLM-L12-v2',
    use_reranking: bool = True
) -> List[Dict]:
    if not k:
        k = len(embeddings)
    query = preprocess_text(query)
    query_embedding = model.encode(query, convert_to_tensor=False,
                                   show_progress_bar=False, normalize_embeddings=True).astype('float32')
    D, I = index.search(np.array([query_embedding]), k)
    results = []
    seen_doc_ids = set()

    if use_reranking:
        cross_encoder = CrossEncoderRegistry.load_model(cross_encoder_model)
        pairs = [[query, embeddings[idx]["content"]] for idx in I[0]]
        cross_scores = cross_encoder.predict(pairs)
        scores = cross_scores
        # Placeholder for cross-encoder case
        individual_scores = [{} for _ in range(len(I[0]))]
    else:
        scores = []
        individual_scores = []
        for idx in I[0]:
            doc = embeddings[idx]
            score, sim_scores = compute_similarity(
                query_embedding=query_embedding,
                header_text=doc.get("header"),
                content_text=doc.get("content"),
                parent_header=doc.get("parent_header"),
                model=model,
                mode="average"  # or "average", depending on desired behavior
            )
            scores.append(score)
            individual_scores.append(sim_scores)

    rank = 0
    for idx, score, ind_scores in zip(I[0], scores, individual_scores):
        doc_id = embeddings[idx]["doc_id"]
        if (use_reranking and score < threshold) or doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        embeddings[idx]["score"] = float(score)
        merge_entry = next(
            (entry for entry in merge_info if entry["merged_doc_id"] == doc_id), None)
        selected_doc_id = merge_entry["original_doc_ids"][0] if merge_entry and len(
            merge_entry["original_doc_ids"]) > 1 else doc_id
        rank += 1
        results.append({
            "rank": rank,
            "score": float(score),
            "similarity_scores": ind_scores,
            "merged_doc_id": doc_id,
            "chunk_id": embeddings[idx]["chunk_id"],
            "doc_index": embeddings[idx]["doc_index"],
            "chunk_index": embeddings[idx].get("chunk_index", 0),
            "url": embeddings[idx]["url"],
            "mtld": embeddings[idx]["mtld"],
            "mtld_category": embeddings[idx]["mtld_category"],
            "word_count": embeddings[idx]["word_count"],
            "link_to_text_ratio": embeddings[idx]["link_to_text_ratio"],
            "num_tokens": embeddings[idx]["num_tokens"],
            "selected_doc_ids": [selected_doc_id],
            "parent_level": embeddings[idx].get("parent_level", None),
            "level": embeddings[idx].get("level", None),
            "parent_header": embeddings[idx].get("parent_header", None),
            "header": embeddings[idx]["header"],
            "content": embeddings[idx]["content"],
        })
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results


def group_results_by_url_for_llm_context(
    documents: List[Dict],
    llm_model: 'LLMModelType',
    max_tokens: int = 2000,
    buffer: int = 100
) -> str:
    def strip_hashtags(text: str) -> str:
        if text:
            return text.lstrip('#').strip()
        return text

    high_score_docs = [
        doc for doc in documents if doc["score"] >= 0.6
    ]
    med_score_docs = [
        doc for doc in documents if 0.35 <= doc["score"] < 0.6
    ]

    high_score_docs_sorted = sorted(
        high_score_docs,
        key=lambda x: x.get("score", 0),
        reverse=True
    )
    med_score_docs_sorted = sorted(
        med_score_docs,
        key=lambda x: x.get("score", 0),
        reverse=True
    )

    filtered_documents = []
    total_tokens = 0

    for doc in high_score_docs_sorted:
        doc_tokens = doc.get("num_tokens", 0)
        if total_tokens + doc_tokens > max_tokens - buffer:
            break
        filtered_documents.append(doc)
        total_tokens += doc_tokens

    if total_tokens < max_tokens - buffer:
        for doc in med_score_docs_sorted:
            doc_tokens = doc.get("num_tokens", 0)
            if total_tokens + doc_tokens > max_tokens - buffer:
                break
            filtered_documents.append(doc)
            total_tokens += doc_tokens

    documents = filtered_documents

    tokenizer = get_tokenizer_fn(
        llm_model, add_special_tokens=False, remove_pad_tokens=True)
    separator = "\n\n"
    separator_tokens = len(tokenizer.encode(separator))
    sorted_docs = sorted(
        documents, key=lambda x: x.get("score", 0), reverse=True)
    filtered_docs = []
    total_tokens = 0
    grouped_temp: DefaultDict[str, List[Dict]] = defaultdict(list)
    seen_header_text: DefaultDict[str, set] = defaultdict(set)

    for doc in sorted_docs:
        text = doc.get("content", "")
        url = doc.get("url", "Unknown Source")
        parent_header = doc.get("parent_header", "None")
        header = doc.get("header", None)
        level = doc.get("level", 0)
        parent_level = doc.get("parent_level", None)

        if not isinstance(text, str):
            logger.debug(
                f"Non-string content found for url: {url}, doc_index: {doc.get('doc_index', 0)}, type: {type(text)}. Converting to string.")
            text = str(text) if text else ""

        doc_tokens = doc.get("num_tokens", len(tokenizer.encode(text)))
        header_tokens = 0

        if not grouped_temp[url]:
            header_tokens += len(tokenizer.encode(
                f"<!-- Source: {url} -->\n\n"))
            header_tokens += separator_tokens if filtered_docs else 0

        parent_header_key = strip_hashtags(
            parent_header) if parent_header and parent_header != "None" else None
        header_key = strip_hashtags(header) if header else None

        if header_key and header_key not in seen_header_text[url] and level >= 0:
            header_tokens += len(tokenizer.encode(f"{header}\n\n"))
            seen_header_text[url].add(header_key)

        additional_tokens = doc_tokens + header_tokens + separator_tokens

        filtered_docs.append(doc)
        grouped_temp[url].append(doc)
        total_tokens += additional_tokens

    context_blocks = []
    total_tokens = 0
    for url, docs in grouped_temp.items():
        block = f"<!-- Source: {url} -->\n\n"
        block_tokens = len(tokenizer.encode(block))
        seen_header_text_in_block = set()
        docs = sorted(docs, key=lambda x: (
            x.get("doc_index", 0),
            x.get("chunk_index", 0)
        ))
        for doc in docs:
            header = doc.get("header", None)
            parent_header = doc.get("parent_header", "None")
            text = doc.get("content", "")

            if not isinstance(text, str):
                logger.debug(
                    f"Non-string content in block for url: {url}, doc_index: {doc.get('doc_index', 0)}, type: {type(text)}. Converting to string.")
                text = str(text) if text else ""

            doc_tokens = doc.get("num_tokens", len(tokenizer.encode(text)))
            doc_level = doc.get("level", 0) if doc.get(
                "level") is not None else 0
            parent_level = doc.get("parent_level", None)
            parent_header_key = strip_hashtags(
                parent_header) if parent_header and parent_header != "None" else None
            header_key = strip_hashtags(header) if header else None

            has_matching_child = any(
                strip_hashtags(d.get("header", "")) == parent_header_key
                for d in docs
                if d.get("header") and d.get("level", 0) >= 0
            )
            if parent_header_key and parent_level is not None and not has_matching_child and parent_header_key not in seen_header_text_in_block:
                parent_header_text = f"{parent_header}\n\n"
                block += parent_header_text
                block_tokens += len(tokenizer.encode(parent_header_text))
                seen_header_text_in_block.add(parent_header_key)

            if header_key and header_key not in seen_header_text_in_block and doc_level >= 0:
                subheader_text = f"{header}\n\n"
                block += subheader_text
                block_tokens += len(tokenizer.encode(subheader_text))
                seen_header_text_in_block.add(header_key)

            block += text + "\n\n"
            block_tokens += doc_tokens + separator_tokens

        if block_tokens > len(tokenizer.encode(f"<!-- Source: {url} -->\n\n")):
            context_blocks.append(block.strip())
            total_tokens += block_tokens
        else:
            logger.warning(
                f"Empty block for {url} after processing; skipping.")

    result = "\n\n".join(context_blocks)
    final_token_count = len(tokenizer.encode(result))
    contexts_data = {
        "query": documents[0].get("query", "") if documents else "",
        "count": len(filtered_docs),
        "total_tokens": sum(doc.get("num_tokens", 0) for doc in filtered_docs),
        "results": [
            {
                "rank": doc.get("rank"),
                "merged_doc_id": doc.get("merged_doc_id"),
                "chunk_id": doc.get("chunk_id"),
                "doc_index": doc.get("doc_index"),
                "chunk_index": doc.get("chunk_index", 0),
                "header": doc.get("header"),
                "content": doc.get("content"),
                "url": doc.get("url"),
                "score": doc.get("score"),
                "mtld": doc.get("mtld"),
                "mtld_category": doc.get("mtld_category"),
                "word_count": doc.get("word_count"),
                "link_to_text_ratio": doc.get("link_to_text_ratio"),
                "num_tokens": doc.get("num_tokens"),
                "parent_header": doc.get("parent_header"),
                "parent_level": doc.get("parent_level"),
                "level": doc.get("level"),
                "selected_doc_ids": doc.get("selected_doc_ids")
            } for doc in filtered_docs
        ]
    }
    save_file(contexts_data, f"{OUTPUT_DIR}/contexts.json")
    if final_token_count > max_tokens:
        logger.warning(
            f"Final context exceeds max_tokens: {final_token_count} > {max_tokens}")
    else:
        logger.debug(
            f"Grouped context created with {final_token_count} tokens for {len(grouped_temp)} URLs")
    return result


async def main():
    p = argparse.ArgumentParser(
        description="Run semantic search and processing pipeline.")
    p.add_argument("query_pos", type=str, nargs="?",
                   help="Search query as positional argument")
    p.add_argument("-q", "--query", type=str,
                   help="Search query using optional flag")
    p.add_argument("-r", "--reset_cache", action="store_true",
                   default=False, help="Reset and ignore cached search results")
    args = p.parse_args()

    global OUTPUT_DIR

    query = args.query if args.query else args.query_pos or "Top isekai anime 2025."

    query_sub_dir = format_sub_dir(query)

    OUTPUT_DIR = f"{OUTPUT_DIR}/{query_sub_dir}"
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    llm_model: LLMModelType = "qwen3-1.7b-4bit-dwq-053125"
    embed_model: EmbedModelType = "all-MiniLM-L6-v2"
    chunk_size = 200
    chunk_overlap = 20
    urls_limit = 10
    use_cache = not args.reset_cache

    save_file(query, f"{OUTPUT_DIR}/query.md")
    save_file({
        "query": query,
        "urls_limit": urls_limit,
        "use_cache": use_cache,
        # Save the LLM model and chunking parameters to input.json for reproducibility
        "llm_model": llm_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    }, f"{OUTPUT_DIR}/input.json")

    search_results = search_data(query, use_cache=use_cache)
    save_file({"query": query, "count": len(search_results),
              "results": search_results}, f"{OUTPUT_DIR}/search_results.json")
    urls = [result["url"] for result in search_results]
    all_orig_documents, all_documents, all_results = await prepare_for_rag(urls, embed_model=embed_model, urls_limit=urls_limit, chunk_size=chunk_size, chunk_overlap=chunk_overlap, query=query, tokenizer_model=llm_model)

    # labels: List[str] = generate_labels(query, model_path=llm_model)
    # save_file({"text": query, "labels": labels}, f"{OUTPUT_DIR}/labels.json")

    # embeddings, merge_info = merge_similar_docs(
    #     all_documents, similarity_threshold=0.8)

    # if not embeddings:
    #     logger.error("No data indexed, exiting.")
    #     return
    # merged_embedding_matrix = np.array(
    #     [doc["embedding"] for doc in embeddings]).astype('float32')
    # index = faiss.IndexFlatIP(merged_embedding_matrix.shape[1])
    # index.add(merged_embedding_matrix)

    sorted_results = sorted(
        all_results, key=lambda x: x["score"], reverse=True)
    save_file(sorted_results, f"{OUTPUT_DIR}/contexts_search_results.json")
    result_texts = [
        f"{r["header"].lstrip('#').strip()}\n{r["content"]}" for r in sorted_results]

    id_to_result = {r["merged_doc_id"]: r for r in sorted_results}

    ids = [r["merged_doc_id"] for r in sorted_results]
    grouped_results = group_similar_texts(
        result_texts, threshold=0.7, model_name=embed_model, ids=ids)

    # diverse_results = sort_by_mmr_diversity(result_texts, ids=ids)

    # Replace grouped_results ids by {id, rank, score, parent_header, header, content}
    contexts_grouped_results = []
    for group in grouped_results:
        group_info = []
        for idx, doc_id in enumerate(group):
            r = id_to_result[doc_id]
            group_info.append({
                "id": doc_id,
                "chunk_index": r.get("chunk_index"),
                "rank": r.get("rank"),
                "score": r.get("score"),
                "header": r.get("header"),
                "content": r.get("content"),
            })
        contexts_grouped_results.append(group_info)

    save_file(contexts_grouped_results,
              f"{OUTPUT_DIR}/contexts_grouped_results.json")

    # Map back to sorted_results
    unique_results = []
    for group in grouped_results:
        # Each group is a list of ids; pick the first two ids as representatives (if available)
        for idx in range(min(2, len(group))):
            result = id_to_result[group[idx]]
            unique_results.append(result)

    save_file(unique_results, f"{OUTPUT_DIR}/contexts_before_max_filter.json")

    print("\nQuery Results (With Reranking):")
    for i, result in enumerate(unique_results, 1):
        print(f"\nResult {i}:")
        print(f"Header: {result['header'] or 'No header'}")
        print(f"Text: {result['content'][:200]}...")
        print(f"URL: {result['url']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Token Count: {result['num_tokens']}")
        print(f"Parent Header: {result['parent_header'] or 'None'}")
        print(f"Parent Level: {result['parent_level'] or 'None'}")
        print(f"Level: {result['level'] or 'None'}")
        print(f"Selected Doc IDs: {result['selected_doc_ids']}")

    context = group_results_by_url_for_llm_context(unique_results, llm_model)
    save_file(context, f"{OUTPUT_DIR}/context.md")
    save_file({"num_tokens": count_tokens(llm_model, context)},
              f"{OUTPUT_DIR}/context_info.json")
    mlx = MLXModelRegistry.load_model(llm_model)
    prompt = PROMPT_TEMPLATE.format(query=query, context=context)
    llm_response = mlx.chat(prompt, llm_model, temperature=0.7, verbose=True)
    save_file(llm_response["content"], f"{OUTPUT_DIR}/response.md")

if __name__ == "__main__":
    asyncio.run(main())
