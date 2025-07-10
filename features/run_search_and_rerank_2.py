import argparse
from collections import defaultdict
import shutil
import string
from jet.code.html_utils import preprocess_html
from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.code.markdown_utils._markdown_analyzer import analyze_markdown
from jet.logger import logger
from jet.models.embeddings.base import generate_embeddings
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.models.tokenizer.base import count_tokens, get_tokenizer_fn
from jet.wordnet.similarity import group_similar_texts
import asyncio
import os
from typing import DefaultDict, List, Dict, Optional
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
import logging
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


def clean_html(html: str, language: str = "English", max_link_density: float = 0.2, max_link_ratio: float = 0.3) -> List:
    paragraphs = justext.justext(
        html,
        justext.get_stoplist(language),
        max_link_density=max_link_density,
        length_low=50,
        length_high=150,
        no_headings=False
    )
    filtered_paragraphs = [
        p for p in paragraphs
        if not p.is_boilerplate and p.links_density() < max_link_ratio
    ]
    return filtered_paragraphs


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
        "level": 0  # Default to 0 for non-heading content
    }
    header_stack = []
    header_tags = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

    for paragraph in paragraphs:
        if paragraph.is_heading:
            # Extract header tag from xpath using regex (e.g., '/html/body/h1' -> 'h1')
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
                    "parent_header": f"{'#' * parent['level']} {parent['text']}" if parent else None,
                    "parent_level": parent["level"] if parent else None,
                    "level": level
                }
                header_stack.append({"level": level, "text": paragraph.text})
            else:
                # Treat as content if header tag is invalid
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
                "text": merged_text,
                "num_tokens": merged_num_tokens,
                # Fixed typo: sock_doc to best_doc
                "header": best_doc["header"],
                "url": best_doc["url"],
                "xpath": best_doc["xpath"],
                "score": best_doc.get("score", None),
                "mtld": best_doc["mtld"],
                "mtld_category": best_doc["mtld_category"],
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
                "text": doc["content"],
                "num_tokens": doc["num_tokens"],
                "header": doc["header"],
                "url": doc["url"],
                "xpath": doc["xpath"],
                "score": doc.get("score", None),
                "mtld": doc["mtld"],
                "mtld_category": doc["mtld_category"],
                "parent_header": doc.get("parent_header", None),
                "parent_level": doc.get("parent_level", None),
                "level": doc.get("level", None),
                "chunk_index": doc.get("chunk_index", None),
                "doc_index": doc.get("doc_index", None)
            })
    save_file(merge_info, f"{OUTPUT_DIR}/merged_docs.json")
    return merged_docs, merge_info


async def prepare_for_rag(urls: List[str], model_name: EmbedModelType = 'all-MiniLM-L6-v2', urls_limit: Optional[int] = None, max_retries: int = 3, query: str = "") -> tuple:
    model = SentenceTransformerRegistry.load_model(model_name)
    all_documents = []
    all_links = []
    all_results = []
    seen_texts = set()
    total_tokens = 0
    high_quality_docs = 0
    MIN_HIGH_QUALITY_DOCS = 5
    HIGH_QUALITY_SCORE = 0.5
    TARGET_TOKEN_COUNT = 1500
    TOKEN_BUFFER = 200
    async for url, status, html in scrape_urls(urls, show_progress=True, limit=urls_limit):
        if status == "completed" and html:
            links = set(scrape_links(html, url))
            links = [link for link in links if (
                link != url if isinstance(link, str) else link["url"] != url)]
            all_links.extend(links)
            save_file(all_links, os.path.join(OUTPUT_DIR, "links.json"))
            sub_url_dir = format_sub_url_dir(url)
            sub_output_dir = os.path.join(OUTPUT_DIR, "pages", sub_url_dir)
            os.makedirs(sub_output_dir, exist_ok=True)
            html = preprocess_html(html)
            save_file(html, f"{sub_output_dir}/page.html")
            doc_markdown = convert_html_to_markdown(html)
            save_file(doc_markdown, f"{sub_output_dir}/page.md")
            doc_analysis = analyze_markdown(doc_markdown)
            save_file(doc_analysis, f"{sub_output_dir}/analysis.json")
            paragraphs = clean_html(
                html, max_link_density=0.15, max_link_ratio=0.3)
            save_file(paragraphs, f"{sub_output_dir}/elements.json")
            sections = separate_by_headers(paragraphs)
            documents = []
            for section in tqdm(sections, desc=f"Chunking sections for {url}", leave=False):
                markdown_text = (f"{section['header']}\n" + "\n".join(
                    section["content"]) if section["header"] else "\n".join(section["content"]))
                chunks = chunk_headers_by_hierarchy(
                    markdown_text,
                    chunk_size=200,
                )
                for chunk in chunks:
                    chunk["doc_id"] = generate_unique_id()
                    chunk["chunk_id"] = generate_unique_id()
                    chunk["url"] = url
                    chunk["xpath"] = section["xpath"]
                    chunk["parent_header"] = section.get(
                        "parent_header", chunk.get("parent_header", None))
                    chunk["parent_level"] = section.get(
                        "parent_level", chunk.get("parent_level", None))
                    # Ensure level is always an integer
                    chunk["level"] = section.get(
                        "level", chunk.get("level", 0))
                    text_key = chunk["content"].strip().replace(
                        "\n", " ").replace("\r", " ")
                    text_key = re.sub(r"\s+", " ", text_key)
                    if text_key in seen_texts:
                        continue
                    seen_texts.add(text_key)
                    chunk["num_tokens"] = len(word_tokenize(chunk["content"]))
                    if chunk["level"] is None:  # Debug check
                        logger.debug(f"Chunk with None level: {chunk}")
                    documents.append(chunk)
            if not documents:
                logger.warning(f"No documents collected for {url}.")
                continue
            for doc in documents:
                readability = analyze_readability(doc["content"])
                doc["mtld"] = readability["mtld"]
                doc["mtld_category"] = readability["mtld_category"]
                doc["doc_index"] = len(all_documents) + len(documents)
            texts = [doc["content"] for doc in documents]
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
                                k=20, threshold=-0.0, use_reranking=False)
            total_tokens += sum(result["num_tokens"] for result in results)
            high_quality_docs += sum(
                1 for result in results if result["score"] > HIGH_QUALITY_SCORE)
            all_results.extend(results)
            save_file({"query": query, "count": len(results), "total_tokens": sum(result["num_tokens"] for result in results),
                       "results": results}, f"{sub_output_dir}/rag_results.json")
            all_documents.extend(embeddings)
            save_file({"count": len(documents), "total_tokens": sum(doc["num_tokens"] for doc in documents),
                       "documents": [doc for doc in documents if "embedding" in doc]},
                      f"{sub_output_dir}/docs.json")
            if high_quality_docs >= MIN_HIGH_QUALITY_DOCS and total_tokens >= TARGET_TOKEN_COUNT - TOKEN_BUFFER:
                logger.info(
                    f"Stopping scrape: {high_quality_docs} high-quality docs and {total_tokens} tokens collected.")
                break
    if not all_documents:
        logger.warning("No documents collected after scraping.")
        return None, [], model, []
    # Save combined docs.json with all documents
    save_file({
        "count": len(all_documents),
        "total_tokens": sum(doc.get("num_tokens", 0) for doc in all_documents),
        "documents": [
            {
                "doc_id": doc.get("doc_id"),
                "chunk_id": doc.get("chunk_id"),
                "url": doc.get("url"),
                "xpath": doc.get("xpath"),
                "header": doc.get("header"),
                "parent_header": doc.get("parent_header"),
                "level": doc.get("level"),
                "parent_level": doc.get("parent_level"),
                "num_tokens": doc.get("num_tokens"),
                "mtld": doc.get("mtld"),
                "mtld_category": doc.get("mtld_category"),
                "doc_index": doc.get("doc_index"),
                "chunk_index": doc.get("chunk_index", 0)
            } for doc in all_documents
        ]
    }, f"{OUTPUT_DIR}/docs.json")
    # Save combined rag_results.json with all results
    save_file({
        "query": query,
        "count": len(all_results),
        "total_tokens": sum(result.get("num_tokens", 0) for result in all_results),
        "results": [
            {
                "merged_doc_id": result.get("merged_doc_id"),
                "chunk_id": result.get("chunk_id"),
                "doc_index": result.get("doc_index"),
                "chunk_index": result.get("chunk_index", 0),
                "header": result.get("header"),
                "text": result.get("text"),
                "url": result.get("url"),
                "score": result.get("score"),
                "mtld": result.get("mtld"),
                "mtld_category": result.get("mtld_category"),
                "num_tokens": result.get("num_tokens"),
                "parent_header": result.get("parent_header"),
                "parent_level": result.get("parent_level"),
                "level": result.get("level"),
                "selected_doc_ids": result.get("selected_doc_ids")
            } for result in all_results
        ]
    }, f"{OUTPUT_DIR}/rag_results.json")
    # Save summary.json with per-URL score insights and categories
    score_ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    url_summary = defaultdict(
        lambda: {"scores": [], "num_tokens": 0, "score_categories": defaultdict(int)})
    for result in all_results:
        url = result.get("url", "Unknown Source")
        score = result.get("score", 0.0)
        num_tokens = result.get("num_tokens", 0)
        url_summary[url]["scores"].append(float(score))
        url_summary[url]["num_tokens"] += num_tokens
        # Categorize score into ranges
        for lower, upper in score_ranges:
            if lower <= score < upper:
                url_summary[url]["score_categories"][f"{upper}"] += 1
                break
    summary_data = {
        "urls": [
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
                }
            } for url, data in url_summary.items()
        ]
    }
    save_file(summary_data, f"{OUTPUT_DIR}/summary.json")
    logger.info(
        f"Saved combined docs.json with {len(all_documents)} documents to {OUTPUT_DIR}/docs.json")
    logger.info(
        f"Saved combined rag_results.json with {len(all_results)} results to {OUTPUT_DIR}/rag_results.json")
    logger.info(
        f"Saved summary.json with insights for {len(url_summary)} URLs to {OUTPUT_DIR}/summary.json")
    logger.info(f"Clustering {len(all_documents)} documents...")
    merged_docs, merge_info = merge_similar_docs(
        all_documents, similarity_threshold=0.8)
    merged_embedding_matrix = np.array(
        [doc["embedding"] for doc in merged_docs]).astype('float32')
    merged_index = faiss.IndexFlatIP(merged_embedding_matrix.shape[1])
    merged_index.add(merged_embedding_matrix)
    return merged_index, merged_docs, model, merge_info


def query_rag(
    index,
    embeddings: List[Dict],
    model,
    merge_info: List[Dict],
    query: str,
    k: int = 10,
    threshold: float = 0.0,
    cross_encoder_model: str = 'cross-encoder/ms-marco-MiniLM-L-12-v2',
    use_reranking: bool = True
) -> List[Dict]:
    query_embedding = model.encode(query, convert_to_tensor=False,
                                   show_progress_bar=False, normalize_embeddings=True).astype('float32')
    D, I = index.search(np.array([query_embedding]), k)
    results = []
    seen_doc_ids = set()
    if use_reranking:
        cross_encoder = CrossEncoder(
            cross_encoder_model)  # Correct instantiation
        pairs = [[query, embeddings[idx]["content"]] for idx in I[0]]
        cross_scores = cross_encoder.predict(pairs)
        scores = cross_scores
    else:
        scores = D[0]
    for idx, score in zip(I[0], scores):
        doc_id = embeddings[idx]["doc_id"]
        if (use_reranking and score < threshold) or doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        embeddings[idx]["score"] = float(score)
        merge_entry = next(
            (entry for entry in merge_info if entry["merged_doc_id"] == doc_id), None)
        selected_doc_id = merge_entry["original_doc_ids"][0] if merge_entry and len(
            merge_entry["original_doc_ids"]) > 1 else doc_id
        results.append({
            "merged_doc_id": doc_id,
            "chunk_id": embeddings[idx]["chunk_id"],
            "doc_index": embeddings[idx]["doc_index"],
            "chunk_index": embeddings[idx].get("chunk_index", 0),
            "header": embeddings[idx]["header"],
            "text": embeddings[idx]["content"],
            "url": embeddings[idx]["url"],
            "score": float(score),
            "mtld": embeddings[idx]["mtld"],
            "mtld_category": embeddings[idx]["mtld_category"],
            "num_tokens": embeddings[idx]["num_tokens"],
            "parent_header": embeddings[idx].get("parent_header", None),
            "parent_level": embeddings[idx].get("parent_level", None),
            "level": embeddings[idx].get("level", None),
            "selected_doc_ids": [selected_doc_id]
        })
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results


def group_results_by_url_for_llm_context(
    documents: List[Dict],
    llm_model: LLMModelType,
    max_tokens: int = 2000,
    buffer: int = 100
) -> str:
    tokenizer = get_tokenizer_fn(
        llm_model, add_special_tokens=False, remove_pad_tokens=True)
    separator = "\n\n"
    separator_tokens = len(tokenizer.encode(separator))
    sorted_docs = sorted(
        documents, key=lambda x: x.get("score", 0), reverse=True)
    filtered_docs = []
    total_tokens = 0
    grouped_temp: DefaultDict[str, DefaultDict[str, List[Dict]]] = defaultdict(
        lambda: defaultdict(list))
    seen_header_text: DefaultDict[str, set] = defaultdict(
        set)  # Track all header text per URL

    for doc in sorted_docs:
        text = doc.get("text", "")
        url = doc.get("url", "Unknown Source")
        parent_header = doc.get("parent_header", "None")
        header = doc.get("header", None)
        level = doc.get("level", 0)  # Default to 0 to avoid None
        parent_level = doc.get("parent_level", None)
        doc_tokens = doc.get("num_tokens", len(tokenizer.encode(text)))
        header_tokens = 0

        # Calculate header tokens for URL
        if not grouped_temp[url]:
            header_tokens += len(tokenizer.encode(
                f"<!-- Source: {url} -->\n\n"))
            header_tokens += separator_tokens if filtered_docs else 0

        # Check for duplicate parent_header text
        parent_header_key = parent_header if parent_header and parent_header != "None" else None
        if parent_header_key and parent_header_key in seen_header_text[url]:
            logger.debug(
                f"Skipping duplicate parent_header '{parent_header}' for URL: {url}")
            continue
        if parent_header_key and parent_level is not None:
            header_tokens += len(tokenizer.encode(f"{parent_header}\n\n"))
            seen_header_text[url].add(parent_header_key)

        # Check for duplicate header text
        header_key = header if header else None
        if header_key and header_key in seen_header_text[url]:
            logger.debug(
                f"Skipping duplicate header '{header}' for URL: {url}, parent: {parent_header}")
            continue
        if header_key and header != parent_header and level >= 0:
            header_tokens += len(tokenizer.encode(f"{header}\n\n"))
            seen_header_text[url].add(header_key)

        additional_tokens = doc_tokens + header_tokens + separator_tokens
        if total_tokens + additional_tokens <= max_tokens - buffer:
            filtered_docs.append(doc)
            grouped_temp[url][parent_header].append(doc)
            total_tokens += additional_tokens
        else:
            logger.debug(
                f"Skipping document (score: {doc.get('score', 0)}): would exceed max_tokens ({total_tokens + additional_tokens} > {max_tokens - buffer})")

    context_blocks = []
    total_tokens = 0
    for url, parent_groups in grouped_temp.items():
        block = f"<!-- Source: {url} -->\n\n"
        block_tokens = len(tokenizer.encode(block))
        seen_header_text_in_block = set()  # Track headers within this block
        for parent_header, parent_docs in parent_groups.items():
            if not parent_docs:  # Skip empty parent_docs
                logger.debug(
                    f"Skipping empty parent_header group: {parent_header} for URL: {url}")
                continue
            parent_docs = sorted(parent_docs, key=lambda x: (
                x.get("level", 0) if x.get("level") is not None else 0,
                x.get("chunk_index", 0) if x.get(
                    "chunk_index") is not None else 0
            ))
            parent_level = parent_docs[0].get("parent_level", None)
            parent_header_key = parent_header if parent_header and parent_header != "None" else None
            if parent_header_key and parent_level is not None and parent_header_key not in seen_header_text_in_block:
                parent_header_text = f"{parent_header}\n\n"
                block += parent_header_text
                block_tokens += len(tokenizer.encode(parent_header_text))
                seen_header_text_in_block.add(parent_header_key)
            for doc in parent_docs:
                header = doc.get("header", None)
                text = doc.get("text", "")
                doc_tokens = doc.get("num_tokens", len(tokenizer.encode(text)))
                doc_level = doc.get("level", 0) if doc.get(
                    "level") is not None else 0
                header_key = header if header else None
                if header_key and header != parent_header and doc_level >= 0 and header_key not in seen_header_text_in_block:
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
                "merged_doc_id": doc.get("merged_doc_id"),
                "chunk_id": doc.get("chunk_id"),
                "doc_index": doc.get("doc_index"),
                "chunk_index": doc.get("chunk_index", 0),
                "header": doc.get("header"),
                "text": doc.get("text"),
                "url": doc.get("url"),
                "score": doc.get("score"),
                "mtld": doc.get("mtld"),
                "mtld_category": doc.get("mtld_category"),
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
    args = p.parse_args()
    query = args.query if args.query else args.query_pos or "Top isekai anime 2025."
    query_sub_dir = format_sub_dir(query)
    global OUTPUT_DIR
    OUTPUT_DIR = f"{OUTPUT_DIR}/{query_sub_dir}"
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    use_cache = True
    urls_limit = 10
    search_results = search_data(query, use_cache=use_cache)
    save_file({"query": query, "count": len(search_results),
              "results": search_results}, f"{OUTPUT_DIR}/search_results.json")
    urls = [result["url"] for result in search_results]
    index, embeddings, model, merge_info = await prepare_for_rag(urls, urls_limit=urls_limit, max_retries=3, query=query)
    if not embeddings:
        print("No data indexed, exiting.")
        return
    results = []
    pages_dir = os.path.join(OUTPUT_DIR, "pages")
    for sub_url_dir in os.listdir(pages_dir):
        rag_results_path = os.path.join(
            pages_dir, sub_url_dir, "rag_results.json")
        if os.path.exists(rag_results_path):
            with open(rag_results_path, 'r') as f:
                rag_data = json.load(f)
                results.extend(rag_data["results"])
    final_results = query_rag(index, embeddings, model, merge_info, query,
                              k=50, threshold=-1.0, use_reranking=True)
    results.extend(final_results)
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    seen_doc_ids = set()
    unique_results = []
    for result in results:
        if result["merged_doc_id"] not in seen_doc_ids:
            seen_doc_ids.add(result["merged_doc_id"])
            unique_results.append(result)
    print("\nQuery Results (With Reranking):")
    for i, result in enumerate(unique_results, 1):
        print(f"\nResult {i}:")
        print(f"Header: {result['header'] or 'No header'}")
        print(f"Text: {result['text'][:200]}...")
        print(f"URL: {result['url']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Token Count: {result['num_tokens']}")
        print(f"Parent Header: {result['parent_header'] or 'None'}")
        print(f"Parent Level: {result['parent_level'] or 'None'}")
        print(f"Level: {result['level'] or 'None'}")
        print(f"Selected Doc IDs: {result['selected_doc_ids']}")
    llm_model: LLMModelType = "qwen3-1.7b-4bit-dwq-053125"
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
