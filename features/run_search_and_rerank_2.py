from jet.models.embeddings.base import generate_embeddings
from jet.wordnet.similarity import group_similar_texts
import asyncio
import os
from typing import List, Dict, Optional
from jet.data.utils import generate_unique_id
from jet.file.utils import save_file
from jet.scrapers.hrequests_utils import scrape_urls
from jet.scrapers.utils import search_data
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
    if not header:
        return True
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
    current_section = {"header": None, "content": [], "xpath": None}
    header_tags = {"h1", "h2", "h3", "h4", "h5", "h6"}
    for paragraph in paragraphs:
        if paragraph.is_heading:
            if (current_section["content"] or current_section["header"]) and is_valid_header(current_section["header"]):
                sections.append(current_section)
            current_section = {
                "header": paragraph.text,
                "content": [],
                "xpath": paragraph.xpath
            }
        else:
            current_section["content"].append(paragraph.text)
    if (current_section["content"] or current_section["header"]) and is_valid_header(current_section["header"]):
        sections.append(current_section)
    return sections


def merge_similar_docs(embeddings: List[Dict], similarity_threshold: float = 0.7) -> tuple[List[Dict], List[Dict]]:
    texts = [f"{doc["header"]}\n{doc["content"]}" for doc in embeddings]
    embedding_matrix = [doc["embedding"] for doc in embeddings]

    # Group similar texts using group_similar_texts
    clusters = group_similar_texts(
        texts=texts,
        threshold=similarity_threshold,
        embeddings=embedding_matrix,
    )

    merged_chunks = []
    merge_info = []

    # Process each cluster
    for cluster_texts in clusters:
        # Find corresponding embedding documents for the cluster texts
        cluster_chunks = [embeddings[i]
                          for i, text in enumerate(texts) if text in cluster_texts]

        if len(cluster_chunks) > 1:
            # Merge similar chunks
            merged_text = "\n".join([chunk["content"]
                                     for chunk in cluster_chunks])
            merged_token_count = sum(chunk["num_tokens"]
                                     for chunk in cluster_chunks)
            merged_chunk_id = generate_unique_id()
            merged_chunk = cluster_chunks[0].copy()
            merged_chunk["chunk_id"] = merged_chunk_id
            merged_chunk["content"] = merged_text
            merged_chunk["num_tokens"] = merged_token_count
            merge_info.append({
                "merged_chunk_id": merged_chunk_id,
                "original_doc_ids": [chunk["chunk_id"] for chunk in cluster_chunks],
                "text": merged_text,
                "token_count": merged_token_count,
                "header": merged_chunk["header"],
                "url": merged_chunk["url"],
                "xpath": merged_chunk["xpath"],
                "score": merged_chunk.get("score", None)
            })
            merged_chunks.append(merged_chunk)
        else:
            # Single chunk, no merging needed
            chunk = cluster_chunks[0]
            merged_chunks.append(chunk)
            merge_info.append({
                "merged_chunk_id": chunk["chunk_id"],
                "original_doc_ids": [chunk["chunk_id"]],
                "text": chunk["content"],
                "token_count": chunk["num_tokens"],
                "header": chunk["header"],
                "url": chunk["url"],
                "xpath": chunk["xpath"],
                "score": chunk.get("score", None)
            })

    save_file(merge_info, f"{OUTPUT_DIR}/merged_chunks.json")
    return merged_chunks, merge_info


async def prepare_for_rag(urls: List[str], model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 32, max_retries: int = 3) -> tuple:
    model = SentenceTransformer(model_name)
    chunked_documents = []
    seen_texts = set()
    for url in tqdm(urls, desc="Scraping URLs"):
        for attempt in range(max_retries):
            try:
                async for u, status, html in scrape_urls([url], show_progress=True):
                    if status == "completed" and html:
                        paragraphs = clean_html(
                            html, max_link_density=0.15, max_link_ratio=0.3)
                        sections = separate_by_headers(paragraphs)
                        for section in tqdm(sections, desc=f"Chunking sections for {url}", leave=False):
                            # Convert section to markdown format for chunk_headers_by_hierarchy
                            markdown_text = (f"# {section['header']}\n" + "\n".join(
                                section["content"]) if section["header"] else "\n".join(section["content"]))
                            chunks = chunk_headers_by_hierarchy(
                                markdown_text,
                                chunk_size=200,
                            )
                            for chunk in chunks:
                                chunk["chunk_id"] = generate_unique_id()
                                chunk["url"] = url
                                chunk["xpath"] = section["xpath"]
                                text_key = chunk["content"].strip().replace(
                                    "\n", " ").replace("\r", " ")
                                text_key = re.sub(r"\s+", " ", text_key)
                                if text_key in seen_texts:
                                    continue
                                seen_texts.add(text_key)
                                chunked_documents.append(chunk)
                        break
            except Exception as e:
                if attempt == max_retries - 1:
                    continue
    if not chunked_documents:
        return None, [], model
    for doc in chunked_documents:
        readability = analyze_readability(doc["content"])
        doc["mtld"] = readability["mtld"]
        doc["mtld_category"] = readability["mtld_category"]
    save_file(chunked_documents, f"{OUTPUT_DIR}/original_docs.json")
    texts = [chunk["content"] for chunk in chunked_documents]
    generated_embeddings = generate_embeddings(
        texts, model_name, show_progress=True)

    embeddings = []
    for i, (doc, embedding) in enumerate(zip(chunked_documents, generated_embeddings)):
        doc["embedding"] = embedding
        doc["chunk_index"] = i  # Add chunk index for tracking
        embeddings.append(doc)

    embedding_matrix = np.array(generated_embeddings).astype('float32')
    index = faiss.IndexFlatIP(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    merged_chunks, merge_info = merge_similar_docs(embeddings)
    merged_embedding_matrix = np.array(
        [chunk["embedding"] for chunk in merged_chunks]).astype('float32')
    merged_index = faiss.IndexFlatIP(merged_embedding_matrix.shape[1])
    merged_index.add(merged_embedding_matrix)

    return merged_index, merged_chunks, model


def query_rag(index, embeddings: List[Dict], model, query: str, k: int = 10, score_threshold: float = 1.0, cross_encoder_model: str = 'cross-encoder/ms-marco-MiniLM-L-12-v2') -> List[Dict]:
    cross_encoder = CrossEncoder(cross_encoder_model)
    query_embedding = model.encode(query, convert_to_tensor=False,
                                   show_progress_bar=False, normalize_embeddings=True).astype('float32')
    D, I = index.search(np.array([query_embedding]), k)
    results = []
    seen_chunk_ids = set()
    pairs = [[query, embeddings[idx]["content"]] for idx in I[0]]
    cross_scores = cross_encoder.predict(pairs)
    for idx, cross_score, distance in zip(I[0], cross_scores, D[0]):
        chunk_id = embeddings[idx]["chunk_id"]
        if cross_score >= score_threshold and chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk_id)
            embeddings[idx]["score"] = float(cross_score)
            results.append({
                "merged_chunk_id": chunk_id,
                "chunk_index": embeddings[idx]["chunk_index"],
                "header": embeddings[idx]["header"],
                "text": embeddings[idx]["content"],
                "url": embeddings[idx]["url"],
                "score": float(cross_score),
                "mtld": embeddings[idx]["mtld"],
                "mtld_category": embeddings[idx]["mtld_category"],
                "token_count": embeddings[idx]["num_tokens"],
                "parent_header": embeddings[idx].get("parent_header", None),
                "level": embeddings[idx].get("level", None),
                "doc_index": embeddings[idx].get("doc_index", None)
            })
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results


async def main():
    query = "Top isekai anime 2025."
    use_cache = True
    search_results = search_data(query, use_cache=use_cache)
    save_file({"query": query, "count": len(search_results),
              "results": search_results}, f"{OUTPUT_DIR}/search_results.json")
    urls = [result["url"] for result in search_results][:2]
    index, embeddings, model = await prepare_for_rag(urls, batch_size=32, max_retries=3)
    if index is None or not embeddings:
        print("No data indexed, exiting.")
        return
    results = query_rag(index, embeddings, model, query,
                        k=20, score_threshold=1.0)
    print("\nQuery Results:")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Header: {result['header'] or 'No header'}")
        print(f"Text: {result['text'][:200]}...")
        print(f"URL: {result['url']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Token Count: {result['token_count']}")
        print(f"Parent Header: {result['parent_header'] or 'None'}")
        print(f"Level: {result['level'] or 'None'}")
    save_file({"query": query, "count": len(results),
              "results": results}, f"{OUTPUT_DIR}/rag_results.json")

if __name__ == "__main__":
    asyncio.run(main())
