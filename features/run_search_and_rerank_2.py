import asyncio
import os
from typing import List, Dict, Optional
from jet.data.utils import generate_unique_id
from jet.file.utils import save_file
from jet.scrapers.hrequests_utils import scrape_urls
from jet.scrapers.utils import search_data
from jet.wordnet.analyzers.text_analysis import analyze_readability
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

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


def clean_html(html: str, language: str = "English", max_link_density: float = 0.2, max_link_ratio: float = 0.3) -> List:
    """
    Clean HTML using jusText, filtering out boilerplate and high link-to-text ratio content.
    Returns a list of non-boilerplate paragraphs.
    """
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
    logging.info(
        f"Cleaned {len(filtered_paragraphs)} non-boilerplate paragraphs")
    return filtered_paragraphs


def is_valid_header(header: Optional[str]) -> bool:
    """
    Filter out generic or date-based headers, allowing anime-related headers.
    """
    if not header:
        return True  # Allow sections without headers
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
    """
    Group paragraphs into sections based on headers (h1-h6).
    Returns a list of sections with header, content, and xpath.
    """
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

    logging.info(f"Separated into {len(sections)} sections")
    return sections


def chunk_with_overlap(section: Dict, max_tokens: int = 200, overlap_tokens: int = 30, dynamic_max_tokens: Optional[int] = None) -> List[Dict]:
    """
    Split section into chunks with overlap if exceeding max_tokens.
    Returns a list of chunks with metadata.
    """
    text = (section["header"] + "\n" + " ".join(section["content"])
            if section["header"] else " ".join(section["content"]))
    sentences = sent_tokenize(text)
    token_count = sum(len(word_tokenize(s)) for s in sentences)
    max_tokens = dynamic_max_tokens or max_tokens if token_count > max_tokens * 2 else max_tokens
    chunks = []
    current_chunk = []
    current_tokens = 0
    chunk_index = 0

    for sentence in sentences:
        sentence_tokens = len(word_tokenize(sentence))
        if current_tokens + sentence_tokens <= max_tokens:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        else:
            chunks.append({
                "chunk_id": generate_unique_id(),
                "chunk_index": chunk_index,
                "text": " ".join(current_chunk),
                "token_count": current_tokens,
                "header": section["header"],
                "xpath": section["xpath"]
            })
            overlap_sentences = []
            overlap_count = 0
            for s in current_chunk[::-1]:
                s_tokens = len(word_tokenize(s))
                if overlap_count + s_tokens <= overlap_tokens:
                    overlap_sentences.insert(0, s)
                    overlap_count += s_tokens
                else:
                    break
            current_chunk = overlap_sentences + [sentence]
            current_tokens = overlap_count + sentence_tokens
            chunk_index += 1

    if current_chunk:
        chunks.append({
            "chunk_id": generate_unique_id(),
            "chunk_index": chunk_index,
            "text": " ".join(current_chunk),
            "token_count": current_tokens,
            "header": section["header"],
            "xpath": section["xpath"]
        })

    logging.info(
        f"Created {len(chunks)} chunks for section: {section['header'] or 'No header'}")
    return chunks


def merge_similar_chunks(embeddings: List[Dict], similarity_threshold: float = 0.95) -> tuple[List[Dict], List[Dict]]:
    """
    Merge chunks with high similarity based on embeddings and save merge details to a separate file.
    Returns merged embeddings and merge_info.
    """
    logging.debug(
        f"Starting merge_similar_chunks with {len(embeddings)} embeddings")
    embedding_matrix = np.array([doc["embedding"] for doc in embeddings])
    logging.debug(f"Embedding matrix shape: {embedding_matrix.shape}")

    n_unique = len(np.unique(embedding_matrix, axis=0))
    n_clusters = min(max(1, len(embeddings) // 4), n_unique)
    logging.debug(
        f"Adjusted n_clusters: {n_clusters}, unique embeddings: {n_unique}")

    if n_unique < 2:
        logging.warning("Fewer than 2 unique embeddings, skipping clustering")
        return embeddings, []

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embedding_matrix)
    merged_chunks = []
    merge_info = []

    for cluster_id in np.unique(labels):
        cluster_chunks = [embeddings[i] for i in range(
            len(embeddings)) if labels[i] == cluster_id]
        logging.debug(
            f"Processing cluster {cluster_id} with {len(cluster_chunks)} chunks")

        if len(cluster_chunks) > 1:
            similarities = [cosine_similarity([cluster_chunks[0]["embedding"]], [chunk["embedding"]])[0][0]
                            for chunk in cluster_chunks[1:]]
            logging.debug(
                f"Similarities for cluster {cluster_id}: {similarities}")
            if max(similarities, default=0) > similarity_threshold:
                merged_text = " ".join([chunk["text"]
                                       for chunk in cluster_chunks])
                merged_token_count = sum(
                    chunk["token_count"] for chunk in cluster_chunks)
                merged_chunk_id = str(uuid.uuid4())
                merged_chunk = cluster_chunks[0].copy()
                # Set chunk_id to merged_chunk_id
                merged_chunk["chunk_id"] = merged_chunk_id
                merged_chunk["text"] = merged_text
                merged_chunk["token_count"] = merged_token_count
                merge_info.append({
                    "merged_chunk_id": merged_chunk_id,
                    "original_chunk_ids": [chunk["chunk_id"] for chunk in cluster_chunks],
                    "text": merged_text,
                    "token_count": merged_token_count,
                    "header": merged_chunk["header"],
                    "url": merged_chunk["url"],
                    "xpath": merged_chunk["xpath"],
                    "score": merged_chunk.get("score", None)
                })
                merged_chunks.append(merged_chunk)
                logging.debug(
                    f"Merged {len(cluster_chunks)} chunks in cluster {cluster_id} into merged_chunk_id: {merged_chunk_id}")
            else:
                merged_chunks.extend(cluster_chunks)
                for chunk in cluster_chunks:
                    merge_info.append({
                        "merged_chunk_id": chunk["chunk_id"],
                        "original_chunk_ids": [chunk["chunk_id"]],
                        "text": chunk["text"],
                        "token_count": chunk["token_count"],
                        "header": chunk["header"],
                        "url": chunk["url"],
                        "xpath": chunk["xpath"],
                        "score": chunk.get("score", None)
                    })
                logging.debug(
                    f"No merge for cluster {cluster_id}, keeping {len(cluster_chunks)} chunks")
        else:
            merged_chunks.append(cluster_chunks[0])
            merge_info.append({
                "merged_chunk_id": cluster_chunks[0]["chunk_id"],
                "original_chunk_ids": [cluster_chunks[0]["chunk_id"]],
                "text": cluster_chunks[0]["text"],
                "token_count": cluster_chunks[0]["token_count"],
                "header": cluster_chunks[0]["header"],
                "url": cluster_chunks[0]["url"],
                "xpath": cluster_chunks[0]["xpath"],
                "score": cluster_chunks[0].get("score", None)
            })
            logging.debug(
                f"Single chunk in cluster {cluster_id}, no merge needed")

    # Save merge information to a separate file
    save_file(merge_info, f"{OUTPUT_DIR}/merged_chunks.json")
    logging.info(
        f"Saved merge information for {len(merge_info)} chunks to merged_chunks.json")
    logging.info(
        f"Merged {len(embeddings) - len(merged_chunks)} similar chunks")
    return merged_chunks, merge_info


def save_metadata(embeddings: List[Dict], merge_info: List[Dict] = None, original_chunks: List[Dict] = None) -> None:
    """
    Save metadata to JSON file with error handling and backup, sorted by score in descending order with rank.
    Includes original_chunk_ids from merge_info for traceability.
    """
    # Save original chunks if provided
    if original_chunks:
        original_metadata = [
            {
                "chunk_id": doc["chunk_id"],
                "chunk_index": doc["chunk_index"],
                "url": doc["url"],
                "header": doc["header"],
                "token_count": doc.get("token_count", None),
                "text": doc["text"],
                "xpath": doc["xpath"],
                "mtld": doc["mtld"],
                "mtld_category": doc["mtld_category"],
            } for doc in original_chunks
        ]
        save_file(original_metadata, f"{OUTPUT_DIR}/original_chunks.json")
        logging.info(
            f"Saved original chunks for {len(original_metadata)} chunks to original_chunks.json")

    # If merge_info is not provided, load it from merged_chunks.json
    if merge_info is None:
        try:
            with open(f"{OUTPUT_DIR}/merged_chunks.json", "r") as f:
                merge_info = json.load(f)
        except FileNotFoundError:
            logging.warning(
                "merged_chunks.json not found, proceeding without original_chunk_ids")
            merge_info = []

    # Create a mapping from chunk_id to original_chunk_ids
    merge_info_map = {info["merged_chunk_id"]
        : info["original_chunk_ids"] for info in merge_info}

    metadata = [
        {
            "chunk_id": doc["chunk_id"],
            "chunk_index": doc["chunk_index"],
            "url": doc["url"],
            "header": doc["header"],
            "text": doc["text"],
            "xpath": doc["xpath"],
            "index": i,
            "score": doc.get("score", None),
            "token_count": doc.get("token_count", None),
            "rank": None,
            "original_chunk_ids": merge_info_map.get(doc["chunk_id"], [doc["chunk_id"]])
        } for i, doc in enumerate(embeddings)
    ]

    # Sort by score in descending order
    metadata = sorted(
        metadata,
        key=lambda x: x["score"] if x["score"] is not None else float('-inf'),
        reverse=True
    )

    # Assign ranks
    for i, doc in enumerate(metadata):
        doc["rank"] = i + 1 if doc["score"] is not None else None

    save_file(metadata, f"{OUTPUT_DIR}/metadata.json")


async def prepare_for_rag(urls: List[str], model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 32, max_retries: int = 3) -> tuple:
    logging.debug(
        f"Starting prepare_for_rag with {len(urls)} URLs, model: {model_name}, batch_size: {batch_size}")
    model = SentenceTransformer(model_name)
    chunked_documents = []
    seen_texts = set()

    for url in tqdm(urls, desc="Scraping URLs"):
        for attempt in range(max_retries):
            try:
                async for u, status, html in scrape_urls([url], show_progress=True):
                    if status == "completed" and html:
                        logging.debug(f"Successfully scraped {url}")
                        paragraphs = clean_html(
                            html, max_link_density=0.15, max_link_ratio=0.3)
                        logging.info(
                            f"Scraped {len(paragraphs)} paragraphs from {url}")
                        sections = separate_by_headers(paragraphs)
                        for section in tqdm(sections, desc=f"Chunking sections for {url}", leave=False):
                            chunks = chunk_with_overlap(
                                section, max_tokens=200, overlap_tokens=30, dynamic_max_tokens=300)
                            for chunk in chunks:
                                chunk["url"] = url
                                # Deduplication logic: use normalized text as key
                                text_key = chunk["text"].strip().replace(
                                    "\n", " ").replace("\r", " ")
                                text_key = re.sub(r"\s+", " ", text_key)
                                if text_key in seen_texts:
                                    continue
                                seen_texts.add(text_key)
                                chunked_documents.append(chunk)
                        break
                    else:
                        logging.warning(
                            f"Failed to scrape {url}: Status {status}")
            except Exception as e:
                logging.error(
                    f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt == max_retries - 1:
                    logging.error(f"Max retries reached for {url}. Skipping.")
                    continue

    logging.info(
        f"Total unique chunks created after deduplication: {len(chunked_documents)}")
    if not chunked_documents:
        logging.warning("No chunks created. Returning empty index.")
        return None, [], model, []

    # Calculate MTLD score for each embedding
    for doc in chunked_documents:
        readability = analyze_readability(doc["text"])
        doc["mtld"] = readability["mtld"]
        doc["mtld_category"] = readability["mtld_category"]

    # Filter out chunked_documents with "very_low" mtld_category
    chunked_documents = [doc for doc in chunked_documents if doc.get(
        "mtld_category") != "very_low"]

    # Save original chunks before generating embeddings
    save_metadata([], None, chunked_documents)

    embeddings = []
    texts = [chunk["text"] for chunk in chunked_documents]
    logging.debug(f"Generating embeddings for {len(texts)} chunks")
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(
            batch_texts, convert_to_tensor=False, show_progress_bar=False, normalize_embeddings=True)
        for j, embedding in enumerate(batch_embeddings):
            chunked_documents[i + j]["embedding"] = embedding
            embeddings.append(chunked_documents[i + j])

    if embeddings:
        logging.debug(
            f"Merging similar chunks for {len(embeddings)} embeddings")
        embeddings, merge_info = merge_similar_chunks(
            embeddings, similarity_threshold=0.95)

        # Validate chunk_id consistency after merging
        original_chunk_ids = {chunk["chunk_id"] for chunk in chunked_documents}
        for info in merge_info:
            for orig_id in info["original_chunk_ids"]:
                if orig_id not in original_chunk_ids:
                    logging.error(
                        f"support@jet.com Original chunk ID {orig_id} in merge_info not found in original_chunks")
                    raise ValueError(
                        f"Original chunk ID {orig_id} in merge_info not found in original_chunks")
    else:
        logging.warning("No embeddings generated. Returning empty index.")
        return None, [], model, []

    embedding_matrix = np.array([doc["embedding"]
                                for doc in embeddings]).astype('float32')
    logging.debug(f"Embedding matrix shape: {embedding_matrix.shape}")
    logging.debug("Using IndexFlatIP due to small dataset size")
    index = faiss.IndexFlatIP(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    logging.info(f"Indexed {len(embeddings)} chunks in FAISS")
    return index, embeddings, model, merge_info


def query_rag(index, embeddings: List[Dict], model, query: str, k: int = 10, score_threshold: float = 1.0, cross_encoder_model: str = 'cross-encoder/ms-marco-MiniLM-L-12-v2') -> List[Dict]:
    """
    Query the RAG system and return top-k results sorted by cross-encoder score in descending order.
    Uses merged_chunk_id for consistency with merged chunks and deduplicates results by merged_chunk_id.
    """
    cross_encoder = CrossEncoder(cross_encoder_model)
    query_embedding = model.encode(query, convert_to_tensor=False,
                                   show_progress_bar=False, normalize_embeddings=True).astype('float32')
    D, I = index.search(np.array([query_embedding]), k)
    results = []
    seen_chunk_ids = set()

    pairs = [[query, embeddings[idx]["text"]] for idx in I[0]]
    cross_scores = cross_encoder.predict(pairs)
    logging.info(f"Cross-encoder scores: {cross_scores}")

    for idx, cross_score, distance in zip(I[0], cross_scores, D[0]):
        chunk_id = embeddings[idx]["chunk_id"]
        if cross_score >= score_threshold and chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk_id)
            embeddings[idx]["score"] = float(cross_score)
            results.append({
                "merged_chunk_id": chunk_id,
                "chunk_index": embeddings[idx]["chunk_index"],
                "header": embeddings[idx]["header"],
                "text": embeddings[idx]["text"],
                "url": embeddings[idx]["url"],
                "score": float(cross_score),
                "mtld": embeddings[idx]["mtld"],
                "mtld_category": embeddings[idx]["mtld_category"],
                "token_count": embeddings[idx]["token_count"]
            })

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    logging.info(
        f"Retrieved {len(results)} unique results above score threshold {score_threshold} after deduplication")
    return results


async def main():
    query = "Top isekai anime 2025."
    use_cache = True

    search_results = search_data(query, use_cache=use_cache)
    save_file({"query": query, "count": len(search_results),
              "results": search_results}, f"{OUTPUT_DIR}/search_results.json")

    urls = [result["url"] for result in search_results]
    index, embeddings, model, merge_info = await prepare_for_rag(urls, batch_size=32, max_retries=3)

    if index is None or not embeddings:
        print("No data indexed, exiting.")
        return

    results = query_rag(index, embeddings, model, query,
                        k=20, score_threshold=1.0)
    save_metadata(embeddings, merge_info)

    print("\nQuery Results:")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Header: {result['header'] or 'No header'}")
        print(f"Text: {result['text'][:200]}...")
        print(f"URL: {result['url']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Token Count: {result['token_count']}")

    save_file({"query": query, "count": len(results),
              "results": results}, f"{OUTPUT_DIR}/rag_results.json")


if __name__ == "__main__":
    asyncio.run(main())
