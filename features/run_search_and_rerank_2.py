import asyncio
import os
from jet.file.utils import save_file
from jet.scrapers.hrequests_utils import scrape_urls
from jet.scrapers.utils import search_data
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

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
nltk.download('punkt', quiet=True)


def clean_html(html, language="English", max_link_density=0.2, max_link_ratio=0.3):
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


def is_valid_header(header):
    """
    Filter out generic or date-based headers.
    """
    if not header:
        return True  # Allow sections without headers
    generic_keywords = {'planet', 'articles', 'tutorials', 'jobs'}
    date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b'
    if any(keyword in header.lower() for keyword in generic_keywords) or re.match(date_pattern, header):
        return False
    return True


def separate_by_headers(paragraphs):
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


def chunk_with_overlap(section, max_tokens=200, overlap_tokens=50):
    """
    Split section into chunks with overlap if exceeding max_tokens.
    Returns a list of chunks with metadata.
    """
    text = (section["header"] + "\n" + " ".join(section["content"])
            if section["header"] else " ".join(section["content"]))
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_tokens = 0
    chunk_id = 0

    for sentence in sentences:
        sentence_tokens = len(word_tokenize(sentence))
        if current_tokens + sentence_tokens <= max_tokens:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        else:
            chunks.append({
                "chunk_id": chunk_id,
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
            chunk_id += 1

    if current_chunk:
        chunks.append({
            "chunk_id": chunk_id,
            "text": " ".join(current_chunk),
            "token_count": current_tokens,
            "header": section["header"],
            "xpath": section["xpath"]
        })

    logging.info(
        f"Created {len(chunks)} chunks for section: {section['header'] or 'No header'}")
    return chunks


async def prepare_for_rag(urls, model_name='all-MiniLM-L6-v2', batch_size=32):
    """
    Prepare documents for RAG: clean, separate, chunk, embed, and index.
    Returns FAISS index, embeddings, and model.
    """
    model = SentenceTransformer(model_name)
    chunked_documents = []

    # Process URLs asynchronously with scrape_urls
    async for url, status, html in scrape_urls(urls, show_progress=True):
        if status == "completed" and html:
            try:
                # Clean HTML content
                paragraphs = clean_html(
                    html, max_link_density=0.15, max_link_ratio=0.3)

                # Separate and chunk
                sections = separate_by_headers(paragraphs)
                for section in tqdm(sections, desc=f"Chunking sections for {url}", leave=False):
                    chunks = chunk_with_overlap(
                        section, max_tokens=200, overlap_tokens=50)
                    for chunk in chunks:
                        chunk["url"] = url
                    chunked_documents.extend(chunks)
            except Exception as e:
                logging.error(f"Error processing URL {url}: {str(e)}")
                continue

    # Generate embeddings in batches
    embeddings = []
    texts = [chunk["text"] for chunk in chunked_documents]
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(
            batch_texts, convert_to_tensor=False, show_progress_bar=False, normalize_embeddings=True)
        for j, embedding in enumerate(batch_embeddings):
            chunked_documents[i + j]["embedding"] = embedding
            embeddings.append(chunked_documents[i + j])

    # Create FAISS index with cosine similarity
    if not embeddings:
        logging.warning("No embeddings generated. Returning empty index.")
        return None, [], model

    embedding_matrix = np.array([doc["embedding"]
                                for doc in embeddings]).astype('float32')
    # Inner Product for cosine similarity
    index = faiss.IndexFlatIP(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    # Save metadata
    metadata = [
        {
            "chunk_id": doc["chunk_id"],
            "url": doc["url"],
            "header": doc["header"],
            "text": doc["text"],
            "xpath": doc["xpath"],
            "index": i
        } for i, doc in enumerate(embeddings)
    ]
    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Indexed {len(embeddings)} chunks in FAISS")
    return index, embeddings, model


def query_rag(index, embeddings, model, query, k=10, cross_encoder_model='cross-encoder/ms-marco-MiniLM-L-12-v2'):
    """
    Query the RAG system and return top-k results sorted by cross-encoder score in descending order.
    """
    cross_encoder = CrossEncoder(cross_encoder_model)
    query_embedding = model.encode(query, convert_to_tensor=False,
                                   show_progress_bar=False, normalize_embeddings=True).astype('float32')
    D, I = index.search(np.array([query_embedding]), k)
    results = []

    # Prepare pairs for cross-encoder re-ranking
    pairs = [[query, embeddings[idx]["text"]] for idx in I[0]]
    cross_scores = cross_encoder.predict(pairs)

    for idx, cross_score, distance in zip(I[0], cross_scores, D[0]):
        results.append({
            "header": embeddings[idx]["header"],
            "text": embeddings[idx]["text"],
            "url": embeddings[idx]["url"],
            "score": float(cross_score)  # Use cross-encoder score
        })

    # Sort results by cross-encoder score in descending order
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results


async def main():
    """
    Main function to process URLs, prepare for RAG, and demonstrate a query.
    """
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    # Example query
    query = "Top isekai anime 2025."
    use_cache = True

    search_results = search_data(query, use_cache=use_cache)
    save_file({"query": query, "count": len(search_results),
              "results": search_results}, f"{output_dir}/search_results.json")

    # urls = ["https://animebytes.in/15-best-upcoming-isekai-anime-in-2025"]
    urls = [result["url"] for result in search_results]
    index, embeddings, model = await prepare_for_rag(urls, batch_size=32)

    if index is None or not embeddings:
        print("No data indexed, exiting.")
        return

    results = query_rag(index, embeddings, model, query, k=20)

    print("\nQuery Results:")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Header: {result['header'] or 'No header'}")
        print(f"Text: {result['text'][:200]}...")  # Truncate for display
        print(f"URL: {result['url']}")
        print(f"Score: {result['score']:.4f}")

    save_file({"query": query, "count": len(results),
              "results": results}, f"{output_dir}/rag_results.json")

if __name__ == "__main__":
    asyncio.run(main())
