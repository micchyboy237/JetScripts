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
from typing import List, Dict, Optional
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
    current_section = {"header": None, "content": [],
                       "xpath": None, "parent_header": None}
    header_tags = {"h1", "h2", "h3", "h4", "h5", "h6"}
    for paragraph in paragraphs:
        if paragraph.is_heading:
            if (current_section["content"] or current_section["header"]) and is_valid_header(current_section["header"]):
                sections.append(current_section)
            current_section = {
                "header": paragraph.text,
                "content": [],
                "xpath": paragraph.xpath,
                "parent_header": None  # Will be set during chunking
            }
        else:
            current_section["content"].append(paragraph.text)
    if (current_section["content"] or current_section["header"]) and is_valid_header(current_section["header"]):
        sections.append(current_section)
    return sections


def merge_similar_docs(embeddings: List[Dict], similarity_threshold: float = 0.7) -> tuple[List[Dict], List[Dict]]:
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
                "header": best_doc["header"],
                "url": best_doc["url"],
                "xpath": best_doc["xpath"],
                "score": best_doc.get("score", None),
                "mtld": best_doc["mtld"],
                "mtld_category": best_doc["mtld_category"],
                "parent_header": best_doc.get("parent_header", None),
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
                "level": doc.get("level", None),
                "chunk_index": doc.get("chunk_index", None),
                "doc_index": doc.get("doc_index", None)
            })

    save_file(merge_info, f"{OUTPUT_DIR}/merged_docs.json")
    return merged_docs, merge_info


async def prepare_for_rag(urls: List[str], model_name: EmbedModelType = 'all-MiniLM-L6-v2', urls_limit: Optional[int] = None, max_retries: int = 3) -> tuple:
    model = SentenceTransformerRegistry.load_model(model_name)
    documents = []
    all_links = []
    seen_texts = set()

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
                        "parent_header", None)
                    text_key = chunk["content"].strip().replace(
                        "\n", " ").replace("\r", " ")
                    text_key = re.sub(r"\s+", " ", text_key)
                    if text_key in seen_texts:
                        continue
                    seen_texts.add(text_key)
                    chunk["num_tokens"] = len(
                        word_tokenize(chunk["content"]))
                    documents.append(chunk)

    if not documents:
        logger.warning("No documents collected after scraping.")
        return None, [], model, []

    for doc in documents:
        readability = analyze_readability(doc["content"])
        doc["mtld"] = readability["mtld"]
        doc["mtld_category"] = readability["mtld_category"]

    # Update doc_index to be incrementing
    for i, doc in enumerate(documents):
        doc["doc_index"] = i

    # documents = [doc for doc in documents if doc.get(
    #     "mtld_category") != "very_low"]
    total_tokens = sum(doc["num_tokens"] for doc in documents)
    save_file({"count": len(documents), "total_tokens": total_tokens, "documents": documents},
              f"{OUTPUT_DIR}/docs.json")

    texts = [doc["content"] for doc in documents]
    generated_embeddings = generate_embeddings(
        texts, model, show_progress=True)

    embeddings = []
    for i, (doc, embedding) in enumerate(zip(documents, generated_embeddings)):
        doc["embedding"] = embedding
        doc["doc_index"] = i
        embeddings.append(doc)

    embedding_matrix = np.array(generated_embeddings).astype('float32')
    index = faiss.IndexFlatIP(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    merged_docs, merge_info = merge_similar_docs(embeddings)
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
    """
    Query the RAG system to retrieve relevant documents based on the query.

    Args:
        index: FAISS index for embeddings.
        embeddings: List of document dictionaries containing embeddings and metadata.
        model: SentenceTransformer model for encoding queries.
        merge_info: List of merge information for documents.
        query: The query string to search for.
        k: Number of top documents to retrieve (default: 10).
        threshold: Minimum score for including documents (default: 0.0, used only with reranking).
        cross_encoder_model: Cross-encoder model name for reranking (default: 'cross-encoder/ms-marco-MiniLM-L-12-v2').
        use_reranking: Whether to use cross-encoder for reranking (default: True).

    Returns:
        List of dictionaries containing retrieved documents with metadata and scores.
    """
    query_embedding = model.encode(query, convert_to_tensor=False,
                                   show_progress_bar=False, normalize_embeddings=True).astype('float32')
    D, I = index.search(np.array([query_embedding]), k)
    results = []
    seen_doc_ids = set()

    if use_reranking:
        cross_encoder = CrossEncoder(cross_encoder_model)
        pairs = [[query, embeddings[idx]["content"]] for idx in I[0]]
        cross_scores = cross_encoder.predict(pairs)
        scores = cross_scores
    else:
        # Use normalized cosine similarity scores (D is already cosine similarity from FAISS)
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
            "chunk_index": embeddings[idx]["chunk_index"],
            "header": embeddings[idx]["header"],
            "text": embeddings[idx]["content"],
            "url": embeddings[idx]["url"],
            "score": float(score),
            "mtld": embeddings[idx]["mtld"],
            "mtld_category": embeddings[idx]["mtld_category"],
            "num_tokens": embeddings[idx]["num_tokens"],
            "parent_header": embeddings[idx].get("parent_header", None),
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
    """
    Groups RAG query results by URL and formats them into a string for LLM context.
    Organizes documents hierarchically by parent_header and header, sorts by score,
    and filters to respect max_tokens, including headers and separators, using the specified LLM's tokenizer.
    Uses the 'level' field to determine the number of hashtags for headers.
    Ensures the last document is included if it fits within max_tokens with a buffer.

    Args:
        documents: List of dictionaries containing RAG query results with 'text', 'url', 'num_tokens',
                   'doc_index', 'chunk_index', 'header', 'parent_header', 'score', and 'level'.
        llm_model: The LLM model type to use for tokenization.
        max_tokens: Maximum number of tokens allowed in the output context (default: 2000).
        buffer: Number of tokens to reserve as a safety buffer (default: 100).

    Returns:
        A formatted string with grouped content by URL, respecting token limit and hierarchy.
    """
    tokenizer = get_tokenizer_fn(
        llm_model, add_special_tokens=False, remove_pad_tokens=True)
    separator = "\n\n"
    separator_tokens = len(tokenizer.encode(separator))

    # Sort documents by score (descending) to prioritize high-relevance results
    sorted_docs = sorted(
        documents, key=lambda x: x.get("score", 0), reverse=True)

    # Filter documents to respect max_tokens, including headers, separators, and buffer
    filtered_docs = []
    total_tokens = 0
    grouped_temp = defaultdict(lambda: defaultdict(list))

    for doc in sorted_docs:
        text = doc.get("text", "")
        url = doc.get("url", "Unknown Source")
        parent_header = doc.get("parent_header", "None")
        header = doc.get("header", None)
        level = doc.get("level", 0)
        doc_tokens = doc.get("num_tokens", len(tokenizer.encode(text)))
        header_tokens = 0

        # Calculate header tokens for this document
        url_header = f"<!-- Source: {url} -->\n\n"
        parent_header_text = f"# {parent_header}\n\n" if parent_header and parent_header != "None" and level > 0 else ""
        subheader_text = f"## {header}\n" if header and header != parent_header and level >= 0 else ""

        # Only add URL header and separator for new URLs
        if not grouped_temp[url]:
            header_tokens += len(tokenizer.encode(url_header))
            header_tokens += separator_tokens if filtered_docs else 0  # Separator before new URL
        # Only add parent header for new parent_header groups
        if not grouped_temp[url][parent_header] and parent_header and parent_header != "None" and level > 0:
            header_tokens += len(tokenizer.encode(parent_header_text))
        # Add subheader if applicable
        if header and header != parent_header and level >= 0:
            header_tokens += len(tokenizer.encode(subheader_text))

        additional_tokens = doc_tokens + header_tokens + separator_tokens

        # Check if adding this document exceeds max_tokens with buffer
        if total_tokens + additional_tokens <= max_tokens - buffer:
            filtered_docs.append(doc)
            grouped_temp[url][parent_header].append(doc)
            total_tokens += additional_tokens
        else:
            logger.debug(
                f"Skipping document (score: {doc.get('score', 0)}): would exceed max_tokens ({total_tokens + additional_tokens} > {max_tokens - buffer})")

    # Build context by grouping filtered documents
    context_blocks = []
    total_tokens = 0
    for url, parent_groups in grouped_temp.items():
        block = f"<!-- Source: {url} -->\n\n"
        block_tokens = len(tokenizer.encode(block))

        for parent_header, parent_docs in parent_groups.items():
            # Sort by level (ascending) and then chunk_index
            parent_docs = sorted(parent_docs, key=lambda x: (
                x.get("level", 0), x.get("chunk_index", 0)))

            # Add parent header if it exists and has a value
            level = parent_docs[0].get("level", 0) if parent_docs else 0
            if parent_header and parent_header != "None" and level > 0:
                parent_header_text = f"# {parent_header}\n\n"
                block += parent_header_text
                block_tokens += len(tokenizer.encode(parent_header_text))

            for doc in parent_docs:
                header = doc.get("header", None)
                text = doc.get("text", "")
                doc_tokens = doc.get("num_tokens", len(tokenizer.encode(text)))
                doc_level = doc.get("level", 0)

                # Add header if it exists, has a value, and is not the same as parent_header
                if header and header != parent_header and doc_level >= 0:
                    subheader_text = f"## {header}\n"
                    block += subheader_text
                    block_tokens += len(tokenizer.encode(subheader_text))

                block += text + "\n\n"
                block_tokens += doc_tokens + separator_tokens

        if block_tokens > len(tokenizer.encode(f"<!-- Source: {url} -->\n\n")):
            context_blocks.append(block.strip())
            total_tokens += block_tokens
        else:
            logger.warning(
                f"Empty block for {url} after processing; skipping.")

    result = separator.join(context_blocks)
    final_token_count = len(tokenizer.encode(result))
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
    # Positional query (optional)
    p.add_argument("query_pos", type=str, nargs="?",
                   help="Search query as positional argument")
    # Optional query flag
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
    index, embeddings, model, merge_info = await prepare_for_rag(urls, urls_limit=urls_limit, max_retries=3)
    if index is None or not embeddings:
        print("No data indexed, exiting.")
        return

    # Query RAG without reranking
    results_no_reranking = query_rag(index, embeddings, model, merge_info, query,
                                     k=20, threshold=0.0, use_reranking=False)
    total_tokens_no_reranking = sum(
        result["num_tokens"] for result in results_no_reranking)
    save_file({"query": query, "count": len(results_no_reranking), "total_tokens": total_tokens_no_reranking,
               "results": results_no_reranking}, f"{OUTPUT_DIR}/results_no_reranking.json")

    # Query RAG with reranking
    results_with_reranking = query_rag(index, embeddings, model, merge_info, query,
                                       k=20, threshold=0.0, use_reranking=True)
    total_tokens_with_reranking = sum(
        result["num_tokens"] for result in results_with_reranking)
    save_file({"query": query, "count": len(results_with_reranking), "total_tokens": total_tokens_with_reranking,
               "results": results_with_reranking}, f"{OUTPUT_DIR}/results_with_reranking.json")

    results = results_no_reranking

    print("\nQuery Results (With Reranking):")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Header: {result['header'] or 'No header'}")
        print(f"Text: {result['text'][:200]}...")
        print(f"URL: {result['url']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Token Count: {result['num_tokens']}")
        print(f"Parent Header: {result['parent_header'] or 'None'}")
        print(f"Level: {result['level'] or 'None'}")
        print(f"Selected Doc IDs: {result['selected_doc_ids']}")

    llm_model: LLMModelType = "qwen3-1.7b-4bit-dwq-053125"
    context = group_results_by_url_for_llm_context(
        results, llm_model)
    save_file(context, f"{OUTPUT_DIR}/context.md")
    save_file({"num_tokens": count_tokens(llm_model, context)},
              f"{OUTPUT_DIR}/context_info.json")

    mlx = MLXModelRegistry.load_model(llm_model)
    prompt = PROMPT_TEMPLATE.format(query=query, context=context)
    llm_response = mlx.chat(prompt, llm_model, temperature=0.7, verbose=True)

    save_file(llm_response["content"], f"{OUTPUT_DIR}/response.md")

if __name__ == "__main__":
    asyncio.run(main())
