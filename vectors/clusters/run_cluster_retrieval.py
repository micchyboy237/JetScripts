from collections import defaultdict
import math
import os
import shutil
from typing import DefaultDict, List, Set, Tuple
from jet.code.markdown_types.markdown_parsed_types import HeaderDoc
from jet.file.utils import load_file, save_file
from jet.llm.llm_generator import LLMGenerator
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.logger import logger
from jet.logger.config import colorize_log
from jet.models.tokenizer.base import count_tokens, get_tokenizer_fn
from jet.utils.text import format_sub_dir
from jet.vectors.clusters.retrieval import ChunkSearchResult, RetrievalConfigDict, VectorRetriever
from jet.wordnet.text_chunker import chunk_texts_with_data

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

PROMPT_TEMPLATE = """\
Context information is below.
---------------------
{context}
---------------------

Given the context information, answer the query.

Query: {query}
"""


def strip_hashtags(header: str) -> str:
    """Remove markdown hashtags from header."""
    return header.lstrip('#').strip()


def group_results_by_source_for_llm_context(
    results: List[ChunkSearchResult],
    llm_model: LLMModelType,
) -> str:
    tokenizer = get_tokenizer_fn(llm_model)
    separator = "\n\n"
    separator_tokens = len(tokenizer.encode(separator))

    # Sort results by score for relevance
    sorted_docs = sorted(results, key=lambda x: x["score"], reverse=True)

    # Group results by source
    grouped_temp: DefaultDict[str, List[ChunkSearchResult]] = defaultdict(list)
    for doc in sorted_docs:
        source = doc["metadata"].get("source", "Unknown Source")
        grouped_temp[source].append(doc)

    context_blocks = []
    for source, docs in grouped_temp.items():
        block = f"<!-- Source: {source} -->\n\n"
        seen_headers: Set[str] = set()

        # Group by doc_index and header for structured output
        grouped_by_header: DefaultDict[Tuple[int, str],
                                       List[ChunkSearchResult]] = defaultdict(list)
        for doc in sorted(docs, key=lambda x: (x["metadata"].get("doc_index", 0), x["metadata"].get("start_idx", 0))):
            doc_index = doc["metadata"].get("doc_index", 0)
            # Extract header from text (assuming text starts with header)
            header = doc["text"].split(
                "\n")[0] if doc["text"].startswith("#") else ""
            grouped_by_header[(doc_index, header)].append(doc)

        for (doc_index, header), chunks in sorted(grouped_by_header.items(), key=lambda x: x[0][0]):
            header_key = strip_hashtags(header) if header else None
            parent_header = chunks[0]["metadata"].get("parent_header", None)
            parent_header_key = strip_hashtags(
                parent_header) if parent_header else None

            # Include parent header only if it has relevant children and hasn't been seen
            if parent_header_key and parent_header_key not in seen_headers:
                has_relevant_child = any(
                    strip_hashtags(d["text"].split("\n")[
                                   0]) == parent_header_key
                    for d in docs if d["text"].startswith("#")
                )
                if has_relevant_child:
                    block += f"{parent_header}\n\n"
                    seen_headers.add(parent_header_key)

            # Include header only if it hasn't been seen
            if header_key and header_key not in seen_headers:
                block += f"{header}\n\n"
                seen_headers.add(header_key)

            # Merge chunks by sorting and combining overlapping content
            chunks.sort(key=lambda x: x["metadata"].get("start_idx", 0))
            merged_content = ""
            if chunks:
                current_chunk = chunks[0]
                # Remove header from content if present
                content = current_chunk["text"].split(
                    "\n", 1)[1] if current_chunk["text"].startswith("#") else current_chunk["text"]
                merged_content = content.strip()
                end_idx = current_chunk["metadata"].get("end_idx", 0)

                for next_chunk in chunks[1:]:
                    next_start = next_chunk["metadata"].get("start_idx", 0)
                    next_content = next_chunk["text"].split(
                        "\n", 1)[1] if next_chunk["text"].startswith("#") else next_chunk["text"]
                    next_content = next_content.strip()
                    next_end = next_chunk["metadata"].get("end_idx", 0)

                    if next_start <= end_idx + 1:
                        # Merge overlapping or contiguous chunks
                        overlap = end_idx - next_start + 1 if next_start <= end_idx else 0
                        additional_content = next_content[overlap:
                                                          ] if overlap > 0 else next_content
                        merged_content += additional_content
                        end_idx = max(end_idx, next_end)
                    else:
                        # Non-contiguous chunk, start new segment
                        block += merged_content + "\n\n"
                        merged_content = next_content
                        end_idx = next_end

                block += merged_content + "\n\n"

        # Only include non-empty blocks
        block_tokens = len(tokenizer.encode(block))
        if block_tokens > len(tokenizer.encode(f"<!-- Source: {source} -->\n\n")) + separator_tokens:
            context_blocks.append(block.strip())
        else:
            logger.debug(
                f"Skipping empty or near-empty block for source: {source}")

    result = separator.join(context_blocks)
    final_token_count = len(tokenizer.encode(result))
    logger.debug(
        f"Grouped context created with {final_token_count} tokens for {len(grouped_temp)} sources")
    return result


def main():
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_4/top_isekai_anime_2025/docs.json"
    header_docs: List[HeaderDoc] = load_file(docs_file)["documents"]

    embed_model: EmbedModelType = "all-MiniLM-L6-v2"
    llm_model: LLMModelType = "qwen3-1.7b-4bit"

    config: RetrievalConfigDict = {
        "embed_model": embed_model,
        "min_cluster_size": 2,
        "k_clusters": 3,
        "top_k": None,
        "cluster_threshold": 20,
        "cache_file": None,
        "threshold": None,
    }
    chunk_size = 500
    chunk_overlap = 100
    max_tokens = 2000

    query = "Top isekai anime 2025."
    chunked_docs = []
    all_chunks_with_metadata = []
    for header_doc in header_docs:
        doc_index = header_doc["doc_index"]
        header = header_doc["header"]
        content = header_doc['content']
        source = header_doc.get("source", "Unknown Source")
        parent_header = header_doc.get("parent_header", None)
        buffer: int = count_tokens(embed_model, header)
        doc_ids = [header_doc["id"]]
        chunks = chunk_texts_with_data(content, chunk_size,
                                       chunk_overlap, embed_model, doc_ids, buffer)

        chunks_with_metadata = []
        # Create metadata for each chunk
        for chunk in chunks:
            header_doc_copy = header_doc.copy()
            header_doc_copy.pop("doc_index")

            header = header_doc_copy.pop("header")
            level = header_doc_copy.pop("level")
            parent_level = header_doc_copy.pop("parent_level")
            parent_header = header_doc_copy.pop("parent_header")
            parent_headers = header_doc_copy.pop("parent_headers")

            source = header_doc_copy.pop("source")
            tokens = header_doc_copy.pop("tokens")

            chunk_copy = chunk.copy()
            header_doc_copy["id"] = chunk_copy.pop("id")
            chunk_content = chunk_copy.pop("content")

            text = f"{header}\n{chunk_content}"

            merged_data = {
                **header_doc_copy,
                "text": text,
                "metadata": {
                    **chunk_copy,
                    "source": source,
                    "parent_level": parent_level,
                    "level": level,
                    "parent_headers": parent_headers,
                    "parent_header": parent_header,
                    "header": header,
                    "content": chunk_content,
                    # "tokens": tokens,
                }
            }
            chunks_with_metadata.append(merged_data)
            all_chunks_with_metadata.append(merged_data)

        chunked_docs.append({
            "doc_index": doc_index,
            "count": len(chunks_with_metadata),
            "total_tokens": sum(chunk["metadata"]["num_tokens"] for chunk in chunks_with_metadata),
            "chunks": chunks_with_metadata,
        })

    save_file({
        "query": query,
        "model": embed_model,
        "count": len(all_chunks_with_metadata),
        "total_tokens": sum(chunk["metadata"]["num_tokens"] for chunk in all_chunks_with_metadata),
        "chunks": all_chunks_with_metadata
    }, f"{OUTPUT_DIR}/all_chunks_with_metadata.json")

    token_counts = [chunk["metadata"]["num_tokens"]
                    for chunk in all_chunks_with_metadata]
    texts_with_tokens = [{
        "tokens": tokens,
        "text": chunk["text"],
    } for tokens, chunk in zip(token_counts, all_chunks_with_metadata)]
    save_file({
        "count": len(all_chunks_with_metadata),
        "max_tokens": max(token_counts),
        "ave_tokens": math.ceil(sum(token_counts) / len(token_counts)) if token_counts else 0,
        "min_tokens": min(token_counts),
        "texts": texts_with_tokens
    }, f"{OUTPUT_DIR}/texts.json")
    save_file({
        "count": len(chunked_docs),
        "chunks": chunked_docs
    }, f"{OUTPUT_DIR}/chunked_docs.json")

    all_texts = [
        chunk["text"]
        for chunk in all_chunks_with_metadata
    ]
    all_ids = [chunk["id"] for chunk in all_chunks_with_metadata]
    all_metadatas = [chunk["metadata"] for chunk in all_chunks_with_metadata]
    token_counts_all_texts: List[int] = count_tokens(
        embed_model, all_texts, prevent_total=True)
    all_texts_for_clustering = [{
        "tokens": tokens,
        "text": text,
    } for tokens, text in zip(token_counts_all_texts, all_texts)]
    save_file({
        "query": query,
        "model": llm_model,
        "count": len(all_texts_for_clustering),
        "total_tokens": sum(d["tokens"] for d in all_texts_for_clustering),
        "texts": all_texts_for_clustering
    }, f"{OUTPUT_DIR}/all_texts_for_clustering.json")

    retriever = VectorRetriever(config)
    retriever.load_or_compute_embeddings(
        all_texts, ids=all_ids, metadatas=all_metadatas)
    retriever.cluster_embeddings()
    retriever.build_index()

    # Save cluster assignments to clusters.json
    clusters = retriever.get_clusters()
    save_file({
        "count": len(clusters),
        "clusters": clusters
    }, f"{OUTPUT_DIR}/clusters.json")

    # Save centroid information and similarity scores
    centroids = retriever.get_centroids()
    save_file({
        "count": len(centroids),
        "centroids": centroids
    }, f"{OUTPUT_DIR}/centroids.json")

    search_centroid_results = retriever.search_centroids(query)
    save_file({
        "query": query,
        "count": len(search_centroid_results),
        "centroids": search_centroid_results
    }, f"{OUTPUT_DIR}/search_centroid_results.json")

    search_chunk_results = retriever.search_chunks(query)
    save_file({
        "query": query,
        "count": len(search_chunk_results),
        "results": search_chunk_results
    }, f"{OUTPUT_DIR}/search_chunk_results.json")

    logger.info("\nTop relevant chunks for RAG:")
    for i, result in enumerate(search_chunk_results, 1):
        print(f"{colorize_log(f"{i}.", "ORANGE")} | {colorize_log(f"{result['score']:.4f}", "SUCCESS")} | {
              colorize_log(f"{result['num_tokens']}", "DEBUG")} | {result['text'][:50]}")

    # Generate LLM response
    query_output_dir = f"{OUTPUT_DIR}/{format_sub_dir(query)}"

    # Filter results based on score and token count
    current_tokens = 0
    filtered_results = []
    for result in search_centroid_results:
        content = result["text"].strip()
        tokens = count_tokens(llm_model, content)
        if current_tokens + tokens > max_tokens:
            break
        filtered_results.append(result)
        current_tokens += tokens

    save_file({
        "query": query,
        "count": len(filtered_results),
        "total_tokens": current_tokens,
        "results": filtered_results
    }, f"{query_output_dir}/contexts.json")

    context = group_results_by_source_for_llm_context(
        filtered_results, llm_model)
    save_file(context, f"{query_output_dir}/context.md")
    mlx = MLXModelRegistry.load_model(llm_model)
    prompt = PROMPT_TEMPLATE.format(query=query, context=context)
    save_file(prompt, f"{query_output_dir}/prompt.md")
    llm_response = mlx.chat(prompt, llm_model, temperature=0.7, verbose=True)
    save_file(query, f"{query_output_dir}/query.md")
    save_file(context, f"{query_output_dir}/context.md")
    save_file(llm_response["content"], f"{query_output_dir}/response.md")

    save_file(f"# LLM Generation\n\n## Prompt\n\n{prompt}\n\n## Response\n\n{llm_response['content']}",
              f"{query_output_dir}/llm_generation.md")


if __name__ == "__main__":
    main()
