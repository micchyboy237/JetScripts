from collections import defaultdict
import os
import shutil
from typing import Dict, Optional, TypedDict, List
import numpy as np
from jet.logger import logger
from jet.vectors.semantic_search.base import vector_search
from jet.file.utils import load_file, save_file
from jet.models.embeddings.base import generate_embeddings
from jet.models.embeddings.chunking import chunk_docs_by_hierarchy
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.models.tokenizer.base import count_tokens, get_tokenizer_fn
from jet.wordnet.keywords.helpers import extract_query_candidates
from shared.data_types.job import JobData

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


def format_sub_dir(text: str) -> str:
    return text.lower().strip('.,!?').replace(' ', '_').replace('.', '_').replace(',', '_').replace('!', '_').replace('?', '_').strip()


def aggregate_doc_scores(
    chunks: List[dict],
    data_dict: Dict[str, dict],
    query_candidates: List[str]
) -> List[dict]:
    """Aggregate chunk scores into document scores and normalize by query candidate count."""
    doc_scores = defaultdict(list)
    for chunk in chunks:
        doc_id = chunk.get("doc_id")
        if doc_id is not None:
            doc_scores[doc_id].append(chunk)
            logger.debug(
                f"Added chunk {chunk['id']} to doc {doc_id}, query_scores={chunk['metadata']['query_scores']}")

    mapped_docs_with_scores = []
    for doc_id, chunks in doc_scores.items():
        query_max_scores = defaultdict(float)
        all_matches = []
        for chunk in chunks:
            query_scores = chunk.get("metadata", {}).get("query_scores", {})
            logger.debug(
                f"Doc ID: {doc_id}, Chunk ID: {chunk.get('id')}, Query Scores: {query_scores}")
            if not query_scores:
                logger.warning(
                    f"No query_scores found for chunk {chunk.get('id')} in doc {doc_id}")
            for query_term, score in query_scores.items():
                query_max_scores[query_term] = max(
                    query_max_scores[query_term], score)
            matches = chunk.get("matches", [])
            if matches:
                all_matches.extend(matches)

        # Sort matches by start_idx, then by descending end_idx
        all_matches.sort(key=lambda m: (m["start_idx"], -m["end_idx"]))

        # Normalize final score by query candidates count
        final_score = sum(query_max_scores.values()) / \
            len(query_candidates) if query_candidates else 0
        logger.debug(
            f"Doc ID: {doc_id}, Query Max Scores: {dict(query_max_scores)}, Final Score: {final_score}")
        if final_score == 0:
            logger.warning(f"Final score is 0 for doc {doc_id}")

        best_chunk = max(chunks, key=lambda c: c["score"])
        doc = data_dict.get(doc_id, {})
        merged_metadata = {}
        doc_metadata = doc.get("metadata", {})
        chunk_metadata = best_chunk.get("metadata", {})
        merged_metadata.update(doc_metadata)
        merged_metadata.update(chunk_metadata)
        merged_metadata["query_scores"] = dict(query_max_scores)

        mapped_doc = {
            "score": final_score,
            **doc,
            "metadata": merged_metadata,
            "matches": all_matches,
        }
        mapped_docs_with_scores.append(mapped_doc)

    mapped_docs_with_scores.sort(key=lambda d: (-d["score"], d.get("id", "")))
    for rank, doc in enumerate(mapped_docs_with_scores, 1):
        reordered_doc = {
            "rank": rank,
            **{k: v for k, v in doc.items()}
        }
        mapped_docs_with_scores[rank - 1] = reordered_doc

    return mapped_docs_with_scores


if __name__ == "__main__":
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    data: list[JobData] = load_file(data_file)
    embed_model: EmbedModelType = "all-MiniLM-L6-v2"
    llm_model: LLMModelType = "qwen3-1.7b-4bit-dwq-053125"
    chunk_size = 300
    # query = "React.js web with Python AI development"
    query = "React Native"
    top_k = None
    system = None
    batch_size = 32

    sub_dir = f"{OUTPUT_DIR}/{format_sub_dir(query)}"
    shutil.rmtree(sub_dir, ignore_errors=True)

    doc_ids = [d["id"] for d in data]
    texts = []
    for item in data:
        if not item or not item.get("title") or not item.get("details"):
            continue  # Skip if item is empty or missing required fields
        text_parts = [
            f"## Job Title\n{item['title']}",
            f"## Details\n{item['details']}",
            f"## Company\n{item['company']}",
        ]
        # Entities
        if item.get("entities"):
            for key in item["entities"]:
                values = item["entities"][key]
                if values:
                    text_parts.append(
                        f"## {key.replace('_', ' ').title()}\n" +
                        "\n".join([f"- {value}" for value in values])
                    )
        # Keywords
        if item.get("keywords"):
            text_parts.append(
                f"## Keywords\n" +
                "\n".join([f"- {keyword}" for keyword in item["keywords"]])
            )
        # Tags
        if item.get("tags"):
            text_parts.append(
                f"## Tags\n" + "\n".join([f"- {tag}" for tag in item["tags"]])
            )
        # Domain
        if item.get("domain"):
            text_parts.append(f"## Domain\n- {item['domain']}")
        # Salary
        if item.get("salary"):
            text_parts.append(f"## Salary\n- {item['salary']}")
        # Job Type
        if item.get("job_type"):
            text_parts.append(f"## Job Type\n- {item['job_type']}")
        # Hours per Week
        if item.get("hours_per_week"):
            text_parts.append(f"## Hours per Week\n- {item['hours_per_week']}")
        texts.append("\n\n".join(text_parts))

    texts_token_counts: List[int] = count_tokens(
        embed_model, texts, prevent_total=True)
    docs = [{
        "id": doc["id"],
        "posted_date": doc["posted_date"],
        "link": doc["link"],
        "num_tokens": num_tokens,
        "text": text,
    } for text, num_tokens, doc in zip(texts, texts_token_counts, data)]
    save_file(docs, f"{sub_dir}/docs.json")

    tokenizer = get_tokenizer_fn(embed_model)
    chunks = chunk_docs_by_hierarchy(
        texts, chunk_size, tokenizer, ids=doc_ids)
    save_file(chunks, f"{sub_dir}/chunks.json")

    texts_to_search = [
        "\n".join([
            chunk["parent_header"] or "",
            chunk["header"],
            chunk["content"]
        ]).strip()
        for chunk in chunks
    ]
    chunk_ids = [chunk["id"] for chunk in chunks]
    chunk_metadatas = [chunk["metadata"] for chunk in chunks]
    query_candidates = extract_query_candidates(query)
    save_file(query_candidates, f"{sub_dir}/query_candidates.json")
    search_results = vector_search(
        query_candidates, texts_to_search, embed_model, top_k=top_k, ids=chunk_ids, metadatas=chunk_metadatas, batch_size=batch_size)
    save_file({
        "query": query,
        "candidates": query_candidates,
        "count": len(search_results),
        "results": search_results
    }, f"{sub_dir}/search_results.json")

    mapped_chunks_with_scores = []
    chunk_dict = {chunk["id"]: chunk for chunk in chunks}
    for result in search_results:
        chunk = chunk_dict.get(result["id"], {})
        query_scores = result.get("metadata", {}).get("query_scores", {})
        logger.debug(
            f"Search result for chunk {result['id']}: score={result['score']}, query_scores={query_scores}")
        if not query_scores:
            logger.warning(
                f"Empty query_scores in search result for chunk {result['id']}")
        mapped_chunk = {
            "rank": result["rank"],
            "score": result["score"],
            "matches": result["matches"],
            **chunk,
            "metadata": {"query_scores": query_scores, **chunk["metadata"]},
        }
        mapped_chunks_with_scores.append(mapped_chunk)
    save_file({
        "query": query,
        "count": len(mapped_chunks_with_scores),
        "results": mapped_chunks_with_scores
    }, f"{sub_dir}/chunks_with_scores.json")

    # Initialize data_dict before calling aggregate_doc_scores
    data_dict = {d["id"]: d for d in data}
    logger.debug(f"Initialized data_dict with {len(data_dict)} documents")

    # Call aggregate_doc_scores with debug logging
    logger.debug(
        f"Calling aggregate_doc_scores with {len(mapped_chunks_with_scores)} chunks and {len(query_candidates)} query candidates")
    mapped_docs_with_scores = aggregate_doc_scores(
        mapped_chunks_with_scores, data_dict, query_candidates)
    logger.debug(f"Aggregated {len(mapped_docs_with_scores)} documents")

    logger.debug(
        f"Final mapped docs: {[{'id': d['id'], 'rank': d['rank'], 'score': d['score'], 'query_scores': d['metadata']['query_scores'], 'matches_count': len(d.get('matches', []))} for d in mapped_docs_with_scores[:5]]}")
    save_file({
        "query": query,
        "count": len(mapped_docs_with_scores),
        "results": mapped_docs_with_scores
    }, f"{sub_dir}/final_results.json")
    save_file({
        "query": query,
        "results": mapped_docs_with_scores[:10]
    }, f"{sub_dir}/final_results_top_10.json")
    save_file({
        "query": query,
        "results": mapped_docs_with_scores[:20]
    }, f"{sub_dir}/final_results_top_20.json")
    high_score_results = [
        doc for doc in mapped_docs_with_scores if doc["score"] >= 0.7]
    save_file({
        "query": query,
        "count": len(high_score_results),
        "results": high_score_results
    }, f"{sub_dir}/final_results_high_scores.json")
