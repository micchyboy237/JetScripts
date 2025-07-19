from collections import defaultdict
import os
import shutil
from typing import Dict, Optional, TypedDict, List
import numpy as np
from jet.logger import logger
from jet.vectors.semantic_search.base import vector_search
from jet.vectors.semantic_search.aggregation import aggregate_doc_scores
from jet.vectors.semantic_search.search_types import Match
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


if __name__ == "__main__":
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    data: list[JobData] = load_file(data_file)
    embed_model: EmbedModelType = "all-MiniLM-L6-v2"
    llm_model: LLMModelType = "qwen3-1.7b-4bit-dwq-053125"
    chunk_size = 300
    # query = "React Native"
    query = "React.js web with Python AI development"
    top_k = None
    system = None
    batch_size = 32

    sub_dir = f"{OUTPUT_DIR}/{format_sub_dir(query)}"
    shutil.rmtree(sub_dir, ignore_errors=True)

    doc_ids = [d["id"] for d in data]
    texts = []
    for item in data:
        if not item or not item.get("title") or not item.get("details"):
            logger.warning(
                f"Skipping item with id {item.get('id', 'unknown')} due to missing title or details")
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
    } for text, num_tokens, doc in zip(texts, texts_token_counts, data) if text]
    save_file(docs, f"{sub_dir}/docs.json")

    # Validate data_dict entries
    data_dict = {d["id"]: d for d in docs}
    for doc_id in doc_ids:
        if doc_id not in data_dict:
            logger.warning(
                f"Document {doc_id} not found in data_dict after processing")
    logger.debug(f"Initialized data_dict with {len(data_dict)} documents")

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
        "embed_model": embed_model,
        "candidates": query_candidates,
        "count": len(search_results),
        "results": search_results
    }, f"{sub_dir}/search_results.json")

    mapped_chunks_with_scores = []
    chunk_dict = {chunk["id"]: chunk for chunk in chunks}
    for result in search_results:
        chunk = chunk_dict.get(result["id"], {})
        if not chunk:
            logger.warning(f"Chunk {result['id']} not found in chunk_dict")
            continue
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
        "embed_model": embed_model,
        "candidates": query_candidates,
        "count": len(mapped_chunks_with_scores),
        "results": mapped_chunks_with_scores
    }, f"{sub_dir}/chunks_with_scores.json")

    logger.debug(
        f"Calling aggregate_doc_scores with {len(mapped_chunks_with_scores)} chunks and {len(query_candidates)} query candidates")
    mapped_docs_with_scores = aggregate_doc_scores(
        mapped_chunks_with_scores, data_dict, query_candidates)
    logger.debug(f"Aggregated {len(mapped_docs_with_scores)} documents")

    logger.debug(
        f"Final mapped docs: {[{'id': d['id'], 'rank': d['rank'], 'score': d['score'], 'query_scores': d['metadata']['query_scores'], 'matches_count': len(d.get('matches', []))} for d in mapped_docs_with_scores[:5]]}")
    save_file({
        "query": query,
        "embed_model": embed_model,
        "candidates": query_candidates,
        "count": len(mapped_docs_with_scores),
        "results": mapped_docs_with_scores
    }, f"{sub_dir}/final_results.json")
    save_file({
        "query": query,
        "embed_model": embed_model,
        "candidates": query_candidates,
        "results": mapped_docs_with_scores[:10]
    }, f"{sub_dir}/final_results_top_10.json")
    save_file({
        "query": query,
        "embed_model": embed_model,
        "candidates": query_candidates,
        "results": mapped_docs_with_scores[:20]
    }, f"{sub_dir}/final_results_top_20.json")
    high_score_results = [
        doc for doc in mapped_docs_with_scores if doc["score"] >= 0.7]
    save_file({
        "query": query,
        "embed_model": embed_model,
        "candidates": query_candidates,
        "count": len(high_score_results),
        "results": high_score_results
    }, f"{sub_dir}/final_results_high_scores.json")
