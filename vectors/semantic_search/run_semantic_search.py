from collections import defaultdict
import os
import shutil
from typing import Optional, TypedDict, List
import numpy as np
from jet.vectors.semantic_search.base import vector_search
from jet.file.utils import load_file, save_file
from jet.models.embeddings.base import generate_embeddings
from jet.models.embeddings.chunking import ChunkMetadata, DocChunkResult, chunk_docs_by_hierarchy
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.models.tokenizer.base import get_tokenizer_fn
from jet.wordnet.keywords.helpers import Keyword, SimilarityResult, extract_query_candidates
from jet.wordnet.keywords.keyword_extraction import rerank_by_keywords
from shared.data_types.job import JobData

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


class RerankedChunk(DocChunkResult):
    rank: int
    score: float
    keywords: List[Keyword]


# Use with type checking (mypy will validate the structure)


def rerank_chunks(chunks: List[DocChunkResult], query: str, embed_model: EmbedModelType = "all-MiniLM-L6-v2") -> List[RerankedChunk]:
    texts = [f"{doc['header']}\n{doc['content']}" for doc in chunks]

    ids = [d["id"] for d in chunks]
    id_to_result = {r["id"]: r for r in chunks}

    seed_keywords = extract_query_candidates(query)
    reranked_results = rerank_by_keywords(
        texts=texts,
        embed_model=embed_model,
        ids=ids,
        top_n=10,
        # candidates=candidates,
        seed_keywords=seed_keywords,
        min_count=1,
        # use_mmr=True,
        # diversity=0.7,
        threshold=0.7,
    )

    results: List[RerankedChunk] = []
    # Map reranked results back to original doc dicts by id
    for result in reranked_results:
        doc_id = result["id"]
        doc = id_to_result[doc_id]
        mapped_result: RerankedChunk = {
            "rank": result.get("rank"),
            "score": result.get("score"),
            **doc,
            "keywords": result["keywords"],
        }
        results.append(mapped_result)

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results


if __name__ == "__main__":
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"

    data: list[JobData] = load_file(data_file)
    embed_model: EmbedModelType = "all-MiniLM-L6-v2"
    llm_model: LLMModelType = "qwen3-1.7b-4bit-dwq-053125"
    chunk_size = 300
    query = "React web"
    top_k = None
    system = None
    batch_size = 32

    doc_ids = [d["id"] for d in data]
    # texts = [
    #     "\n\n".join([
    #         f"## Job Title\n{item['title']}",
    #         f"## Details\n{item['details']}",
    #         *[
    #             f"## {key.replace('_', ' ').title()}\n" +
    #             "\n".join([f"- {value}" for value in item["entities"][key]])
    #             for key in item["entities"]
    #         ],
    #         f"## Tags\n" + "\n".join([f"- {tag}" for tag in item["tags"]]),
    #     ])
    #     for item in data
    # ]
    texts = []
    for item in data:
        if not item or not item.get("title") or not item.get("details"):
            continue  # Skip if item is empty or missing required fields
        text_parts = [
            f"# Job Title\n{item['title']}",
            # f"## Details\n{item['details']}",
            # f"## Company\n{item['company']}",
        ]

        # text_parts.append("## Metadata")
        # # Keywords
        # if item.get("keywords"):
        #     text_parts.append(
        #         f"Keywords:\n" +
        #         "\n".join([f"- {keyword}" for keyword in item["keywords"]])
        #     )
        # # Tags
        # if item.get("tags"):
        #     text_parts.append(
        #         f"Tags:\n" + "\n".join([f"- {tag}" for tag in item["tags"]])
        #     )

        # # Entities
        # if item.get("entities"):
        #     for key in item["entities"]:
        #         values = item["entities"][key]
        #         if values:
        #             text_parts.append(
        #                 f"{key.replace('_', ' ').title()}:\n" +
        #                 "\n".join([f"- {value}" for value in values])
        #             )

        # # Domain
        # if item.get("domain"):
        #     text_parts.append(f"## Domain\n- {item['domain']}")
        # # Salary
        # if item.get("salary"):
        #     text_parts.append(f"## Salary\n- {item['salary']}")
        # # Job Type
        # if item.get("job_type"):
        #     text_parts.append(f"## Job Type\n- {item['job_type']}")
        # # Hours per Week
        # if item.get("hours_per_week"):
        #     text_parts.append(f"## Hours per Week\n- {item['hours_per_week']}")
        texts.append("\n\n".join(text_parts))
    save_file(texts, f"{OUTPUT_DIR}/docs.json")

    tokenizer = get_tokenizer_fn(embed_model)
    chunks = chunk_docs_by_hierarchy(
        texts, chunk_size, tokenizer, ids=doc_ids)
    save_file(chunks, f"{OUTPUT_DIR}/chunks.json")

    reranked_chunks = rerank_chunks(chunks, query, embed_model)
    save_file({
        "query": query,
        "count": len(reranked_chunks),
        "results": reranked_chunks
    }, f"{OUTPUT_DIR}/reranked_chunks.json")

    # Map mapped_chunks_with_scores to data by doc_id to id with rank and score
    # Use highest chunk score for each doc
    doc_scores = defaultdict(list)
    data_dict = {d["id"]: d for d in data}
    for chunk in reranked_chunks:
        doc_id = chunk.get("doc_id")
        if doc_id is not None:
            doc_scores[doc_id].append(chunk)

    mapped_docs_with_scores = []
    for doc_id, chunks in doc_scores.items():
        # Find the chunk with the highest score for this doc
        best_chunk = max(chunks, key=lambda c: c["score"])
        doc = data_dict.get(doc_id, {})
        # Merge chunk metadata with doc metadata (chunk metadata takes precedence if keys overlap)
        merged_metadata = {}
        doc_metadata = doc.get("metadata", {})
        chunk_metadata = best_chunk.get("metadata", {})
        merged_metadata.update(doc_metadata)
        merged_metadata.update(chunk_metadata)
        mapped_doc = {
            "rank": best_chunk["rank"],
            "score": best_chunk["score"],
            **doc,
            "metadata": merged_metadata,
        }
        mapped_docs_with_scores.append(mapped_doc)

    # Sort docs by score descending, then by rank ascending
    mapped_docs_with_scores.sort(key=lambda d: (-d["score"], d["rank"]))

    save_file({
        "query": query,
        "count": len(mapped_docs_with_scores),
        "results": mapped_docs_with_scores
    }, f"{OUTPUT_DIR}/final_results.json")

    save_file({
        "query": query,
        "results": mapped_docs_with_scores[:10]
    }, f"{OUTPUT_DIR}/final_results_top_10.json")

    save_file({
        "query": query,
        "results": mapped_docs_with_scores[:20]
    }, f"{OUTPUT_DIR}/final_results_top_20.json")
