from collections import defaultdict
import os
import shutil
from typing import Optional, TypedDict, List
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
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    data: list[JobData] = load_file(data_file)
    data = data[:30]

    llm_model: LLMModelType = "qwen3-1.7b-4bit-dwq-053125"
    embed_model: EmbedModelType = "mxbai-embed-large"
    dimensions = 512
    chunk_size = 300
    query = "React.js web with Python AI development"
    top_k = None
    system = None
    batch_size = 32

    doc_ids = [d["id"] for d in data]
    texts = []
    for item in data:
        if not item or not item.get("title") or not item.get("details"):
            continue
        text_parts = [
            f"## Job Title\n{item['title']}",
            f"## Details\n{item['details']}",
            f"## Company\n{item['company']}",
        ]
        if item.get("entities"):
            for key in item["entities"]:
                values = item["entities"][key]
                if values:
                    text_parts.append(
                        f"## {key.replace('_', ' ').title()}\n" +
                        "\n".join([f"- {value}" for value in values])
                    )
        if item.get("keywords"):
            text_parts.append(
                f"## Keywords\n" +
                "\n".join([f"- {keyword}" for keyword in item["keywords"]])
            )
        if item.get("tags"):
            text_parts.append(
                f"## Tags\n" + "\n".join([f"- {tag}" for tag in item["tags"]])
            )
        if item.get("domain"):
            text_parts.append(f"## Domain\n- {item['domain']}")
        if item.get("salary"):
            text_parts.append(f"## Salary\n- {item['salary']}")
        if item.get("job_type"):
            text_parts.append(f"## Job Type\n- {item['job_type']}")
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
    save_file(docs, f"{OUTPUT_DIR}/docs.json")

    tokenizer = get_tokenizer_fn(embed_model)
    chunks = chunk_docs_by_hierarchy(
        texts, chunk_size, tokenizer, ids=doc_ids)
    save_file(chunks, f"{OUTPUT_DIR}/chunks.json")

    texts_to_search = [
        "\n".join([
            chunk["header"],
            chunk["content"]
        ]).strip()
        for chunk in chunks
    ]
    chunk_ids = [chunk["id"] for chunk in chunks]
    chunk_metadatas = [chunk["metadata"] for chunk in chunks]
    query_candidates = extract_query_candidates(query)
    search_results = vector_search(
        query_candidates, texts_to_search, embed_model, top_k=top_k, ids=chunk_ids, metadatas=chunk_metadatas, batch_size=batch_size, truncate_dim=dimensions)
    save_file({
        "query": query,
        "candidates": query_candidates,
        "count": len(search_results),
        "results": search_results
    }, f"{OUTPUT_DIR}/search_results.json")

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
    }, f"{OUTPUT_DIR}/chunks_with_scores.json")

    doc_scores = defaultdict(list)
    data_dict = {d["id"]: d for d in data}
    for chunk in mapped_chunks_with_scores:
        doc_id = chunk.get("doc_id")
        if doc_id is not None:
            doc_scores[doc_id].append(chunk)
            logger.debug(
                f"Added chunk {chunk['id']} to doc {doc_id}, query_scores={chunk['metadata']['query_scores']}")

    mapped_docs_with_scores = []
    for doc_id, chunks in doc_scores.items():
        # Aggregate max score for each query term across chunks
        query_max_scores = defaultdict(float)
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
        # Calculate balance score: mean of scores minus standard deviation
        scores = list(query_max_scores.values())
        if not scores:
            logger.warning(f"No scores available for doc {doc_id}")
            final_score = 0
        else:
            mean_score = np.mean(scores)
            std_score = np.std(scores) if len(scores) > 1 else 0
            final_score = mean_score - std_score
            logger.debug(
                f"Doc ID: {doc_id}, Query Max Scores: {dict(query_max_scores)}, Mean: {mean_score}, Std: {std_score}, Final Score: {final_score}")
        if final_score == 0:
            logger.warning(f"Final score is 0 for doc {doc_id}")
        # Use the chunk with the highest original score for rank and metadata
        best_chunk = max(chunks, key=lambda c: c["score"])
        doc = data_dict.get(doc_id, {})
        merged_metadata = {}
        doc_metadata = doc.get("metadata", {})
        chunk_metadata = best_chunk.get("metadata", {})
        merged_metadata.update(doc_metadata)
        merged_metadata.update(chunk_metadata)
        merged_metadata["query_scores"] = dict(query_max_scores)
        mapped_doc = {
            "rank": best_chunk["rank"],
            "score": final_score,
            **doc,
            "metadata": merged_metadata,
        }
        mapped_docs_with_scores.append(mapped_doc)
    mapped_docs_with_scores.sort(key=lambda d: (-d["score"], d["rank"]))
    logger.debug(
        f"Top 5 mapped docs: {[{'id': d['id'], 'score': d['score'], 'query_scores': d['metadata']['query_scores']} for d in mapped_docs_with_scores[:5]]}")
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
    high_score_results = [
        doc for doc in mapped_docs_with_scores if doc["score"] >= 0.7]
    save_file({
        "query": query,
        "count": len(high_score_results),
        "results": high_score_results
    }, f"{OUTPUT_DIR}/final_results_high_scores.json")
