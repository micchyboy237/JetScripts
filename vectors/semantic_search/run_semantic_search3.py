import os
import shutil
from typing import Optional, TypedDict, List
import numpy as np
from jet.vectors.semantic_search.base import vector_search
from jet.file.utils import load_file, save_file
from jet.models.embeddings.base import generate_embeddings
from jet.models.embeddings.chunking import chunk_docs_by_hierarchy, chunk_headers_by_hierarchy
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.models.tokenizer.base import get_tokenizer_fn
from shared.data_types.job import JobData


class SearchResultWithJobData(TypedDict):
    rank: int
    score: float
    job: JobData


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Apps/my-jobs/saved/jobs.json"
    data: List[JobData] = load_file(data_file)
    embed_model: EmbedModelType = "all-MiniLM-L12-v2"
    llm_model: LLMModelType = "qwen3-1.7b-4bit-dwq-053125"
    chunk_size = 150
    query = "React web"
    top_k = None
    system = None

    # Create texts and map each text to its JobData ID
    texts = [
        "\n\n".join([
            f"{item['title']}",
            f"{item['company']}",
            *[
                f"{key}\n" +
                "\n".join([f"- {value}" for value in item["entities"][key]])
                for key in item["entities"]
            ],
            f"Keywords\n" +
            "\n".join([f"- {keyword}" for keyword in item["keywords"]]),
            f"Tags\n" + "\n".join([f"- {tag}" for tag in item["tags"]]),
            f"Domain\n- {item['domain']}",
            f"Salary\n- {item['salary'] or 'Not specified'}",
            f"Job Type\n- {item['job_type'] or 'Not specified'}",
            f"Hours per Week\n- {item['hours_per_week'] or 'Not specified'}"
        ])
        for item in data
    ]
    text_to_id = {i: item["id"] for i, item in enumerate(data)}
    doc_ids = [d["id"] for d in data]
    save_file(texts, f"{OUTPUT_DIR}/docs.json")

    tokenizer = get_tokenizer_fn(embed_model)
    chunks = chunk_docs_by_hierarchy(texts, chunk_size, tokenizer, ids=doc_ids)
    save_file(chunks, f"{OUTPUT_DIR}/chunks.json")

    # Map chunks to their source document ID
    chunk_to_doc = [
        (chunk["doc_id"], text_to_id[chunk["doc_index"]],
         f"{chunk['header']}\n{chunk['content']}")
        for chunk in chunks
    ]

    texts_to_search = [chunk[2]
                       for chunk in chunk_to_doc]  # Content for embedding

    # Perform vector search
    search_results = vector_search(
        query, texts_to_search, embed_model, top_k=None)

    # Aggregate scores by doc_id, taking the maximum score
    doc_scores = {}
    for result, (doc_id, job_id, _) in zip(search_results, chunk_to_doc):
        if doc_id not in doc_scores or result["score"] > doc_scores[doc_id]["score"]:
            doc_scores[doc_id] = {"score": result["score"], "job_id": job_id}

    # Map back to JobData, excluding 'details'
    final_results: List[SearchResultWithJobData] = []
    job_data_by_id = {item["id"]: item for item in data}
    for rank, (doc_id, info) in enumerate(
        sorted(doc_scores.items(),
               key=lambda x: x[1]["score"], reverse=True), 1
    ):
        if top_k is not None and rank > top_k:
            break
        job_data = job_data_by_id[info["job_id"]]
        job_without_details = {
            key: value for key, value in job_data.items() if key != "details"
        }
        final_results.append({
            "rank": rank,
            "score": info["score"],
            "job": job_without_details
        })

    save_file(final_results, f"{OUTPUT_DIR}/search_results.json")
