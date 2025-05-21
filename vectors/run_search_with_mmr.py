import json
import os
from typing import List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from jet.file.utils import load_file, save_file
from jet.vectors.search_with_mmr import search_diverse_context, Header
import time
from jet.logger import logger

if __name__ == "__main__":
    start_time = time.time()
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/headers.json"
    embed_model = "all-mpnet-base-v2"
    rerank_model = "cross-encoder/ms-marco-MiniLM-L-12-v2"

    logger.info(f"Loading headers from {docs_file}")
    headers: List[Header] = load_file(docs_file)
    logger.info(f"Loaded {len(headers)} headers")
    if headers:
        logger.debug(
            f"Sample header: {json.dumps(headers[0]['header'][:50])}..., content: {json.dumps(headers[0]['content'][:50])}...")

    query = "List trending isekai reincarnation anime this year.\nList trending isekai reincarnation anime this year.\nWhat are some popular isekai reincarnation anime that have gained significant attention in recent months?\nAre there any notable isekai reincarnation anime this year that have been widely praised or recommended?"
    logger.info(f"Executing query (length: {len(query.split())} words)")

    results = search_diverse_context(
        query=query,
        headers=headers,
        model_name=embed_model,
        rerank_model=rerank_model,
        top_k=20,
        num_results=10,
        lambda_param=0.5
    )

    logger.info(f"Search returned {len(results)} results")
    for i, result in enumerate(results):
        logger.info(f"Result {i+1}:")
        logger.info(f"Header: {json.dumps(result['header'])}")
        logger.info(f"Content: {json.dumps(result['content'])}")
        logger.info(f"Header Level: {result['header_level']}")
        logger.info(f"Parent Header: {result['parent_header'] or 'None'}")
        logger.info(f"Tokens: {result['tokens']}")
        logger.success(f"Embedding Score: {result['score']:.4f}")
        logger.success(f"Rerank Score: {result['rerank_score']:.4f}")
        logger.success(f"Diversity Score: {result['diversity_score']:.4f}")
        logger.gray("-" * 50)

    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    results_without_embeddings = [
        {k: v for k, v in result.items() if k != 'embedding'} for result in results]
    output_data = {
        "query": query,
        "results": results_without_embeddings,
        "embed_model": embed_model,
        "rerank_model": rerank_model,
        "execution_time_seconds": time.time() - start_time
    }
    logger.info(f"Saving results to {output_dir}/search_with_mmr_results.json")
    save_file(output_data, f"{output_dir}/search_with_mmr_results.json")
    logger.info(f"Script completed in {time.time() - start_time:.2f} seconds")
