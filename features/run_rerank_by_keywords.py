import os
import json
import re
from typing import List
from jet.file.utils import load_file, save_file
from jet.models.embeddings.chunking import ChunkResult
from jet.models.model_types import EmbedModelType
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from jet.wordnet.keywords.helpers import extract_query_candidates
from jet.wordnet.keywords.keyword_extraction import rerank_by_keywords
from jet.wordnet.keywords.utils import preprocess_text

output_dir = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

if __name__ == '__main__':
    query_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_2/top_isekai_anime_2025/query.md"
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/vectors/semantic_search/generated/run_semantic_search/chunks_with_scores.json"

    docs: List[ChunkResult] = load_file(docs_file)["results"]
    query = "React web"

    embed_model: EmbedModelType = "all-MiniLM-L6-v2"

    texts = [f"{doc['header']}\n{doc['content']}" for doc in docs]

    ids = [d["id"] for d in docs]
    id_to_result = {r["id"]: r for r in docs}

    candidates = extract_query_candidates(query)
    # candidates = extract_query_candidates(
    #     preprocess_text(query) + "\n" + "\n".join([d["header"].lstrip('#') for d in docs]))
    reranked_results = rerank_by_keywords(
        texts=texts,
        embed_model=embed_model,
        ids=ids,
        top_n=10,
        # candidates=candidates,
        seed_keywords=candidates,
        min_count=1,
        use_mmr=True,
        diversity=0.7,
    )

    results = []
    # Map reranked results back to original doc dicts by id
    for result in reranked_results:
        doc_id = result["id"]
        doc = id_to_result.get(doc_id, {})
        mapped_result = {
            "rerank_rank": result.get("rank"),
            "rerank_score": result.get("score"),
            **doc,
            "keywords": result["keywords"],
        }
        results.append(mapped_result)

    results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

    for r in results[:10]:
        print("========================================")
        print(f"Rerank Rank   : {r.get('rerank_rank')}")
        print(f"Rerank Score  : {r.get('rerank_score'):.4f}")
        print(f"Header        : {r.get('header')}")
        # Print keywords compactly, showing text and score if available
        keywords = r.get('keywords', [])
        if keywords and isinstance(keywords[0], dict):
            kw_str = ", ".join(
                f"{k['text']} ({k['score']:.4f})" if 'score' in k else k['text'] for k in keywords)
        else:
            kw_str = ", ".join(str(k) for k in keywords)
        print(f"Keywords      : {kw_str}")
        print("Text Preview  :")
        print(json.dumps(r.get('content', '')[:100], ensure_ascii=False))
        print("========================================\n")

    logger.gray("\nSummary:")
    for r in results[:25]:
        # Use keywords from the current result
        keywords = r.get('keywords', [])
        kw_str = ", ".join(
            f"{k['text']} ({k['score']:.4f})" if 'score' in k else k['text'] for k in keywords)
        logger.log(
            f"{r.get('rerank_rank')}: ",
            f"{r.get('rerank_score'):.4f}",
            f" {r['header']}",
            f" | {kw_str}",
            colors=["ORANGE", "SUCCESS", "DEBUG", "GRAY"]
        )

    save_file({
        "query": query,
        "candidates": candidates,
        "model": embed_model,
        "count": len(results),
        "results": results
    }, f"{output_dir}/results.json")
