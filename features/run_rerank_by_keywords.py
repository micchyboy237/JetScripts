import os
import json
import re
from typing import List
from jet.file.utils import load_file, save_file
from jet.models.embeddings.chunking import DocChunkResult
from jet.models.model_types import EmbedModelType
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from jet.wordnet.keywords.helpers import Keyword, extract_query_candidates
from jet.wordnet.keywords.keyword_extraction import rerank_by_keywords

output_dir = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


class RerankedChunk(DocChunkResult):
    rank: int
    score: float
    keywords: List[Keyword]


def rerank_chunks(chunks: List[DocChunkResult], query: str, embed_model: EmbedModelType = "all-MiniLM-L6-v2") -> tuple[List[RerankedChunk], List[str]]:
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
        # threshold=0.7,
        # use_mmr=True,
        # diversity=0.7,
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

    return results, seed_keywords


if __name__ == '__main__':
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/vectors/semantic_search/generated/run_semantic_search/chunks.json"

    docs: List[DocChunkResult] = load_file(docs_file)
    query = "React web"

    embed_model: EmbedModelType = "all-MiniLM-L6-v2"

    results, seed_keywords = rerank_chunks(docs, query, embed_model)

    for r in results[:10]:
        print("========================================")
        print(f"Rerank Rank   : {r.get('rank')}")
        print(f"Rerank Score  : {r.get('score'):.4f}")
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
            f"{r.get('rank')}: ",
            f"{r.get('score'):.4f}",
            f" {r['header']}",
            f" | {kw_str}",
            colors=["ORANGE", "SUCCESS", "DEBUG", "GRAY"]
        )

    save_file({
        "query": query,
        "seed_keywords": seed_keywords,
        "model": embed_model,
        "count": len(results),
        "results": results
    }, f"{output_dir}/results.json")
