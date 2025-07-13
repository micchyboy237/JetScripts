import os
import json
from jet.file.utils import load_file, save_file
from jet.models.model_types import EmbedModelType
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from jet.wordnet.keywords.helpers import extract_query_candidates
from jet.wordnet.keywords.keyword_extraction import rerank_by_keywords

output_dir = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

if __name__ == '__main__':
    query_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_2/top_isekai_anime_2025/query.md"
    contexts_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_2/top_isekai_anime_2025/contexts_before_max_filter.json"

    query = load_file(query_file)
    contexts = load_file(contexts_file)

    embed_model: EmbedModelType = "all-MiniLM-L12-v2"

    texts = []
    for d in contexts:
        _texts = []
        # texts.append(d["parent_header"] or "")
        _texts.append(d["header"])
        _texts.append(d["content"])
        text = "\n".join(_texts)
        texts.append(text)

    ids = [d["merged_doc_id"] for d in contexts]
    id_to_result = {r["merged_doc_id"]: r for r in contexts}

    candidates = extract_query_candidates(query)
    reranked_results = rerank_by_keywords(
        texts=texts,
        embed_model=embed_model,
        ids=ids,
        top_n=10,
        candidates=candidates,
        seed_keywords=candidates,
    )

    results = []
    # Map reranked results back to original context dicts by id
    for result in reranked_results:
        doc_id = result["id"]
        context = id_to_result.get(doc_id, {})
        mapped_result = {
            "rerank_rank": result.get("rank"),
            "rerank_score": result.get("score"),
            "average_score": ((result["score"] + context["score"]) / 2),
            **context,
            "keywords": result["keywords"],
        }
        results.append(mapped_result)

    results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

    for r in results[:5]:
        print("========================================")
        print(f"Rerank Rank   : {r.get('rerank_rank')}")
        print(f"Rerank Score  : {r.get('rerank_score'):.4f}")
        print(f"Embed Score   : {r.get('score'):.4f}")
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
    for r in results[:5]:
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
        "results": results
    }, f"{output_dir}/results.json")
