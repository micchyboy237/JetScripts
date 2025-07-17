import os
import json
import re
from jet.file.utils import load_file, save_file
from jet.models.model_types import EmbedModelType
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from jet.wordnet.keywords.helpers import extract_query_candidates, preprocess_texts
from jet.wordnet.keywords.keyword_extraction_cross_encoder import extract_keywords_cross_encoder

output_dir = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

if __name__ == '__main__':
    query_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_2/top_isekai_anime_2025/query.md"
    contexts_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_2/top_isekai_anime_2025/contexts_search_results.json"

    query = load_file(query_file)
    contexts = load_file(contexts_file)

    cross_encoder_model: EmbedModelType = "ms-marco-MiniLM-L6-v2"

    texts = []
    for d in contexts:
        _texts = []
        if d["header"].lstrip('#') != (d["parent_header"] or "").lstrip('#'):
            _texts.append((d["parent_header"] or "").lstrip('#'))
        _texts.append(d["header"].lstrip('#'))
        _texts.append(d["content"])
        text = "\n".join(_texts)
        texts.append(text)
    texts = preprocess_texts(texts)

    ids = [d["merged_doc_id"] for d in contexts]
    id_to_result = {r["merged_doc_id"]: r for r in contexts}

    # candidates = extract_query_candidates(query)
    candidates = extract_query_candidates(
        preprocess_texts(query)[0] + "\n" + "\n".join([preprocess_texts(d["header"].lstrip('#'))[0] for d in contexts]))
    # Filter out candidates that are only punctuation (e.g., ".", "!!", etc.)
    candidates = [c for c in candidates if re.search(r'\w', c)]
    reranked_results = extract_keywords_cross_encoder(
        texts=texts,
        cross_encoder_model=cross_encoder_model,
        ids=ids,
        top_n=10,
        candidates=candidates,
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
