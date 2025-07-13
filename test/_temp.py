from typing import TypedDict
from jet.file.utils import load_file
from jet.logger import logger
from jet.models.model_registry.transformers.cross_encoder_model_registry import CrossEncoderRegistry
from jet.models.model_types import EmbedModelType
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from jet.wordnet.keywords.helpers import extract_keyword_candidates, extract_query_candidates


class RerankResult(TypedDict):
    score: float
    text: str


if __name__ == '__main__':
    query_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_2/top_isekai_anime_2025/query.md"
    contexts_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_2/top_isekai_anime_2025/contexts_search_results.json"

    query = load_file(query_file)
    contexts = load_file(contexts_file)

    rerank_model: EmbedModelType = "ms-marco-MiniLM-L6-v2"

    model = CrossEncoderRegistry.load_model(rerank_model)

    candidates = []
    for d in contexts:
        _texts = []
        _texts.append(d["header"])
        _texts.append(d["content"])
        text = "\n".join(_texts)
        candidates.append(text)

    queries = extract_keyword_candidates(query)

    pairs = [(query, path) for query in queries for path in candidates]

    scores = model.predict(pairs)
    ranked_results: list[RerankResult] = [
        {"score": score, "text": text}
        for text, score in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    ]

    print("Results:")
    logger.success(format_json(ranked_results))
    copy_to_clipboard(ranked_results)
