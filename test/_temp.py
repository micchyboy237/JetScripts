from jet.file.utils import load_file
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from jet.wordnet.keywords.keyword_extraction import rerank_by_keywords


contexts_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_2/top_isekai_anime_2025/contexts_before_max_filter.json"

contexts = load_file(contexts_file)

texts = [f"{d["header"]}\n{d["content"]}" for d in contexts]
ids = [d["merged_doc_id"] for d in contexts]

reranked_results = rerank_by_keywords(
    texts=texts,
    ids=ids,
    top_n=10
)

logger.success(format_json(reranked_results))
copy_to_clipboard(reranked_results)
