import os
import shutil
from jet.features.nltk_search import search_by_pos
from jet.file.utils import load_file, save_file
from jet.llm.mlx.tasks.eval.evaluate_multiple_contexts_relevance import evaluate_multiple_contexts_relevance, ContextRelevanceResult
from jet.llm.mlx.mlx_types import LLMModelType
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.vectors.hybrid_reranker import search_documents
from jet.vectors.hybrid_reranker import Models, ScoreResults, SearchResult, calculate_scores, load_models, search_documents

if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(
            os.path.basename(__file__))[0]
    )
    shutil.rmtree(output_dir, ignore_errors=True)

    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/headers.json"
    headers: list[dict] = load_file(docs_file)

    model: LLMModelType = "llama-3.2-3b-instruct-4bit"
    # query = "What is the capital of France?"
    # contexts = [
    #     "The capital of France is Paris.",
    #     "Paris is a popular tourist destination.",
    #     "Einstein developed the theory of relativity."
    # ]
    query = "Today's date is May 22, 2025\nList trending isekai reincarnation anime this year."
    header_texts = [
        f"{header["parent_header"]}\n{header["text"]}"
        for header in headers
        if header["header_level"] != 1
    ]

    # Search by POS
    search_doc_results = search_by_pos(query, header_texts)
    save_file(search_doc_results, f"{output_dir}/search_doc_results.json")

    searched_contexts = [
        header["text"]
        for header in search_doc_results
    ]

    evaluate_multiple_contexts_relevance_results: list[ContextRelevanceResult] = evaluate_multiple_contexts_relevance(
        query, searched_contexts, model)

    logger.gray("Results:")
    logger.success(format_json(evaluate_multiple_contexts_relevance_results))

    save_file(evaluate_multiple_contexts_relevance_results,
              f"{output_dir}/evaluate_multiple_contexts_relevance_results.json")
