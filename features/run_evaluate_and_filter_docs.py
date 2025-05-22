import os
from jet.file.utils import load_file, save_file
from jet.llm.mlx.tasks.eval.evaluate_multiple_contexts_relevance import evaluate_multiple_contexts_relevance, ContextRelevanceResult
from jet.llm.mlx.mlx_types import LLMModelType
from jet.logger import logger
from jet.transformers.formatters import format_json

if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/headers.json"
    headers: list[dict] = load_file(docs_file)

    model: LLMModelType = "llama-3.2-3b-instruct-4bit"
    query = "Today's date is May 22, 2025\nList trending isekai reincarnation anime this year."
    contexts = [
        header["text"]
        for header in headers
        if header["header_level"] != 1
    ]

    # query = "What is the capital of France?"
    # contexts = [
    #     "The capital of France is Paris.",
    #     "Paris is a popular tourist destination.",
    #     "Einstein developed the theory of relativity."
    # ]

    evaluate_multiple_contexts_relevance_results: list[ContextRelevanceResult] = evaluate_multiple_contexts_relevance(
        query, contexts, model)

    logger.gray("Results:")
    logger.success(format_json(evaluate_multiple_contexts_relevance_results))

    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(
            os.path.basename(__file__))[0]
    )
    save_file(evaluate_multiple_contexts_relevance_results,
              f"{output_dir}/evaluate_multiple_contexts_relevance_results.json")
