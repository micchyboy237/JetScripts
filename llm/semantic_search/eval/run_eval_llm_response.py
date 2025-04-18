import asyncio
import os
from jet.features.eval_search_and_chat import evaluate_llm_response
from jet.file.utils import load_file, save_file
from jet.llm.evaluators.answer_relevancy_evaluator import evaluate_answer_relevancy
from jet.llm.ollama.base import Ollama
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


def main():
    output_dir = OUTPUT_DIR

    # data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/generated/run_ollama_base_chat/llm_chat_history.json"
    # data = load_file(data_file)

    query = "What is the capital of France?"
    response = "The capital of France is Paris."
    context = "France is a country in Western Europe, and its capital city is Paris."

    eval_result = evaluate_llm_response(
        query=query,
        response=response,
        context=context,
        output_dir=output_dir,
    )

    logger.success(format_json(eval_result))
    copy_to_clipboard(eval_result)


if __name__ == "__main__":
    main()
