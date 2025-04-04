from jet.file.utils import load_file, save_file
from jet.llm.evaluators.context_relevancy_evaluator import evaluate_context_relevancy
from jet.llm.ollama.base import Ollama
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard


if __name__ == "__main__":
    eval_model = "gemma3:4b"
    query = "What are the steps in registering a National ID in the Philippines?"

    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/generated/run_llm_reranker/final_llm_reranker_results.json"
    output_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/semantic_search/eval/generated/run_eval_context_relevancy"
    data = load_file(data_file)
    contexts = [d["text"] for d in data]

    eval_result = evaluate_context_relevancy(eval_model, query, contexts)

    logger.success(format_json(eval_result))
    copy_to_clipboard(eval_result)

    output_file = f"{output_dir}/eval_result.json"
    save_file(eval_result, output_file)
