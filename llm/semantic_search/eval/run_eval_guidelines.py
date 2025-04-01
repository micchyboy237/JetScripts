import asyncio
import json
import os
from jet.file.utils import save_file
from jet.llm.helpers.qa_dataset_generator import QADatasetGenerator
from jet.llm.helpers.question_generator import QuestionGenerator
from jet.llm.ollama.base import Ollama
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.transformers.object import make_serializable
from llama_index.core.evaluation.guideline import GuidelineEvaluator

GUIDELINES = [
    "The response should fully answer the query.",
    "The response should avoid being vague or ambiguous.",
    (
        "The response should be specific and use statistics or numbers when"
        " possible."
    ),
]


def run_evaluate_guidelines(query: str, contexts: list[str], response: str):
    llm = Ollama(model="llama3.1")

    evaluators = [
        GuidelineEvaluator(llm=llm, guidelines=guideline)
        for guideline in GUIDELINES
    ]

    results = []
    for guideline, evaluator in zip(GUIDELINES, evaluators):
        eval_result = evaluator.evaluate(
            query=query,
            contexts=contexts,
            response=response,
        )
        print("=====")
        print(f"Guideline: {guideline}")
        print(f"Pass: {eval_result.passing}")
        print(f"Feedback: {eval_result.feedback}")
        results.append({
            "guideline": guideline,
            "passed": eval_result.passing,
            "feedback": eval_result.feedback,
        })
    return results


def main():
    eval_results_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/semantic_search/eval/generated/run_generate_questions/eval_results.json"
    with open(eval_results_path, "r") as f:
        eval_results = json.load(f)

    generated_dir = os.path.dirname(eval_results_path)

    guideline_results = []
    for idx, eval_result in enumerate(eval_results):
        guideline_eval_results = run_evaluate_guidelines(
            eval_result["query"],
            eval_result["contexts"],
            eval_result["response"],
        )
        guideline_results.append({
            "data": {
                "query": eval_result["query"],
                "contexts": eval_result["contexts"],
                "response": eval_result["response"],
            },
            "guideline_eval_results": guideline_eval_results
        })
        logger.newline()
        logger.info(f"Evaluated guideline result {idx + 1}:")
        logger.success(format_json(guideline_eval_results))
        save_file(guideline_results, os.path.join(
            generated_dir, "guideline_eval_results.json"))

    logger.info("\n\n[DONE]", bright=True)


if __name__ == "__main__":
    main()
