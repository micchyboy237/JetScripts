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


def main():
    data_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
    llm_model = "llama3.1"
    num_questions_per_chunk = 3

    file_no_ext = os.path.basename(__file__).split(".")[0]
    generated_dir = os.path.join("generated", file_no_ext)

    questions_generator = QuestionGenerator(
        data_path=data_path,
        num_questions_per_chunk=num_questions_per_chunk,
        llm_model=llm_model,
    )

    questions = questions_generator.generate_questions()
    logger.newline()
    logger.info(f"Generated eval questions ({(len(questions))}):")
    logger.success(format_json(questions))

    save_file(questions, os.path.join(generated_dir, "questions.json"))

    eval_results_stream = questions_generator.evaluate_questions(questions)

    eval_results = []
    guideline_results = []
    for idx, eval_result in enumerate(eval_results_stream):
        eval_results.append(eval_result)
        logger.newline()
        logger.info(f"Evaluated question result {idx + 1}:")
        logger.success(format_json(eval_result))
        save_file(eval_results, os.path.join(
            generated_dir, "eval_results.json"))

        guideline_eval_results = run_evaluate_guidelines(
            eval_result.query,
            eval_result.contexts,
            eval_result.response,
        )
        guideline_results.append({
            "data": {
                "query": eval_result.query,
                "contexts": eval_result.contexts,
                "response": eval_result.response,
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
