import os
from jet.file.utils import save_file
from jet.llm.mlx.tasks.answer_multiple_choice_with_context import answer_multiple_choice_with_context, AnswerResult
from jet.llm.mlx.mlx_types import LLMModelType
from jet.logger import logger
from jet.transformers.formatters import format_json

if __name__ == "__main__":
    question = "Which element is known as the building block of life?"
    choices = ["A) Oxygen", "B) Carbon", "C) Nitrogen", "D) Hydrogen"]
    contexts = [
        "Oxygen is essential for respiration.",
        "Carbon forms the backbone of organic molecules.",
        "Nitrogen is a key component of amino acids.",
        "Hydrogen is present in water and organic compounds."
    ]
    result = answer_multiple_choice_with_context(
        question, choices, contexts, model_path="llama-3.2-3b-instruct-4bit")

    logger.gray("Result:")
    logger.success(format_json(result))

    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    save_file(result, f"{output_dir}/results.json")
