from jet.llm.mlx.helpers.answer_multiple_choice_with_key import answer_multiple_choice_with_key, AnswerResult
from jet.llm.mlx.mlx_types import ModelType
from jet.logger import logger
from jet.transformers.formatters import format_json

if __name__ == "__main__":
    question = "Which element is known as the building block of life?"
    choices = ["A) Oxygen", "B) Carbon", "C) Nitrogen", "D) Hydrogen"]
    model: ModelType = "llama-3.2-3b-instruct-4bit"
    result: AnswerResult = answer_multiple_choice_with_key(
        question, choices, model, method="generate_step", max_tokens=10, temperature=0.0
    )
    logger.gray("Result:")
    logger.success(format_json(result))
