from jet.llm.mlx.tasks.answer_multiple_choice_multiple_selections import answer_multiple_choice_multiple_selections, AnswerResult
from jet.llm.mlx.mlx_types import LLMModelType
from jet.logger import logger
from jet.transformers.formatters import format_json

if __name__ == "__main__":
    question = "Which colors are in the rainbow?"
    choices = ["A) Red", "B) Indigo", "C) Yellow", "D) Black"]
    model: LLMModelType = "llama-3.2-3b-instruct-4bit"
    result: AnswerResult = answer_multiple_choice_multiple_selections(
        question, choices, model, method="generate_step", max_tokens=10, temperature=0.0
    )
    logger.gray("Result:")
    logger.success(format_json(result))
