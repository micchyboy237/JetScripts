from jet.llm.mlx.helpers.answer_multiple_choice_with_confidence import answer_multiple_choice_with_confidence
from jet.llm.mlx.mlx_types import ModelType
from jet.logger import logger
from jet.transformers.formatters import format_json

question = "Which element is known as the building block of life?"
choices = ["Oxygen", "Carbon", "Nitrogen", "Hydrogen"]
model: ModelType = "llama-3.2-3b-instruct-4bit"
result = answer_multiple_choice_with_confidence(
    question, choices, model, max_tokens=10)

logger.gray("Result:")
logger.success(format_json(result))
