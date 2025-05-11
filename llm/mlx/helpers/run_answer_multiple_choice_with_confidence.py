from jet.llm.mlx.helpers.answer_multiple_choice_with_confidence import answer_multiple_choice_with_confidence
from jet.logger import logger
from jet.transformers.formatters import format_json

question = "Which element is known as the building block of life?"
choices = ["Oxygen", "Carbon", "Nitrogen", "Hydrogen"]
result = answer_multiple_choice_with_confidence(question, choices)

logger.gray("Result:")
logger.success(format_json(result))
