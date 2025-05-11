from jet.llm.mlx.helpers.answer_multiple_choice_by_key import answer_multiple_choice_by_key
from jet.logger import logger
from jet.transformers.formatters import format_json


question = "What is the most abundant gas in Earth's atmosphere?"
choices = [
    "A) Oxygen",
    "B) Nitrogen",
    "C) Carbon Dioxide",
    "D) Argon",
]
result = answer_multiple_choice_by_key(question, choices)

logger.gray("Result:")
logger.success(format_json(result))
