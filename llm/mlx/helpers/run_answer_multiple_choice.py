from jet.llm.mlx.helpers.answer_multiple_choice import answer_multiple_choice
from jet.logger import logger
from jet.transformers.formatters import format_json


question = "Which planet is known as the Red Planet?"
choices = ["Mars", "Earth", "Jupiter", "Saturn"]
result = answer_multiple_choice(question, choices)

logger.gray("Result:")
logger.success(format_json(result))
