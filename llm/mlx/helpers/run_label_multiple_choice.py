from jet.llm.mlx.helpers.label_multiple_choice import label_multiple_choice
from jet.logger import logger
from jet.transformers.formatters import format_json

instruction = "Label the sentiment of the text: 'I love this app, it's fantastic!'"
labels = ["Positive", "Negative", "Neutral"]

result = label_multiple_choice(instruction, labels)
logger.gray("Label Result:")
logger.success(format_json(result))
