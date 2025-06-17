from jet.logger import logger
from jet.tasks.intent_classifier import classify_text
from jet.transformers.formatters import format_json


if __name__ == "__main__":

    text = "Your text to classify here."
    result = classify_text(text, batch_size=4)
    logger.gray("Result:")
    logger.success(format_json(result))
