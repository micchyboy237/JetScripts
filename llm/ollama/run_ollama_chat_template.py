from jet.llm.ollama.base import OLLAMA_HF_MODELS, get_chat_template
from jet.logger import logger


def main_get_chat_templates():
    # Iterate over the models and count tokens for each
    for model_key in OLLAMA_HF_MODELS.keys():
        logger.newline()
        logger.log("Getting chat template for", model_key,
                   "...", colors=["WHITE", "INFO"])
        chat_template = get_chat_template(model_key)
        logger.debug(f"{model_key} template:")
        logger.success(chat_template)


if __name__ == "__main__":
    logger.newline()
    logger.debug("main_get_chat_templates()")
    main_get_chat_templates()
