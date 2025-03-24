from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()


logger.info("\n\n[DONE]", bright=True)
