from jet.logger import logger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# String-in, string-out llms

:::tip
You are probably looking for the [Chat Model Concept Guide](/docs/concepts/chat_models) page for more information.
:::

LangChain has implementations for older language models that take a string as input and return a string as output. These models are typically named without the "Chat" prefix (e.g., `Ollama`, `Ollama`, `Ollama`, etc.), and may include the "LLM" suffix (e.g., `OllamaLLM`, `OllamaLLM`, `OllamaLLM`, etc.). These models implement the [BaseLLM](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.llms.BaseLLM.html#langchain_core.language_models.llms.BaseLLM) interface.

Users should be using almost exclusively the newer [Chat Models](/docs/concepts/chat_models) as most
model providers have adopted a chat like interface for interacting with language models.
"""
logger.info("# String-in, string-out llms")

logger.info("\n\n[DONE]", bright=True)