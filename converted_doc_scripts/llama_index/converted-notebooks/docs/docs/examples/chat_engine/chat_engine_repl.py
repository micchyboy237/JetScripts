from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core.chat_engine import SimpleChatEngine
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/chat_engine/chat_engine_repl.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Chat Engine - Simple Mode REPL

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Chat Engine - Simple Mode REPL")

# %pip install llama-index-llms-ollama

# !pip install llama-index

"""
### Get started in 3 lines of code

Using GPT3 ("text-davinci-003")
"""
logger.info("### Get started in 3 lines of code")


chat_engine = SimpleChatEngine.from_defaults()
chat_engine.chat_repl()

"""
### Customize LLM

Use ChatGPT ("gpt-3.5-turbo")
"""
logger.info("### Customize LLM")


llm = OllamaFunctionCallingAdapter(temperature=0.0, model="llama3.2", request_timeout=300.0, context_window=4096)


chat_engine = SimpleChatEngine.from_defaults(llm=llm)
chat_engine.chat_repl()

"""
## Streaming Support
"""
logger.info("## Streaming Support")


llm = OllamaFunctionCallingAdapter(temperature=0.0, model="llama3.2", request_timeout=300.0, context_window=4096)


chat_engine = SimpleChatEngine.from_defaults(llm=llm)

response = chat_engine.stream_chat(
    "Write me a poem about raining cats and dogs."
)
for token in response.response_gen:
    logger.debug(token, end="")

logger.info("\n\n[DONE]", bright=True)