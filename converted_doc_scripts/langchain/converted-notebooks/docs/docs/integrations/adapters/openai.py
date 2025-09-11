from jet.logger import logger
from langchain_community.adapters import ollama as lc_ollama
import ollama
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
# Ollama Adapter

**Please ensure Ollama library is version 1.0.0 or higher; otherwise, refer to the older doc [Ollama Adapter(Old)](/docs/integrations/adapters/ollama-old/).**

A lot of people get started with Ollama but want to explore other models. LangChain's integrations with many model providers make this easy to do so. While LangChain has it's own message and model APIs, we've also made it as easy as possible to explore other models by exposing an adapter to adapt LangChain models to the Ollama api.

At the moment this only deals with output and does not return other information (token counts, stop reasons, etc).
"""
logger.info("# Ollama Adapter")


"""
## chat.completions.create
"""
logger.info("## chat.completions.create")

messages = [{"role": "user", "content": "hi"}]

"""
Original Ollama call
"""
logger.info("Original Ollama call")

result = ollama.chat.completions.create(
    messages=messages, model="llama3.2", temperature=0
)
result.choices[0].message.model_dump()

"""
LangChain Ollama wrapper call
"""
logger.info("LangChain Ollama wrapper call")

lc_result = lc_ollama.chat.completions.create(
    messages=messages, model="llama3.2", temperature=0
)

lc_result.choices[0].message  # Attribute access

lc_result["choices"][0]["message"]  # Also compatible with index access

"""
Swapping out model providers
"""
logger.info("Swapping out model providers")

lc_result = lc_ollama.chat.completions.create(
    messages=messages, model="claude-2", temperature=0, provider="ChatOllama"
)
lc_result.choices[0].message

"""
## chat.completions.stream

Original Ollama call
"""
logger.info("## chat.completions.stream")

for c in ollama.chat.completions.create(
    messages=messages, model="llama3.2", temperature=0, stream=True
):
    logger.debug(c.choices[0].delta.model_dump())

"""
LangChain Ollama wrapper call
"""
logger.info("LangChain Ollama wrapper call")

for c in lc_ollama.chat.completions.create(
    messages=messages, model="llama3.2", temperature=0, stream=True
):
    logger.debug(c.choices[0].delta)

"""
Swapping out model providers
"""
logger.info("Swapping out model providers")

for c in lc_ollama.chat.completions.create(
    messages=messages,
    model="claude-2",
    temperature=0,
    stream=True,
    provider="ChatOllama",
):
    logger.debug(c["choices"][0]["delta"])

logger.info("\n\n[DONE]", bright=True)