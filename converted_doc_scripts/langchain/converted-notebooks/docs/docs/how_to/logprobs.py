from jet.adapters.langchain.chat_ollama import ChatOllama
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
# How to get log probabilities

:::info Prerequisites

This guide assumes familiarity with the following concepts:
- [Chat models](/docs/concepts/chat_models)
- [Tokens](/docs/concepts/tokens)

:::

Certain [chat models](/docs/concepts/chat_models/) can be configured to return token-level log probabilities representing the likelihood of a given token. This guide walks through how to get this information in LangChain.

## Ollama

Install the LangChain x Ollama package and set your API key
"""
logger.info("# How to get log probabilities")

# %pip install -qU langchain-ollama

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass()

"""
For the Ollama API to return log probabilities we need to configure the `logprobs=True` param. Then, the logprobs are included on each output [`AIMessage`](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessage.html) as part of the `response_metadata`:
"""
logger.info("For the Ollama API to return log probabilities we need to configure the `logprobs=True` param. Then, the logprobs are included on each output [`AIMessage`](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessage.html) as part of the `response_metadata`:")


llm = ChatOllama(model="llama3.2").bind(logprobs=True)

msg = llm.invoke(("human", "how are you today"))

msg.response_metadata["logprobs"]["content"][:5]

"""
And are part of streamed Message chunks as well:
"""
logger.info("And are part of streamed Message chunks as well:")

ct = 0
full = None
for chunk in llm.stream(("human", "how are you today")):
    if ct < 5:
        full = chunk if full is None else full + chunk
        if "logprobs" in full.response_metadata:
            logger.debug(full.response_metadata["logprobs"]["content"])
    else:
        break
    ct += 1

"""
## Next steps

You've now learned how to get logprobs from Ollama models in LangChain.

Next, check out the other how-to guides chat models in this section, like [how to get a model to return structured output](/docs/how_to/structured_output) or [how to track token usage](/docs/how_to/chat_token_usage_tracking).
"""
logger.info("## Next steps")

logger.info("\n\n[DONE]", bright=True)