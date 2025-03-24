from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import os
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# Migrating from LLMChain
#
# [`LLMChain`](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.llm.LLMChain.html) combined a prompt template, LLM, and output parser into a class.
#
# Some advantages of switching to the LCEL implementation are:
#
# - Clarity around contents and parameters. The legacy `LLMChain` contains a default output parser and other options.
# - Easier streaming. `LLMChain` only supports streaming via callbacks.
# - Easier access to raw message outputs if desired. `LLMChain` only exposes these via a parameter or via callback.

# %pip install --upgrade --quiet langchain-openai

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass()

# Legacy
#
# <details open>


prompt = ChatPromptTemplate.from_messages(
    [("user", "Tell me a {adjective} joke")],
)

legacy_chain = LLMChain(llm=ChatOllama(model="llama3.1"), prompt=prompt)

legacy_result = legacy_chain({"adjective": "funny"})
legacy_result

# Note that `LLMChain` by default returned a `dict` containing both the input and the output from `StrOutputParser`, so to extract the output, you need to access the `"text"` key.

legacy_result["text"]

# </details>
#
# LCEL
#
# <details open>


prompt = ChatPromptTemplate.from_messages(
    [("user", "Tell me a {adjective} joke")],
)

chain = prompt | ChatOllama(model="gemma2:9b") | StrOutputParser()

chain.invoke({"adjective": "funny"})

# If you'd like to mimic the `dict` packaging of input and output in `LLMChain`, you can use a [`RunnablePassthrough.assign`](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.passthrough.RunnablePassthrough.html) like:


outer_chain = RunnablePassthrough().assign(text=chain)

outer_chain.invoke({"adjective": "funny"})

# </details>
#
# Next steps
#
# See [this tutorial](/docs/tutorials/llm_chain) for more detail on building with prompt templates, LLMs, and output parsers.
#
# Check out the [LCEL conceptual docs](/docs/concepts/lcel) for more background information.

logger.info("\n\n[DONE]", bright=True)
