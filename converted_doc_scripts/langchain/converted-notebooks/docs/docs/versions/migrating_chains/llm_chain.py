from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
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
# Migrating from LLMChain

[`LLMChain`](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.llm.LLMChain.html) combined a prompt template, LLM, and output parser into a class.

Some advantages of switching to the LCEL implementation are:

- Clarity around contents and parameters. The legacy `LLMChain` contains a default output parser and other options.
- Easier streaming. `LLMChain` only supports streaming via callbacks.
- Easier access to raw message outputs if desired. `LLMChain` only exposes these via a parameter or via callback.
"""
logger.info("# Migrating from LLMChain")

# %pip install --upgrade --quiet langchain-ollama

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass()

"""
## Legacy

<details open>
"""
logger.info("## Legacy")


prompt = ChatPromptTemplate.from_messages(
    [("user", "Tell me a {adjective} joke")],
)

legacy_chain = LLMChain(llm=ChatOllama(model="llama3.2"), prompt=prompt)

legacy_result = legacy_chain({"adjective": "funny"})
legacy_result

"""
Note that `LLMChain` by default returned a `dict` containing both the input and the output from `StrOutputParser`, so to extract the output, you need to access the `"text"` key.
"""
logger.info("Note that `LLMChain` by default returned a `dict` containing both the input and the output from `StrOutputParser`, so to extract the output, you need to access the `"text"` key.")

legacy_result["text"]

"""
</details>

## LCEL

<details open>
"""
logger.info("## LCEL")


prompt = ChatPromptTemplate.from_messages(
    [("user", "Tell me a {adjective} joke")],
)

chain = prompt | ChatOllama(model="llama3.2") | StrOutputParser()

chain.invoke({"adjective": "funny"})

"""
If you'd like to mimic the `dict` packaging of input and output in `LLMChain`, you can use a [`RunnablePassthrough.assign`](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.passthrough.RunnablePassthrough.html) like:
"""
logger.info("If you'd like to mimic the `dict` packaging of input and output in `LLMChain`, you can use a [`RunnablePassthrough.assign`](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.passthrough.RunnablePassthrough.html) like:")


outer_chain = RunnablePassthrough().assign(text=chain)

outer_chain.invoke({"adjective": "funny"})

"""
</details>

## Next steps

See [this tutorial](/docs/tutorials/llm_chain) for more detail on building with prompt templates, LLMs, and output parsers.

Check out the [LCEL conceptual docs](/docs/concepts/lcel) for more background information.
"""
logger.info("## Next steps")

logger.info("\n\n[DONE]", bright=True)