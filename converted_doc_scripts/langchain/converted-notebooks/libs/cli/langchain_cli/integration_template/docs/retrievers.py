from __module_name__ import __ModuleName__Retriever
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import ChatModelTabs from "@theme/ChatModelTabs";
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
---
sidebar_label: __ModuleName__
---

# __ModuleName__Retriever

- TODO: Make sure API reference link is correct.

This will help you get started with the __ModuleName__ [retriever](/docs/concepts/retrievers). For detailed documentation of all __ModuleName__Retriever features and configurations head to the [API reference](https://api.python.langchain.com/en/latest/retrievers/__module_name__.retrievers.__ModuleName__.__ModuleName__Retriever.html).

### Integration details

TODO: Select one of the tables below, as appropriate.

1: Bring-your-own data (i.e., index and search a custom corpus of documents):

| Retriever | Self-host | Cloud offering | Package |
| :--- | :--- | :---: | :---: |
[__ModuleName__Retriever](https://api.python.langchain.com/en/latest/retrievers/__package_name__.retrievers.__module_name__.__ModuleName__Retriever.html) | ❌ | ❌ | __package_name__ |

2: External index (e.g., constructed from Internet data or similar)):

| Retriever | Source | Package |
| :--- | :--- | :---: |
[__ModuleName__Retriever](https://api.python.langchain.com/en/latest/retrievers/__package_name__.retrievers.__module_name__.__ModuleName__Retriever.html) | Source description | __package_name__ |

## Setup

- TODO: Update with relevant info.

If you want to get automated tracing from individual queries, you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
"""
logger.info("# __ModuleName__Retriever")



"""
### Installation

This retriever lives in the `__package_name__` package:
"""
logger.info("### Installation")

# %pip install -qU __package_name__

"""
## Instantiation

Now we can instantiate our retriever:

- TODO: Update model instantiation with relevant params.
"""
logger.info("## Instantiation")


retriever = __ModuleName__Retriever(
)

"""
## Usage
"""
logger.info("## Usage")

query = "..."

retriever.invoke(query)

"""
## Use within a chain

Like other retrievers, __ModuleName__Retriever can be incorporated into LLM applications via [chains](/docs/how_to/sequence/).

We will need a LLM or chat model:


<ChatModelTabs customVarName="llm" />
"""
logger.info("## Use within a chain")


llm = ChatOllama(model="llama3.2")


prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided.

Context: {context}

Question: {question}"""
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

chain.invoke("...")

"""
## TODO: Any functionality or considerations specific to this retriever

Fill in or delete if not relevant.

## API reference

For detailed documentation of all __ModuleName__Retriever features and configurations head to the [API reference](https://api.python.langchain.com/en/latest/retrievers/__module_name__.retrievers.__ModuleName__.__ModuleName__Retriever.html).
"""
logger.info("## TODO: Any functionality or considerations specific to this retriever")

logger.info("\n\n[DONE]", bright=True)