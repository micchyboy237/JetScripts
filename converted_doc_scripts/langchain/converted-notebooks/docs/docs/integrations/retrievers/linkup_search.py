from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_linkup import LinkupSearchRetriever
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
sidebar_label: LinkupSearchRetriever
---

# LinkupSearchRetriever

> [Linkup](https://www.linkup.so/) provides an API to connect LLMs to the web and the Linkup Premium Partner sources.

This will help you get started with the LinkupSearchRetriever [retriever](/docs/concepts/retrievers/). For detailed documentation of all LinkupSearchRetriever features and configurations head to the [API reference](https://python.langchain.com/api_reference/linkup/retrievers/linkup_langchain.search_retriever.LinkupSearchRetriever.html).

### Integration details

| Retriever | Source | Package |
| :--- | :--- | :---: |
[LinkupSearchRetriever](https://python.langchain.com/api_reference/linkup/retrievers/linkup_langchain.search_retriever.LinkupSearchRetriever.html) | Web and partner sources | langchain-linkup |

## Setup

# To use the Linkup provider, you need a valid API key, which you can find by signing-up [here](https://app.linkup.so/sign-up). You can then set it up as the `LINKUP_API_KEY` environment variable. For the chain example below, you also need to set an Ollama API key as `OPENAI_API_KEY` environment variable, which you can also do here:
"""
logger.info("# LinkupSearchRetriever")



"""
If you want to get automated tracing from individual queries, you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
"""
logger.info("If you want to get automated tracing from individual queries, you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:")



"""
### Installation

This retriever lives in the `langchain-linkup` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-linkup

"""
## Instantiation

Now we can instantiate our retriever:
"""
logger.info("## Instantiation")


retriever = LinkupSearchRetriever(
    depth="deep",  # "standard" or "deep"
    linkup_api_key=None,  # API key can be passed here or set as the LINKUP_API_KEY environment variable
)

"""
## Usage
"""
logger.info("## Usage")

query = "Who won the latest US presidential elections?"

retriever.invoke(query)

"""
## Use within a chain

Like other retrievers, LinkupSearchRetriever can be incorporated into LLM applications via [chains](/docs/how_to/sequence/).

We will need a LLM or chat model:

```{=mdx}

<ChatModelTabs customVarName="llm" />
```
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

chain.invoke("Who won the 3 latest US presidential elections?")

"""
## API reference

For detailed documentation of all LinkupSearchRetriever features and configurations head to the [API reference](https://python.langchain.com/api_reference/linkup/retrievers/linkup_langchain.search_retriever.LinkupSearchRetriever.html).
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)