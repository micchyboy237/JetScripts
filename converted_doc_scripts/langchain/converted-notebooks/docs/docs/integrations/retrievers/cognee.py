from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_cognee import CogneeRetriever
from langchain_core.documents import Document
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
sidebar_label: Cognee
---

# CogneeRetriever

This will help you get started with the Cognee [retriever](/docs/concepts/retrievers). For detailed documentation of all CogneeRetriever features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/retrievers/langchain_community.retrievers.cognee.CogneeRetriever.html).

### Integration details

Bring-your-own data (i.e., index and search a custom corpus of documents):

| Retriever | Self-host | Cloud offering | Package |
| :--- | :--- | :---: | :---: |
[CogneeRetriever](https://python.langchain.com/api_reference/community/retrievers/langchain_community.retrievers.cognee.CogneeRetriever.html) | ✅ | ❌ | langchain-cognee |

## Setup

For cognee default setup, only thing you need is your Ollama API key.

If you want to get automated tracing from individual queries, you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
"""
logger.info("# CogneeRetriever")



"""
### Installation

This retriever lives in the `langchain-cognee` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-cognee

# import nest_asyncio

# nest_asyncio.apply()

"""
## Instantiation

Now we can instantiate our retriever:
"""
logger.info("## Instantiation")


retriever = CogneeRetriever(
    llm_# Ollama API Key
    dataset_name="my_dataset",
    k=3,
)

"""
## Usage

Add some documents, process them, and then run queries. Cognee retrieves relevant knowledge to your queries and generates final answers.
"""
logger.info("## Usage")


docs = [
    Document(page_content="Elon Musk is the CEO of SpaceX."),
    Document(page_content="SpaceX focuses on rockets and space travel."),
]

retriever.add_documents(docs)
retriever.process_data()

query = "Tell me about Elon Musk"
results = retriever.invoke(query)

for idx, doc in enumerate(results, start=1):
    logger.debug(f"Doc {idx}: {doc.page_content}")

"""
## Use within a chain

Like other retrievers, CogneeRetriever can be incorporated into LLM applications via [chains](/docs/how_to/sequence/).

We will need a LLM or chat model:


<ChatModelTabs customVarName="llm" />
"""
logger.info("## Use within a chain")


llm = ChatOllama(model="llama3.2")


retriever = CogneeRetriever(llm_dataset_name="my_dataset", k=3)

retriever.prune()

docs = [
    Document(page_content="Elon Musk is the CEO of SpaceX."),
    Document(page_content="SpaceX focuses on space travel."),
]
retriever.add_documents(docs)
retriever.process_data()


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

answer = chain.invoke("What companies do Elon Musk own?")

logger.debug("\nFinal chain answer:\n", answer)

"""
## API reference

TODO: add link to API reference.


"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)