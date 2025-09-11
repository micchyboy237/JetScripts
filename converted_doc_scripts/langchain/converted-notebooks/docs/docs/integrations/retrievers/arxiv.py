from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.retrievers import ArxivRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import ChatModelTabs from "@theme/ChatModelTabs";
import os
import shutil
import {ItemTable} from "@theme/FeatureTables";


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
sidebar_label: Arxiv
---

# ArxivRetriever

>[arXiv](https://arxiv.org/) is an open-access archive for 2 million scholarly articles in the fields of physics, mathematics, computer science, quantitative biology, quantitative finance, statistics, electrical engineering and systems science, and economics.

This notebook shows how to retrieve scientific articles from Arxiv.org into the [Document](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html) format that is used downstream.

For detailed documentation of all `ArxivRetriever` features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/retrievers/langchain_community.retrievers.arxiv.ArxivRetriever.html).

### Integration details


<ItemTable category="external_retrievers" item="ArxivRetriever" />

## Setup

If you want to get automated tracing from individual queries, you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
"""
logger.info("# ArxivRetriever")



"""
### Installation

This retriever lives in the `langchain-community` package. We will also need the [arxiv](https://pypi.org/project/arxiv/) dependency:
"""
logger.info("### Installation")

# %pip install -qU langchain-community arxiv

"""
## Instantiation

`ArxivRetriever` parameters include:
- optional `load_max_docs`: default=100. Use it to limit number of downloaded documents. It takes time to download all 100 documents, so use a small number for experiments. There is a hard limit of 300 for now.
- optional `load_all_available_meta`: default=False. By default only the most important fields downloaded: `Published` (date when document was published/last updated), `Title`, `Authors`, `Summary`. If True, other fields also downloaded.
- `get_full_documents`: boolean, default False. Determines whether to fetch full text of documents.

See [API reference](https://python.langchain.com/api_reference/community/retrievers/langchain_community.retrievers.arxiv.ArxivRetriever.html) for more detail.
"""
logger.info("## Instantiation")


retriever = ArxivRetriever(
    load_max_docs=2,
    get_ful_documents=True,
)

"""
## Usage

`ArxivRetriever` supports retrieval by article identifier:
"""
logger.info("## Usage")

docs = retriever.invoke("1605.08386")

docs[0].metadata  # meta-information of the Document

docs[0].page_content[:400]  # a content of the Document

"""
`ArxivRetriever` also supports retrieval based on natural language text:
"""

docs = retriever.invoke("What is the ImageBind model?")

docs[0].metadata

"""
## Use within a chain

Like other retrievers, `ArxivRetriever` can be incorporated into LLM applications via [chains](/docs/how_to/sequence/).

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

chain.invoke("What is the ImageBind model?")

"""
## API reference

For detailed documentation of all `ArxivRetriever` features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/retrievers/langchain_community.retrievers.arxiv.ArxivRetriever.html).
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)