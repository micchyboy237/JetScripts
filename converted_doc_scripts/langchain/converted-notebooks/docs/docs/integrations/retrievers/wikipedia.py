from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.retrievers import WikipediaRetriever
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
sidebar_label: Wikipedia
---

# WikipediaRetriever

>[Wikipedia](https://wikipedia.org/) is a multilingual free online encyclopedia written and maintained by a community of volunteers, known as Wikipedians, through open collaboration and using a wiki-based editing system called MediaWiki. `Wikipedia` is the largest and most-read reference work in history.

This notebook shows how to retrieve wiki pages from `wikipedia.org` into the [Document](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html) format that is used downstream.

### Integration details


<ItemTable category="external_retrievers" item="WikipediaRetriever" />

## Setup
To enable automated tracing of individual tools, set your [LangSmith](https://docs.smith.langchain.com/) API key:
"""
logger.info("# WikipediaRetriever")



"""
### Installation

The integration lives in the `langchain-community` package. We also need to install the `wikipedia` python package itself.
"""
logger.info("### Installation")

# %pip install -qU langchain-community wikipedia

"""
## Instantiation

Now we can instantiate our retriever:

`WikipediaRetriever` parameters include:
- optional `lang`: default="en". Use it to search in a specific language part of Wikipedia
- optional `load_max_docs`: default=100. Use it to limit number of downloaded documents. It takes time to download all 100 documents, so use a small number for experiments. There is a hard limit of 300 for now.
- optional `load_all_available_meta`: default=False. By default only the most important fields downloaded: `Published` (date when document was published/last updated), `title`, `Summary`. If True, other fields also downloaded.

`get_relevant_documents()` has one argument, `query`: free text which used to find documents in Wikipedia
"""
logger.info("## Instantiation")


retriever = WikipediaRetriever()

"""
## Usage
"""
logger.info("## Usage")

docs = retriever.invoke("TOKYO GHOUL")

logger.debug(docs[0].page_content[:400])

"""
## Use within a chain
Like other retrievers, `WikipediaRetriever` can be incorporated into LLM applications via [chains](/docs/how_to/sequence/).

We will need a LLM or chat model:


<ChatModelTabs customVarName="llm" />
"""
logger.info("## Use within a chain")


llm = ChatOllama(model="llama3.2")


prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the context provided.
    Context: {context}
    Question: {question}
    """
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

chain.invoke(
    "Who is the main character in `Tokyo Ghoul` and does he transform into a ghoul?"
)

"""
## API reference

For detailed documentation of all `WikipediaRetriever` features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/retrievers/langchain_community.retrievers.wikipedia.WikipediaRetriever.html#langchain-community-retrievers-wikipedia-wikipediaretriever).
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)