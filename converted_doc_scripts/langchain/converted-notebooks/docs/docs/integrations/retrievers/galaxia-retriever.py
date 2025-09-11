from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_galaxia_retriever.retriever import GalaxiaRetriever
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
sidebar_label: Galaxia
---

# Galaxia Retriever

Galaxia is GraphRAG solution, which automates document processing, knowledge base (Graph Language Model) creation and retrieval:
[galaxia-rag](https://smabbler.gitbook.io/smabbler/api-rag/smabblers-api-rag)

To use Galaxia first upload your texts and create a Graph Language Model here: [smabbler-cloud](https://beta.cloud.smabbler.com)

After the model is built and activated, you will be able to use this integration to retrieve what you need.

The module repository is located here: [github](https://github.com/rrozanski-smabbler/galaxia-langchain)

### Integration details
| Retriever | Self-host | Cloud offering | Package |
| :--- | :--- | :---: | :---: |
[Galaxia Retriever](https://github.com/rrozanski-smabbler/galaxia-langchain) | ❌ | ✅ | __langchain-galaxia-retriever__ |

## Setup
Before you can retrieve anything you need to create your Graph Language Model here: [smabbler-cloud](https://beta.cloud.smabbler.com)

following these 3 simple steps: [rag-instruction](https://smabbler.gitbook.io/smabbler/api-rag/build-rag-model-in-3-steps)

Don't forget to activate the model after building it!

### Installation
The retriever is implemented in the following package: [pypi](https://pypi.org/project/langchain-galaxia-retriever/)
"""
logger.info("# Galaxia Retriever")

# %pip install -qU langchain-galaxia-retriever

"""
## Instantiation
"""
logger.info("## Instantiation")


gr = GalaxiaRetriever(
    api_url="beta.api.smabbler.com",  # you can find it here: https://beta.cloud.smabbler.com/user/account
    knowledge_base_id="<knowledge_base_id>",  # you can find it in https://beta.cloud.smabbler.com , in the model table
    n_retries=10,
    wait_time=5,
)

"""
## Usage
"""
logger.info("## Usage")

result = gr.invoke("<test question>")
logger.debug(result)

"""
## Use within a chain
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
    {"context": gr | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

chain.invoke("<test question>")

"""
## API reference

For more information about Galaxia Retriever check its implementation on github [github](https://github.com/rrozanski-smabbler/galaxia-langchain)
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)