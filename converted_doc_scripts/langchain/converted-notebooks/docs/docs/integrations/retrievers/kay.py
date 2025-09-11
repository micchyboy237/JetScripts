from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chains import ConversationalRetrievalChain
from langchain_community.retrievers import KayAiRetriever
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
# Kay.ai

>[Kai Data API](https://www.kay.ai/) built for RAG 🕵️ We are curating the world's largest datasets as high-quality embeddings so your AI agents can retrieve context on the fly. Latest models, fast retrieval, and zero infra.

This notebook shows you how to retrieve datasets supported by [Kay](https://kay.ai/). You can currently search `SEC Filings` and `Press Releases of US companies`. Visit [kay.ai](https://kay.ai) for the latest data drops. For any questions, join our [discord](https://discord.gg/hAnE4e5T6M) or [tweet at us](https://twitter.com/vishalrohra_)

## Installation

First, install the [`kay` package](https://pypi.org/project/kay/).
"""
logger.info("# Kay.ai")

# !pip install kay

"""
You will also need an API key: you can get one for free at [https://kay.ai](https://kay.ai/). Once you have an API key, you must set it as an environment variable `KAY_API_KEY`.

`KayAiRetriever` has a static `.create()` factory method that takes the following arguments:

* `dataset_id: string` required -- A Kay dataset id. This is a collection of data about a particular entity such as companies, people, or places. For example, try `"company"` 
* `data_type: List[string]` optional -- This is a category within a  dataset based on its origin or format, such as ‘SEC Filings’, ‘Press Releases’, or ‘Reports’ within the “company” dataset. For example, try ["10-K", "10-Q", "PressRelease"] under the “company” dataset. If left empty, Kay will retrieve the most relevant context across all types.
* `num_contexts: int` optional, defaults to 6 -- The number of document chunks to retrieve on each call to `get_relevant_documents()`

## Examples

### Basic Retriever Usage
"""
logger.info("## Examples")

# from getpass import getpass

# KAY_API_KEY = getpass()



os.environ["KAY_API_KEY"] = KAY_API_KEY
retriever = KayAiRetriever.create(
    dataset_id="company", data_types=["10-K", "10-Q", "PressRelease"], num_contexts=3
)
docs = retriever.invoke(
    "What were the biggest strategy changes and partnerships made by Roku in 2023??"
)

docs

"""
### Usage in a chain
"""
logger.info("### Usage in a chain")

# OPENAI_API_KEY = getpass()

# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


model = ChatOllama(model="llama3.2")
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)

questions = [
    "What were the biggest strategy changes and partnerships made by Roku in 2023?"
]
chat_history = []

for question in questions:
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    logger.debug(f"-> **Question**: {question} \n")
    logger.debug(f"**Answer**: {result['answer']} \n")

logger.info("\n\n[DONE]", bright=True)