from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chains import ConversationalRetrievalChain
from langchain_community.retrievers import OutlineRetriever
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
# Outline

>[Outline](https://www.getoutline.com/) is an open-source collaborative knowledge base platform designed for team information sharing.

This notebook shows how to retrieve documents from your Outline instance into the Document format that is used downstream.

## Setup
"""
logger.info("# Outline")

# %pip install --upgrade --quiet langchain langchain-ollama

"""
You first need to [create an api key](https://www.getoutline.com/developers#section/Authentication) for your Outline instance. Then you need to set the following environment variables:
"""
logger.info("You first need to [create an api key](https://www.getoutline.com/developers#section/Authentication) for your Outline instance. Then you need to set the following environment variables:")


os.environ["OUTLINE_API_KEY"] = "xxx"
os.environ["OUTLINE_INSTANCE_URL"] = "https://app.getoutline.com"

"""
`OutlineRetriever` has these arguments:
- optional `top_k_results`: default=3. Use it to limit number of documents retrieved.
- optional `load_all_available_meta`: default=False. By default only the most important fields retrieved: `title`, `source` (the url of the document). If True, other fields also retrieved.
- optional `doc_content_chars_max` default=4000. Use it to limit the number of characters for each document retrieved.

`get_relevant_documents()` has one argument, `query`: free text which used to find documents in your Outline instance.

## Examples

### Running retriever
"""
logger.info("## Examples")


retriever = OutlineRetriever()

retriever.invoke("LangChain", doc_content_chars_max=100)

"""
### Answering Questions on Outline Documents
"""
logger.info("### Answering Questions on Outline Documents")

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass("Ollama API Key:")


model = ChatOllama(model="llama3.2")
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)

qa({"question": "what is langchain?", "chat_history": {}})

logger.info("\n\n[DONE]", bright=True)