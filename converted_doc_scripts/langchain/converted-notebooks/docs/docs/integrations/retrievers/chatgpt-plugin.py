from jet.logger import logger
from langchain_community.document_loaders import CSVLoader
from langchain_community.retrievers import (
ChatGPTPluginRetriever,
)
from langchain_core.documents import Document
from typing import List
import json
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
# ChatGPT plugin

>[Ollama plugins](https://platform.ollama.com/docs/plugins/introduction) connect `ChatGPT` to third-party applications. These plugins enable `ChatGPT` to interact with APIs defined by developers, enhancing `ChatGPT's` capabilities and allowing it to perform a wide range of actions.

>Plugins allow `ChatGPT` to do things like:
>- Retrieve real-time information; e.g., sports scores, stock prices, the latest news, etc.
>- Retrieve knowledge-base information; e.g., company docs, personal notes, etc.
>- Perform actions on behalf of the user; e.g., booking a flight, ordering food, etc.

This notebook shows how to use the ChatGPT Retriever Plugin within LangChain.
"""
logger.info("# ChatGPT plugin")


loader = CSVLoader(
    file_path="../../document_loaders/examples/example_data/mlb_teams_2012.csv"
)
data = loader.load()





def write_json(path: str, documents: List[Document]) -> None:
    results = [{"text": doc.page_content} for doc in documents]
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


write_json("foo.json", data)

"""
## Using the ChatGPT Retriever Plugin

Okay, so we've created the ChatGPT Retriever Plugin, but how do we actually use it?

The below code walks through how to do that.

We want to use `ChatGPTPluginRetriever` so we have to get the Ollama API Key.
"""
logger.info("## Using the ChatGPT Retriever Plugin")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")


retriever = ChatGPTPluginRetriever(url="http://0.0.0.0:8000", bearer_token="foo")

retriever.invoke("alice's phone number")

logger.info("\n\n[DONE]", bright=True)