from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chains import ConversationalRetrievalChain
from langchain_community.retrievers import RememberizerRetriever
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
# Rememberizer

>[Rememberizer](https://rememberizer.ai/) is a knowledge enhancement service for AI applications created by  SkyDeck AI Inc.

This notebook shows how to retrieve documents from `Rememberizer` into the Document format that is used downstream.

# Preparation

You will need an API key: you can get one after creating a common knowledge at [https://rememberizer.ai](https://rememberizer.ai/). Once you have an API key, you must set it as an environment variable `REMEMBERIZER_API_KEY` or pass it as `rememberizer_api_key` when initializing `RememberizerRetriever`.

`RememberizerRetriever` has these arguments:
- optional `top_k_results`: default=10. Use it to limit number of returned documents. 
- optional `rememberizer_api_key`: required if you don't set the environment variable `REMEMBERIZER_API_KEY`.

`get_relevant_documents()` has one argument, `query`: free text which used to find documents in the common knowledge of `Rememberizer.ai`

# Examples

## Basic usage
"""
logger.info("# Rememberizer")

# from getpass import getpass

# REMEMBERIZER_API_KEY = getpass()



os.environ["REMEMBERIZER_API_KEY"] = REMEMBERIZER_API_KEY
retriever = RememberizerRetriever(top_k_results=5)

docs = retriever.get_relevant_documents(query="How does Large Language Models works?")

docs[0].metadata  # meta-information of the Document

logger.debug(docs[0].page_content[:400])  # a content of the Document

"""
# Usage in a chain
"""
logger.info("# Usage in a chain")

# OPENAI_API_KEY = getpass()

# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


model = ChatOllama(model="llama3.2")
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)

questions = [
    "What is RAG?",
    "How does Large Language Models works?",
]
chat_history = []

for question in questions:
    result = qa.invoke({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    logger.debug(f"-> **Question**: {question} \n")
    logger.debug(f"**Answer**: {result['answer']} \n")

logger.info("\n\n[DONE]", bright=True)