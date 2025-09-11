from jet.logger import logger
from langchain_cohere import ChatCohere, CohereRagRetriever
from langchain_core.documents import Document
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
# Cohere RAG

>[Cohere](https://cohere.ai/about) is a Canadian startup that provides natural language processing models that help companies improve human-machine interactions.

This notebook covers how to get started with the `Cohere RAG` retriever. This allows you to leverage the ability to search documents over various connectors or by supplying your own.
"""
logger.info("# Cohere RAG")

# import getpass

# os.environ["COHERE_API_KEY"] = getpass.getpass()


rag = CohereRagRetriever(llm=ChatCohere())

def _pretty_logger.debug(docs):
    for doc in docs:
        logger.debug(doc.metadata)
        logger.debug("\n\n" + doc.page_content)
        logger.debug("\n\n" + "-" * 30 + "\n\n")

_pretty_logger.debug(rag.invoke("What is cohere ai?"))

_pretty_logger.debug(await rag.ainvoke("What is cohere ai?"))  # async version

docs = rag.invoke(
    "Does langchain support cohere RAG?",
    documents=[
        Document(page_content="Langchain supports cohere RAG!"),
        Document(page_content="The sky is blue!"),
    ],
)
_pretty_logger.debug(docs)

"""
Please note that connectors and documents cannot be used simultaneously. If you choose to provide documents in the `invoke` method, they will take precedence, and connectors will not be utilized for that particular request, as shown in the snippet above!
"""
logger.info("Please note that connectors and documents cannot be used simultaneously. If you choose to provide documents in the `invoke` method, they will take precedence, and connectors will not be utilized for that particular request, as shown in the snippet above!")


logger.info("\n\n[DONE]", bright=True)