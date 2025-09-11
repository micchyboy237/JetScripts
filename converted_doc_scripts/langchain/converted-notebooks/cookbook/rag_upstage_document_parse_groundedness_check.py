from jet.logger import logger
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableSerializable
from langchain_upstage import (
ChatUpstage,
UpstageDocumentParseLoader,
UpstageEmbeddings,
UpstageGroundednessCheck,
)
from typing import List
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
# RAG using Upstage Document Parse and Groundedness Check
This example illustrates RAG using [Upstage](https://python.langchain.com/docs/integrations/providers/upstage/) Document Parse and Groundedness Check.
"""
logger.info("# RAG using Upstage Document Parse and Groundedness Check")



model = ChatUpstage()

files = ["/PATH/TO/YOUR/FILE.pdf", "/PATH/TO/YOUR/FILE2.pdf"]

loader = UpstageDocumentParseLoader(file_path=files, split="element")

docs = loader.load()

vectorstore = DocArrayInMemorySearch.from_documents(
    docs, embedding=UpstageEmbeddings(model="solar-embedding-1-large")
)
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

retrieved_docs = retriever.get_relevant_documents("How many parameters in SOLAR model?")

groundedness_check = UpstageGroundednessCheck()
groundedness = ""
while groundedness != "grounded":
    chain: RunnableSerializable = RunnablePassthrough() | prompt | model | output_parser

    result = chain.invoke(
        {
            "context": retrieved_docs,
            "question": "How many parameters in SOLAR model?",
        }
    )

    groundedness = groundedness_check.invoke(
        {
            "context": retrieved_docs,
            "answer": result,
        }
    )

logger.info("\n\n[DONE]", bright=True)