from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain_chroma import Chroma
from langchain_core.output_parsers.ollama_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List, Optional
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
sidebar_position: 5
---

# How to handle multiple retrievers when doing query analysis

Sometimes, a query analysis technique may allow for selection of which [retriever](/docs/concepts/retrievers/) to use. To use this, you will need to add some logic to select the retriever to do. We will show a simple example (using mock data) of how to do that.

## Setup
#### Install dependencies
"""
logger.info("# How to handle multiple retrievers when doing query analysis")

# %pip install -qU langchain langchain-community langchain-ollama langchain-chroma

"""
#### Set environment variables

We'll use Ollama in this example:
"""
logger.info("#### Set environment variables")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass()

"""
### Create Index

We will create a vectorstore over fake information.
"""
logger.info("### Create Index")


texts = ["Harrison worked at Kensho"]
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = Chroma.from_texts(texts, embeddings, collection_name="harrison")
retriever_harrison = vectorstore.as_retriever(search_kwargs={"k": 1})

texts = ["Ankush worked at Facebook"]
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = Chroma.from_texts(texts, embeddings, collection_name="ankush")
retriever_ankush = vectorstore.as_retriever(search_kwargs={"k": 1})

"""
## Query analysis

We will use function calling to structure the output. We will let it return multiple queries.
"""
logger.info("## Query analysis")




class Search(BaseModel):
    """Search for information about a person."""

    query: str = Field(
        ...,
        description="Query to look up",
    )
    person: str = Field(
        ...,
        description="Person to look things up for. Should be `HARRISON` or `ANKUSH`.",
    )


output_parser = PydanticToolsParser(tools=[Search])

system = """You have the ability to issue search queries to get information to help answer user information."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
llm = ChatOllama(model="llama3.2")
structured_llm = llm.with_structured_output(Search)
query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm

"""
We can see that this allows for routing between retrievers
"""
logger.info("We can see that this allows for routing between retrievers")

query_analyzer.invoke("where did Harrison Work")

query_analyzer.invoke("where did ankush Work")

"""
## Retrieval with query analysis

So how would we include this in a chain? We just need some simple logic to select the retriever and pass in the search query
"""
logger.info("## Retrieval with query analysis")


retrievers = {
    "HARRISON": retriever_harrison,
    "ANKUSH": retriever_ankush,
}

@chain
def custom_chain(question):
    response = query_analyzer.invoke(question)
    retriever = retrievers[response.person]
    return retriever.invoke(response.query)

custom_chain.invoke("where did Harrison Work")

custom_chain.invoke("where did ankush Work")

logger.info("\n\n[DONE]", bright=True)