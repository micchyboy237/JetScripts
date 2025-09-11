from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_chroma import Chroma
from langchain_core.output_parsers.ollama_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import Optional
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
sidebar_position: 3
---

# How to handle cases where no queries are generated

Sometimes, a query analysis technique may allow for any number of queries to be generated - including no queries! In this case, our overall chain will need to inspect the result of the query analysis before deciding whether to call the retriever or not.

We will use mock data for this example.

## Setup
#### Install dependencies
"""
logger.info("# How to handle cases where no queries are generated")

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
vectorstore = Chroma.from_texts(
    texts,
    embeddings,
)
retriever = vectorstore.as_retriever()

"""
## Query analysis

We will use function calling to structure the output. However, we will configure the LLM such that is doesn't NEED to call the function representing a search query (should it decide not to). We will also then use a prompt to do query analysis that explicitly lays when it should and shouldn't make a search.
"""
logger.info("## Query analysis")


class Search(BaseModel):
    """Search over a database of job records."""

    query: str = Field(
        ...,
        description="Similarity search query applied to job record.",
    )


system = """You have the ability to issue search queries to get information to help answer user information.

You do not NEED to look things up. If you don't need to, then just respond normally."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
llm = ChatOllama(model="llama3.2")
structured_llm = llm.bind_tools([Search])
query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm

"""
We can see that by invoking this we get an message that sometimes - but not always - returns a tool call.
"""
logger.info("We can see that by invoking this we get an message that sometimes - but not always - returns a tool call.")

query_analyzer.invoke("where did Harrison Work")

query_analyzer.invoke("hi!")

"""
## Retrieval with query analysis

So how would we include this in a chain? Let's look at an example below.
"""
logger.info("## Retrieval with query analysis")


output_parser = PydanticToolsParser(tools=[Search])


@chain
def custom_chain(question):
    response = query_analyzer.invoke(question)
    if "tool_calls" in response.additional_kwargs:
        query = output_parser.invoke(response)
        docs = retriever.invoke(query[0].query)
        return docs
    else:
        return response


custom_chain.invoke("where did Harrison Work")

custom_chain.invoke("hi!")

logger.info("\n\n[DONE]", bright=True)
