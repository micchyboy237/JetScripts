from jet.adapters.langchain.chat_ollama import ChatOllama, OllamaEmbeddings
from jet.logger import logger
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
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
# How to inspect runnables

:::info Prerequisites

This guide assumes familiarity with the following concepts:
- [LangChain Expression Language (LCEL)](/docs/concepts/lcel)
- [Chaining runnables](/docs/how_to/sequence/)

:::

Once you create a runnable with [LangChain Expression Language](/docs/concepts/lcel), you may often want to inspect it to get a better sense for what is going on. This notebook covers some methods for doing so.

This guide shows some ways you can programmatically introspect the internal steps of chains. If you are instead interested in debugging issues in your chain, see [this section](/docs/how_to/debugging) instead.

First, let's create an example chain. We will create one that does retrieval:
"""
logger.info("# How to inspect runnables")

# %pip install -qU langchain langchain-ollama faiss-cpu tiktoken


vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=OllamaEmbeddings(model="nomic-embed-text")
)
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOllama(model="llama3.2")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

"""
## Get a graph

You can use the `get_graph()` method to get a graph representation of the runnable:
"""
logger.info("## Get a graph")

chain.get_graph()

"""
## Print a graph

While that is not super legible, you can use the `print_ascii()` method to show that graph in a way that's easier to understand:
"""
logger.info("## Print a graph")

chain.get_graph().print_ascii()

"""
## Get the prompts

You may want to see just the prompts that are used in a chain with the `get_prompts()` method:
"""
logger.info("## Get the prompts")

chain.get_prompts()

"""
## Next steps

You've now learned how to introspect your composed LCEL chains.

Next, check out the other how-to guides on runnables in this section, or the related how-to guide on [debugging your chains](/docs/how_to/debugging).
"""
logger.info("## Next steps")


logger.info("\n\n[DONE]", bright=True)
