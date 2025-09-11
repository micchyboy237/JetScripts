from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.retrievers.you import YouRetriever
from langchain_community.utilities import YouSearchAPIWrapper
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
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
# You.com

>[you.com API](https://api.you.com) is a suite of tools designed to help developers ground the output of LLMs in the most recent, most accurate, most relevant information that may not have been included in their training dataset.

## Setup

The retriever lives in the `langchain-community` package.

You also need to set your you.com API key.
"""
logger.info("# You.com")

# %pip install --upgrade --quiet langchain-community


os.environ["YDC_API_KEY"] = ""

# os.environ["OPENAI_API_KEY"] = ""

"""
It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability
"""
logger.info("It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability")



"""
## Utility Usage
"""
logger.info("## Utility Usage")


utility = YouSearchAPIWrapper(num_web_results=1)

utility


response = utility.raw_results(query="What is the weather in NY")
hits = response["hits"]

logger.debug(len(hits))

logger.debug(json.dumps(hits, indent=2))

response = utility.results(query="What is the weather in NY")

logger.debug(len(response))

logger.debug(response)

"""
## Retriever Usage
"""
logger.info("## Retriever Usage")


retriever = YouRetriever(num_web_results=1)

retriever

response = retriever.invoke("What is the weather in NY")

logger.debug(len(response))

logger.debug(response)

"""
## Chaining
"""
logger.info("## Chaining")

# !pip install --upgrade --quiet langchain-ollama


runnable = RunnablePassthrough

retriever = YouRetriever(num_web_results=1)

model = ChatOllama(model="llama3.2")

output_parser = StrOutputParser()

"""
### Invoke
"""
logger.info("### Invoke")

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided.

Context: {context}

Question: {question}"""
)

chain = (
    runnable.assign(context=(lambda x: x["question"]) | retriever)
    | prompt
    | model
    | output_parser
)

output = chain.invoke({"question": "what is the weather in NY today"})

logger.debug(output)

"""
### Stream
"""
logger.info("### Stream")

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided.

Context: {context}

Question: {question}"""
)

chain = (
    runnable.assign(context=(lambda x: x["question"]) | retriever)
    | prompt
    | model
    | output_parser
)

for s in chain.stream({"question": "what is the weather in NY today"}):
    logger.debug(s, end="", flush=True)

"""
### Batch
"""
logger.info("### Batch")

chain = (
    runnable.assign(context=(lambda x: x["question"]) | retriever)
    | prompt
    | model
    | output_parser
)

output = chain.batch(
    [
        {"question": "what is the weather in NY today"},
        {"question": "what is the weather in sf today"},
    ]
)

for o in output:
    logger.debug(o)

logger.info("\n\n[DONE]", bright=True)