from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain import hub
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
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
# Rewrite-Retrieve-Read

**Rewrite-Retrieve-Read** is a method proposed in the paper [Query Rewriting for Retrieval-Augmented Large Language Models](https://arxiv.org/pdf/2305.14283.pdf)

> Because the original query can not be always optimal to retrieve for the LLM, especially in the real world... we first prompt an LLM to rewrite the queries, then conduct retrieval-augmented reading

We show how you can easily do that with LangChain Expression Language

## Baseline

Baseline RAG (**Retrieve-and-read**) can be done like the following:
"""
logger.info("# Rewrite-Retrieve-Read")


template = """Answer the users question based only on the following context:

<context>
{context}
</context>

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOllama(model="llama3.2")

search = DuckDuckGoSearchAPIWrapper()


def retriever(query):
    return search.run(query)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

simple_query = "what is langchain?"

chain.invoke(simple_query)

"""
While this is fine for well formatted queries, it can break down for more complicated queries
"""
logger.info("While this is fine for well formatted queries, it can break down for more complicated queries")

distracted_query = "man that sam bankman fried trial was crazy! what is langchain?"

chain.invoke(distracted_query)

"""
This is because the retriever does a bad job with these "distracted" queries
"""
logger.info("This is because the retriever does a bad job with these "distracted" queries")

retriever(distracted_query)

"""
## Rewrite-Retrieve-Read Implementation

The main part is a rewriter to rewrite the search query
"""
logger.info("## Rewrite-Retrieve-Read Implementation")

template = """Provide a better search query for \
web search engine to answer the given question, end \
the queries with ’**’. Question: \
{x} Answer:"""
rewrite_prompt = ChatPromptTemplate.from_template(template)


rewrite_prompt = hub.pull("langchain-ai/rewrite")

logger.debug(rewrite_prompt.template)

def _parse(text):
    return text.strip('"').strip("**")

rewriter = rewrite_prompt | ChatOllama(model="llama3.2") | StrOutputParser() | _parse

rewriter.invoke({"x": distracted_query})

rewrite_retrieve_read_chain = (
    {
        "context": {"x": RunnablePassthrough()} | rewriter | retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)

rewrite_retrieve_read_chain.invoke(distracted_query)

logger.info("\n\n[DONE]", bright=True)