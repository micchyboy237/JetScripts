from faker import Faker
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field, model_validator
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
sidebar_position: 7
---

# How to deal with high-cardinality categoricals when doing query analysis

You may want to do query analysis to create a filter on a categorical column. One of the difficulties here is that you usually need to specify the EXACT categorical value. The issue is you need to make sure the LLM generates that categorical value exactly. This can be done relatively easy with prompting when there are only a few values that are valid. When there are a high number of valid values then it becomes more difficult, as those values may not fit in the LLM context, or (if they do) there may be too many for the LLM to properly attend to.

In this notebook we take a look at how to approach this.

## Setup
#### Install dependencies
"""
logger.info(
    "# How to deal with high-cardinality categoricals when doing query analysis")

# %pip install -qU langchain langchain-community langchain-ollama faker langchain-chroma

"""
#### Set environment variables

We'll use Ollama in this example:
"""
logger.info("#### Set environment variables")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass()

"""
#### Set up data

We will generate a bunch of fake names
"""
logger.info("#### Set up data")


fake = Faker()

names = [fake.name() for _ in range(10000)]

"""
Let's look at some of the names
"""
logger.info("Let's look at some of the names")

names[0]

names[567]

"""
## Query Analysis

We can now set up a baseline query analysis
"""
logger.info("## Query Analysis")


class Search(BaseModel):
    query: str
    author: str


system = """Generate a relevant search query for a library system"""
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
We can see that if we spell the name exactly correctly, it knows how to handle it
"""
logger.info(
    "We can see that if we spell the name exactly correctly, it knows how to handle it")

query_analyzer.invoke("what are books about aliens by Jesse Knight")

"""
The issue is that the values you want to filter on may NOT be spelled exactly correctly
"""
logger.info(
    "The issue is that the values you want to filter on may NOT be spelled exactly correctly")

query_analyzer.invoke("what are books about aliens by jess knight")

"""
### Add in all values

One way around this is to add ALL possible values to the prompt. That will generally guide the query in the right direction
"""
logger.info("### Add in all values")

system = """Generate a relevant search query for a library system.

`author` attribute MUST be one of:

{authors}

Do NOT hallucinate author name!"""
base_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
prompt = base_prompt.partial(authors=", ".join(names))

query_analyzer_all = {
    "question": RunnablePassthrough()} | prompt | structured_llm

"""
However... if the list of categoricals is long enough, it may error!
"""
logger.info(
    "However... if the list of categoricals is long enough, it may error!")

try:
    res = query_analyzer_all.invoke(
        "what are books about aliens by jess knight")
except Exception as e:
    logger.debug(e)

"""
We can try to use a longer context window... but with so much information in there, it is not garunteed to pick it up reliably
"""
logger.info("We can try to use a longer context window... but with so much information in there, it is not garunteed to pick it up reliably")

llm_long = ChatOllama(model="llama3.2")
structured_llm_long = llm_long.with_structured_output(Search)
query_analyzer_all = {
    "question": RunnablePassthrough()} | prompt | structured_llm_long

query_analyzer_all.invoke("what are books about aliens by jess knight")

"""
### Find and all relevant values

Instead, what we can do is create an index over the relevant values and then query that for the N most relevant values,
"""
logger.info("### Find and all relevant values")


embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_texts(
    names, embeddings, collection_name="author_names")


def select_names(question):
    _docs = vectorstore.similarity_search(question, k=10)
    _names = [d.page_content for d in _docs]
    return ", ".join(_names)


create_prompt = {
    "question": RunnablePassthrough(),
    "authors": select_names,
} | base_prompt

query_analyzer_select = create_prompt | structured_llm

create_prompt.invoke("what are books by jess knight")

query_analyzer_select.invoke("what are books about aliens by jess knight")

"""
### Replace after selection

Another method is to let the LLM fill in whatever value, but then convert that value to a valid value.
This can actually be done with the Pydantic class itself!
"""
logger.info("### Replace after selection")


class Search(BaseModel):
    query: str
    author: str

    @model_validator(mode="before")
    @classmethod
    def double(cls, values: dict) -> dict:
        author = values["author"]
        closest_valid_author = vectorstore.similarity_search(author, k=1)[
            0
        ].page_content
        values["author"] = closest_valid_author
        return values


system = """Generate a relevant search query for a library system"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
corrective_structure_llm = llm.with_structured_output(Search)
corrective_query_analyzer = (
    {"question": RunnablePassthrough()} | prompt | corrective_structure_llm
)

corrective_query_analyzer.invoke("what are books about aliens by jes knight")

logger.info("\n\n[DONE]", bright=True)
