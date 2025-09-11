from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Replicate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
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
## LLaMA2 chat with SQL

Open source, local LLMs are great to consider for any application that demands data privacy.

SQL is one good example. 

This cookbook shows how to perform text-to-SQL using various local versions of LLaMA2 run locally.

## Packages
"""
logger.info("## LLaMA2 chat with SQL")

# ! pip install langchain replicate

"""
## LLM

There are a few ways to access LLaMA2.

To run locally, we use Ollama.ai. 

See [here](/docs/integrations/chat/ollama) for details on installation and setup.

Also, see [here](/docs/guides/development/local_llms) for our full guide on local LLMs.
 
To use an external API, which is not private, we can use Replicate.
"""
logger.info("## LLM")


llama2_chat = ChatOllama(model="llama3.2")
llama2_code = ChatOllama(model="llama3.2")


replicate_id = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
llama2_chat_replicate = Replicate(
    model=replicate_id, input={"temperature": 0.01, "max_length": 500, "top_p": 1}
)

llm = llama2_chat

"""
## DB

Connect to a SQLite DB.

To create this particular DB, you can use the code and follow the steps shown [here](https://github.com/facebookresearch/llama-recipes/blob/main/demo_apps/StructuredLlama.ipynb).
"""
logger.info("## DB")


db = SQLDatabase.from_uri("sqlite:///nba_roster.db", sample_rows_in_table_info=0)


def get_schema(_):
    return db.get_table_info()


def run_query(query):
    return db.run(query)

"""
## Query a SQL Database 

Follow the runnables workflow [here](https://python.langchain.com/docs/expression_language/cookbook/sql_db).
"""
logger.info("## Query a SQL Database")


template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Given an input question, convert it to a SQL query. No pre-amble."),
        ("human", template),
    ]
)


sql_response = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

sql_response.invoke({"question": "What team is Klay Thompson on?"})

"""
We can review the results:

* [LangSmith trace](https://smith.langchain.com/public/afa56a06-b4e2-469a-a60f-c1746e75e42b/r) LLaMA2-13 Replicate API
* [LangSmith trace](https://smith.langchain.com/public/2d4ecc72-6b8f-4523-8f0b-ea95c6b54a1d/r) LLaMA2-13 local
"""
logger.info("We can review the results:")

template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""
prompt_response = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input question and SQL response, convert it to a natural language answer. No pre-amble.",
        ),
        ("human", template),
    ]
)

full_chain = (
    RunnablePassthrough.assign(query=sql_response)
    | RunnablePassthrough.assign(
        schema=get_schema,
        response=lambda x: db.run(x["query"]),
    )
    | prompt_response
    | llm
)

full_chain.invoke({"question": "How many unique teams are there?"})

"""
We can review the results:

* [LangSmith trace](https://smith.langchain.com/public/10420721-746a-4806-8ecf-d6dc6399d739/r) LLaMA2-13 Replicate API
* [LangSmith trace](https://smith.langchain.com/public/5265ebab-0a22-4f37-936b-3300f2dfa1c1/r) LLaMA2-13 local

## Chat with a SQL DB 

Next, we can add memory.
"""
logger.info("## Chat with a SQL DB")


template = """Given an input question, convert it to a SQL query. No pre-amble. Based on the table schema below, write a SQL query that would answer the user's question:
{schema}
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

memory = ConversationBufferMemory(return_messages=True)


sql_chain = (
    RunnablePassthrough.assign(
        schema=get_schema,
        history=RunnableLambda(lambda x: memory.load_memory_variables(x)["history"]),
    )
    | prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)


def save(input_output):
    output = {"output": input_output.pop("output")}
    memory.save_context(input_output, output)
    return output["output"]


sql_response_memory = RunnablePassthrough.assign(output=sql_chain) | save
sql_response_memory.invoke({"question": "What team is Klay Thompson on?"})

template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""
prompt_response = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input question and SQL response, convert it to a natural language answer. No pre-amble.",
        ),
        ("human", template),
    ]
)

full_chain = (
    RunnablePassthrough.assign(query=sql_response_memory)
    | RunnablePassthrough.assign(
        schema=get_schema,
        response=lambda x: db.run(x["query"]),
    )
    | prompt_response
    | llm
)

full_chain.invoke({"question": "What is his salary?"})

"""
Here is the [trace](https://smith.langchain.com/public/54794d18-2337-4ce2-8b9f-3d8a2df89e51/r).
"""
logger.info("Here is the [trace](https://smith.langchain.com/public/54794d18-2337-4ce2-8b9f-3d8a2df89e51/r).")

logger.info("\n\n[DONE]", bright=True)