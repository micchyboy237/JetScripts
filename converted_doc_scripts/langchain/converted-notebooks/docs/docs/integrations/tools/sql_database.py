from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain import hub
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.tools.sql_database.tool import (
InfoSQLDatabaseTool,
ListSQLDatabaseTool,
QuerySQLCheckerTool,
QuerySQLDatabaseTool,
)
from langchain_community.utilities.sql_database import SQLDatabase
from langgraph.prebuilt import create_react_agent
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
import ChatModelTabs from "@theme/ChatModelTabs";
import os
import requests
import shutil
import sqlite3


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
# SQLDatabase Toolkit

This will help you get started with the SQL Database [toolkit](/docs/concepts/tools/#toolkits). For detailed documentation of all `SQLDatabaseToolkit` features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/agent_toolkits/langchain_community.agent_toolkits.sql.toolkit.SQLDatabaseToolkit.html).

Tools within the `SQLDatabaseToolkit` are designed to interact with a `SQL` database. 

A common application is to enable agents to answer questions using data in a relational database, potentially in an iterative fashion (e.g., recovering from errors).

**⚠️ Security note ⚠️**

Building Q&A systems of SQL databases requires executing model-generated SQL queries. There are inherent risks in doing this. Make sure that your database connection permissions are always scoped as narrowly as possible for your chain/agent's needs. This will mitigate though not eliminate the risks of building a model-driven system. For more on general security best practices, [see here](/docs/security).

## Setup

To enable automated tracing of individual tools, set your [LangSmith](https://docs.smith.langchain.com/) API key:
"""
logger.info("# SQLDatabase Toolkit")



"""
### Installation

This toolkit lives in the `langchain-community` package:
"""
logger.info("### Installation")

# %pip install --upgrade --quiet  langchain-community

"""
For demonstration purposes, we will access a prompt in the LangChain [Hub](https://smith.langchain.com/hub). We will also require `langgraph` to demonstrate the use of the toolkit with an agent. This is not required to use the toolkit.
"""
logger.info("For demonstration purposes, we will access a prompt in the LangChain [Hub](https://smith.langchain.com/hub). We will also require `langgraph` to demonstrate the use of the toolkit with an agent. This is not required to use the toolkit.")

# %pip install --upgrade --quiet langchainhub langgraph

"""
## Instantiation

The `SQLDatabaseToolkit` toolkit requires:

- a [SQLDatabase](https://python.langchain.com/api_reference/community/utilities/langchain_community.utilities.sql_database.SQLDatabase.html) object;
- a LLM or chat model (for instantiating the [QuerySQLCheckerTool](https://python.langchain.com/api_reference/community/tools/langchain_community.tools.sql_database.tool.QuerySQLCheckerTool.html) tool).

Below, we instantiate the toolkit with these objects. Let's first create a database object.

This guide uses the example `Chinook` database based on [these instructions](https://database.guide/2-sample-databases-sqlite/).

Below we will use the `requests` library to pull the `.sql` file and create an in-memory SQLite database. Note that this approach is lightweight, but ephemeral and not thread-safe. If you'd prefer, you can follow the instructions to save the file locally as `Chinook.db` and instantiate the database via `db = SQLDatabase.from_uri("sqlite:///Chinook.db")`.
"""
logger.info("## Instantiation")




def get_engine_for_chinook_db():
    """Pull sql file, populate in-memory database, and create engine."""
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(url)
    sql_script = response.text

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script)
    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


engine = get_engine_for_chinook_db()

db = SQLDatabase(engine)

"""
We will also need a LLM or chat model:


<ChatModelTabs customVarName="llm" />
"""
logger.info("We will also need a LLM or chat model:")


llm = ChatOllama(model="llama3.2")

"""
We can now instantiate the toolkit:
"""
logger.info("We can now instantiate the toolkit:")


toolkit = SQLDatabaseToolkit(db=db, llm=llm)

"""
## Tools

View available tools:
"""
logger.info("## Tools")

toolkit.get_tools()

"""
You can use the individual tools directly:
"""
logger.info("You can use the individual tools directly:")


"""
## Use within an agent

Following the [SQL Q&A Tutorial](/docs/tutorials/sql_qa/#agents), below we equip a simple question-answering agent with the tools in our toolkit. First we pull a relevant prompt and populate it with its required parameters:
"""
logger.info("## Use within an agent")


prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")

assert len(prompt_template.messages) == 1
logger.debug(prompt_template.input_variables)

system_message = prompt_template.format(dialect="SQLite", top_k=5)

"""
We then instantiate the agent:
"""
logger.info("We then instantiate the agent:")


agent_executor = create_react_agent(llm, toolkit.get_tools(), prompt=system_message)

"""
And issue it a query:
"""
logger.info("And issue it a query:")

example_query = "Which country's customers spent the most?"

events = agent_executor.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_logger.debug()

"""
We can also observe the agent recover from an error:
"""
logger.info("We can also observe the agent recover from an error:")

example_query = "Who are the top 3 best selling artists?"

events = agent_executor.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_logger.debug()

"""
## Specific functionality

`SQLDatabaseToolkit` implements a [.get_context](https://python.langchain.com/api_reference/community/agent_toolkits/langchain_community.agent_toolkits.sql.toolkit.SQLDatabaseToolkit.html#langchain_community.agent_toolkits.sql.toolkit.SQLDatabaseToolkit.get_context) method as a convenience for use in prompts or other contexts.

**⚠️ Disclaimer ⚠️** : The agent may generate insert/update/delete queries. When this is not expected, use a custom prompt or create a SQL users without write permissions.

The final user might overload your SQL database by asking a simple question such as "run the biggest query possible". The generated query might look like:

```sql
SELECT * FROM "public"."users"
    JOIN "public"."user_permissions" ON "public"."users".id = "public"."user_permissions".user_id
    JOIN "public"."projects" ON "public"."users".id = "public"."projects".user_id
    JOIN "public"."events" ON "public"."projects".id = "public"."events".project_id;
```

For a transactional SQL database, if one of the table above contains millions of rows, the query might cause trouble to other applications using the same database.

Most datawarehouse oriented databases support user-level quota, for limiting resource usage.

## API reference

For detailed documentation of all SQLDatabaseToolkit features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/agent_toolkits/langchain_community.agent_toolkits.sql.toolkit.SQLDatabaseToolkit.html).
"""
logger.info("## Specific functionality")

logger.info("\n\n[DONE]", bright=True)