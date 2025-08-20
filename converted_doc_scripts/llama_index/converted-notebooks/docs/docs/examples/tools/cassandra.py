import asyncio
from jet.transformers.formatters import format_json
from dotenv import load_dotenv
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.tools.cassandra.base import CassandraDatabaseToolSpec
from llama_index.tools.cassandra.cassandra_database_wrapper import (
CassandraDatabase,
)
import cassio
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Cassandra Database Tools

Apache Cassandra® is a widely used database for storing transactional application data. The introduction of functions and tooling in Large Language Models has opened up some exciting use cases for existing data in Generative AI applications. The Cassandra Database toolkit enables AI engineers to efficiently integrate Agents with Cassandra data, offering the following features: 
 - Fast data access through optimized queries. Most queries should run in single-digit ms or less. 
 - Schema introspection to enhance LLM reasoning capabilities 
 - Compatibility with various Cassandra deployments, including Apache Cassandra®, DataStax Enterprise™, and DataStax Astra™ 
 - Currently, the toolkit is limited to SELECT queries and schema introspection operations. (Safety first)

## Quick Start
 - Install the cassio library
 - Set environment variables for the Cassandra database you are connecting to
 - Initialize CassandraDatabase
 - Pass the tools to your agent with spec.to_tool_list()
 - Sit back and watch it do all your work for you

## Theory of Operation
Cassandra Query Language (CQL) is the primary *human-centric* way of interacting with a Cassandra database. While offering some flexibility when generating queries, it requires knowledge of Cassandra data modeling best practices. LLM function calling gives an agent the ability to reason and then choose a tool to satisfy the request. Agents using LLMs should reason using Cassandra-specific logic when choosing the appropriate tool or chain of tools. This reduces the randomness introduced when LLMs are forced to provide a top-down solution. Do you want an LLM to have complete unfettered access to your database? Yeah. Probably not. To accomplish this, we provide a prompt for use when constructing questions for the agent: 

```json
You are an Apache Cassandra expert query analysis bot with the following features 
and rules:
 - You will take a question from the end user about finding specific 
   data in the database.
 - You will examine the schema of the database and create a query path. 
 - You will provide the user with the correct query to find the data they are looking 
   for, showing the steps provided by the query path.
 - You will use best practices for querying Apache Cassandra using partition keys 
   and clustering columns.
 - Avoid using ALLOW FILTERING in the query.
 - The goal is to find a query path, so it may take querying other tables to get 
   to the final answer. 

The following is an example of a query path in JSON format:

 {
  "query_paths": [
    {
      "description": "Direct query to users table using email",
      "steps": [
        {
          "table": "user_credentials",
          "query": 
             "SELECT userid FROM user_credentials WHERE email = 'example@example.com';"
        },
        {
          "table": "users",
          "query": "SELECT * FROM users WHERE userid = ?;"
        }
      ]
    }
  ]
}
```

## Tools Provided

### `cassandra_db_schema`
Gathers all schema information for the connected database or a specific schema. Critical for the agent when determining actions. 

### `cassandra_db_select_table_data`
Selects data from a specific keyspace and table. The agent can pass paramaters for a predicate and limits on the number of returned records. 

### `cassandra_db_query`
Experimental alternative to `cassandra_db_select_table_data` which takes a query string completely formed by the agent instead of parameters. *Warning*: This can lead to unusual queries that may not be as performant(or even work). This may be removed in future releases. If it does something cool, we want to know about that too. You never know!

## Environment Setup

Install the following Python modules:

```bash
pip install ipykernel python-dotenv cassio llama-index llama-index-agent-openai llama-index-llms-ollama llama-index-tools-cassandra
```

### .env file
Connection is via `cassio` using `auto=True` parameter, and the notebook uses MLX. You should create a `.env` file accordingly.

For Cassandra, set:
```bash
CASSANDRA_CONTACT_POINTS
CASSANDRA_USERNAME
CASSANDRA_PASSWORD
CASSANDRA_KEYSPACE
```

For Astra, set:
```bash
ASTRA_DB_APPLICATION_TOKEN
ASTRA_DB_DATABASE_ID
ASTRA_DB_KEYSPACE
```

For example:

```bash
# Connection to Astra:
ASTRA_DB_DATABASE_ID=a1b2c3d4-...
ASTRA_DB_APPLICATION_TOKEN=AstraCS:...
ASTRA_DB_KEYSPACE=notebooks

# Also set 
# OPENAI_API_KEY=sk-....
```

(You may also modify the below code to directly connect with `cassio`.)
"""
logger.info("# Cassandra Database Tools")


load_dotenv(override=True)





"""
## Connect to a Cassandra Database
"""
logger.info("## Connect to a Cassandra Database")

cassio.init(auto=True)

session = cassio.config.resolve_session()
if not session:
    raise Exception(
        "Check environment configuration or manually configure cassio connection parameters"
    )

session = cassio.config.resolve_session()

session.execute("""DROP KEYSPACE IF EXISTS llamaindex_agent_test; """)

session.execute(
    """
CREATE KEYSPACE if not exists llamaindex_agent_test
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};
"""
)

session.execute(
    """
    CREATE TABLE IF NOT EXISTS llamaindex_agent_test.user_credentials (
    user_email text PRIMARY KEY,
    user_id UUID,
    password TEXT
);
"""
)

session.execute(
    """
    CREATE TABLE IF NOT EXISTS llamaindex_agent_test.users (
    id UUID PRIMARY KEY,
    name TEXT,
    email TEXT
);"""
)

session.execute(
    """
    CREATE TABLE IF NOT EXISTS llamaindex_agent_test.user_videos (
    user_id UUID,
    video_id UUID,
    title TEXT,
    description TEXT,
    PRIMARY KEY (user_id, video_id)
);
"""
)

user_id = "522b1fe2-2e36-4cef-a667-cd4237d08b89"
video_id = "27066014-bad7-9f58-5a30-f63fe03718f6"

session.execute(
    f"""
    INSERT INTO llamaindex_agent_test.user_credentials (user_id, user_email)
    VALUES ({user_id}, 'patrick@datastax.com');
"""
)

session.execute(
    f"""
    INSERT INTO llamaindex_agent_test.users (id, name, email)
    VALUES ({user_id}, 'Patrick McFadin', 'patrick@datastax.com');
"""
)

session.execute(
    f"""
    INSERT INTO llamaindex_agent_test.user_videos (user_id, video_id, title)
    VALUES ({user_id}, {video_id}, 'Use Langflow to Build an LLM Application in 5 Minutes');
"""
)

session.set_keyspace("llamaindex_agent_test")

db = CassandraDatabase()

spec = CassandraDatabaseToolSpec(db=db)

tools = spec.to_tool_list()
for tool in tools:
    logger.debug(tool.metadata.name)
    logger.debug(tool.metadata.description)
    logger.debug(tool.metadata.fn_schema)

llm = MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")

agent = FunctionAgent(tools=tools, llm=llm)

"""
### Invoking the agent with tools
We've created an agent that uses an LLM for reasoning and communication with a tool list for actions, Now we can simply ask questions of the agent and watch it utilize the tools we've given it.
"""
logger.info("### Invoking the agent with tools")

async def run_async_code_8ee3b9ac():
    await agent.run("What tables are in the keyspace llamaindex_agent_test?")
    return 
 = asyncio.run(run_async_code_8ee3b9ac())
logger.success(format_json())
async def run_async_code_a169ec08():
    await agent.run("What is the userid for patrick@datastax.com ?")
    return 
 = asyncio.run(run_async_code_a169ec08())
logger.success(format_json())
async def run_async_code_753bc20d():
    await agent.run("What videos did user patrick@datastax.com upload?")
    return 
 = asyncio.run(run_async_code_753bc20d())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)