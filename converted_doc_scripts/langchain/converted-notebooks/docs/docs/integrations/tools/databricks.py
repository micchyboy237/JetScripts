from databricks_langchain import ChatDatabricks
from databricks_langchain.uc_ai import (
DatabricksFunctionClient,
UCFunctionToolkit,
set_uc_function_client,
)
from io import StringIO
from jet.logger import logger
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
import os
import shutil
import sys


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
# Databricks Unity Catalog (UC)

This notebook shows how to use UC functions as LangChain tools, with both LangChain and LangGraph agent APIs.

See Databricks documentation ([AWS](https://docs.databricks.com/en/sql/language-manual/sql-ref-syntax-ddl-create-sql-function.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/sql/language-manual/sql-ref-syntax-ddl-create-sql-function)|[GCP](https://docs.gcp.databricks.com/en/sql/language-manual/sql-ref-syntax-ddl-create-sql-function.html)) to learn how to create SQL or Python functions in UC. Do not skip function and parameter comments, which are critical for LLMs to call functions properly.

In this example notebook, we create a simple Python function that executes arbitrary code and use it as a LangChain tool:

```sql
CREATE FUNCTION main.tools.python_exec (
  code STRING COMMENT 'Python code to execute. Remember to print the final result to stdout.'
)
RETURNS STRING
LANGUAGE PYTHON
COMMENT 'Executes Python code and returns its stdout.'
AS $$
  stdout = StringIO()
  sys.stdout = stdout
  exec(code)
  return stdout.getvalue()
$$
```

It runs in a secure and isolated environment within a Databricks SQL warehouse.
"""
logger.info("# Databricks Unity Catalog (UC)")

# %pip install --upgrade --quiet databricks-sdk langchain-community databricks-langchain langgraph mlflow


llm = ChatDatabricks(endpoint="databricks-meta-llama-3-70b-instruct")


client = DatabricksFunctionClient()
set_uc_function_client(client)

tools = UCFunctionToolkit(
    function_names=["main.tools.python_exec"]
).tools

"""
(Optional) To increase the retry time for getting a function execution response, set environment variable UC_TOOL_CLIENT_EXECUTION_TIMEOUT. Default retry time value is 120s.
## LangGraph agent example
"""
logger.info("## LangGraph agent example")


os.environ["UC_TOOL_CLIENT_EXECUTION_TIMEOUT"] = "200"

"""
## LangGraph agent example
"""
logger.info("## LangGraph agent example")


agent = create_react_agent(
    llm,
    tools,
    prompt="You are a helpful assistant. Make sure to use tool for information.",
)
agent.invoke({"messages": [{"role": "user", "content": "36939 * 8922.4"}]})

"""
## LangChain agent example
"""
logger.info("## LangChain agent example")


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Make sure to use tool for information.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "36939 * 8922.4"})

logger.info("\n\n[DONE]", bright=True)