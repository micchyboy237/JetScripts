from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain_community.tools.dataherald.tool import DataheraldTextToSQL
from langchain_community.utilities.dataherald import DataheraldAPIWrapper
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
# Dataherald

>[Dataherald](https://www.dataherald.com) is a natural language-to-SQL.

This page covers how to use the `Dataherald API` within LangChain.

## Installation and Setup
- Install requirements with
"""
logger.info("# Dataherald")

pip install dataherald

"""
- Go to dataherald and sign up [here](https://www.dataherald.com)
- Create an app and get your `API KEY`
- Set your `API KEY` as an environment variable `DATAHERALD_API_KEY`


## Wrappers

### Utility

There exists a DataheraldAPIWrapper utility which wraps this API. To import this utility:
"""
logger.info("## Wrappers")


"""
For a more detailed walkthrough of this wrapper, see [this notebook](/docs/integrations/tools/dataherald).

### Tool

You can use the tool in an agent like this:
"""
logger.info("### Tool")


api_wrapper = DataheraldAPIWrapper(db_connection_id="<db_connection_id>")
tool = DataheraldTextToSQL(api_wrapper=api_wrapper)
llm = ChatOllama(model="llama3.2")
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
agent_executor.invoke({"input":"Return the sql for this question: How many employees are in the company?"})

"""
Output
"""
logger.info("Output")

> Entering new AgentExecutor chain...
I need to use a tool that can convert this question into SQL.
Action: dataherald
Action Input: How many employees are in the company?Answer: SELECT
    COUNT(*) FROM employeesI now know the final answer
Final Answer: SELECT
    COUNT(*)
FROM
    employees

> Finished chain.
{'input': 'Return the sql for this question: How many employees are in the company?', 'output': "SELECT \n    COUNT(*)\nFROM \n    employees"}

"""
For more information on tools, see [this page](/docs/how_to/tools_builtin).
"""
logger.info("For more information on tools, see [this page](/docs/how_to/tools_builtin).")

logger.info("\n\n[DONE]", bright=True)