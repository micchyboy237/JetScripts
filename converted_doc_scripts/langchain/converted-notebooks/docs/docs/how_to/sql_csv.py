from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.ollama_tools import JsonOutputKeyToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonAstREPLTool
from operator import itemgetter
from sqlalchemy import create_engine
import ChatModelTabs from "@theme/ChatModelTabs";
import os
import pandas as pd
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
# How to do question answering over CSVs

LLMs are great for building question-answering systems over various types of data sources. In this section we'll go over how to build Q&A systems over data stored in a CSV file(s). Like working with SQL databases, the key to working with CSV files is to give an LLM access to tools for querying and interacting with the data. The two main ways to do this are to either:

* **RECOMMENDED**: Load the CSV(s) into a SQL database, and use the approaches outlined in the [SQL tutorial](/docs/tutorials/sql_qa).
* Give the LLM access to a Python environment where it can use libraries like Pandas to interact with the data.

We will cover both approaches in this guide.

## ⚠️ Security note ⚠️

Both approaches mentioned above carry significant risks. Using SQL requires executing model-generated SQL queries. Using a library like Pandas requires letting the model execute Python code. Since it is easier to tightly scope SQL connection permissions and sanitize SQL queries than it is to sandbox Python environments, **we HIGHLY recommend interacting with CSV data via SQL.** For more on general security best practices, [see here](/docs/security).

## Setup
Dependencies for this guide:
"""
logger.info("# How to do question answering over CSVs")

# %pip install -qU langchain langchain-ollama langchain-community langchain-experimental pandas

"""
Set required environment variables:
"""
logger.info("Set required environment variables:")



"""
Download the [Titanic dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset) if you don't already have it:
"""
logger.info("Download the [Titanic dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset) if you don't already have it:")

# !wget https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv -O titanic.csv


df = pd.read_csv("titanic.csv")
logger.debug(df.shape)
logger.debug(df.columns.tolist())

"""
## SQL

Using SQL to interact with CSV data is the recommended approach because it is easier to limit permissions and sanitize queries than with arbitrary Python.

Most SQL databases make it easy to load a CSV file in as a table ([DuckDB](https://duckdb.org/docs/data/csv/overview.html), [SQLite](https://www.sqlite.org/csv.html), etc.). Once you've done this you can use all of the chain and agent-creating techniques outlined in the [SQL tutorial](/docs/tutorials/sql_qa). Here's a quick example of how we might do this with SQLite:
"""
logger.info("## SQL")


engine = create_engine("sqlite:///titanic.db")
df.to_sql("titanic", engine, index=False)

db = SQLDatabase(engine=engine)
logger.debug(db.dialect)
logger.debug(db.get_usable_table_names())
logger.debug(db.run("SELECT * FROM titanic WHERE Age < 2;"))

"""
And create a [SQL agent](/docs/tutorials/sql_qa) to interact with it:


<ChatModelTabs customVarName="llm" />
"""
logger.info("And create a [SQL agent](/docs/tutorials/sql_qa) to interact with it:")


llm = ChatOllama(model="llama3.2")


agent_executor = create_sql_agent(llm, db=db, agent_type="ollama-tools", verbose=True)

agent_executor.invoke({"input": "what's the average age of survivors"})

"""
This approach easily generalizes to multiple CSVs, since we can just load each of them into our database as its own table. See the [Multiple CSVs](/docs/how_to/sql_csv#multiple-csvs) section below.

## Pandas

Instead of SQL we can also use data analysis libraries like pandas and the code generating abilities of LLMs to interact with CSV data. Again, **this approach is not fit for production use cases unless you have extensive safeguards in place**. For this reason, our code-execution utilities and constructors live in the `langchain-experimental` package.

### Chain

Most LLMs have been trained on enough pandas Python code that they can generate it just by being asked to:
"""
logger.info("## Pandas")

ai_msg = llm.invoke(
    "I have a pandas DataFrame 'df' with columns 'Age' and 'Fare'. Write code to compute the correlation between the two columns. Return Markdown for a Python code snippet and nothing else."
)
logger.debug(ai_msg.content)

"""
We can combine this ability with a Python-executing tool to create a simple data analysis chain. We'll first want to load our CSV table as a dataframe, and give the tool access to this dataframe:
"""
logger.info("We can combine this ability with a Python-executing tool to create a simple data analysis chain. We'll first want to load our CSV table as a dataframe, and give the tool access to this dataframe:")


df = pd.read_csv("titanic.csv")
tool = PythonAstREPLTool(locals={"df": df})
tool.invoke("df['Fare'].mean()")

"""
To help enforce proper use of our Python tool, we'll using [tool calling](/docs/how_to/tool_calling):
"""
logger.info("To help enforce proper use of our Python tool, we'll using [tool calling](/docs/how_to/tool_calling):")

llm_with_tools = llm.bind_tools([tool], tool_choice=tool.name)
response = llm_with_tools.invoke(
    "I have a dataframe 'df' and want to know the correlation between the 'Age' and 'Fare' columns"
)
response

response.tool_calls

"""
We'll add a tools output parser to extract the function call as a dict:
"""
logger.info("We'll add a tools output parser to extract the function call as a dict:")


parser = JsonOutputKeyToolsParser(key_name=tool.name, first_tool_only=True)
(llm_with_tools | parser).invoke(
    "I have a dataframe 'df' and want to know the correlation between the 'Age' and 'Fare' columns"
)

"""
And combine with a prompt so that we can just specify a question without needing to specify the dataframe info every invocation:
"""
logger.info("And combine with a prompt so that we can just specify a question without needing to specify the dataframe info every invocation:")

system = f"""You have access to a pandas dataframe `df`. \
Here is the output of `df.head().to_markdown()`:

```
{df.head().to_markdown()}
```

Given a user question, write the Python code to answer it. \
Return ONLY the valid Python code and nothing else. \
Don't assume you have access to any libraries other than built-in Python ones and pandas."""
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])
code_chain = prompt | llm_with_tools | parser
code_chain.invoke({"question": "What's the correlation between age and fare"})

"""
And lastly we'll add our Python tool so that the generated code is actually executed:
"""
logger.info("And lastly we'll add our Python tool so that the generated code is actually executed:")

chain = prompt | llm_with_tools | parser | tool
chain.invoke({"question": "What's the correlation between age and fare"})

"""
And just like that we have a simple data analysis chain. We can take a peak at the intermediate steps by looking at the LangSmith trace: https://smith.langchain.com/public/b1309290-7212-49b7-bde2-75b39a32b49a/r

We could add an additional LLM call at the end to generate a conversational response, so that we're not just responding with the tool output. For this we'll want to add a chat history `MessagesPlaceholder` to our prompt:
"""
logger.info("And just like that we have a simple data analysis chain. We can take a peak at the intermediate steps by looking at the LangSmith trace: https://smith.langchain.com/public/b1309290-7212-49b7-bde2-75b39a32b49a/r")



system = f"""You have access to a pandas dataframe `df`. \
Here is the output of `df.head().to_markdown()`:

```
{df.head().to_markdown()}
```

Given a user question, write the Python code to answer it. \
Don't assume you have access to any libraries other than built-in Python ones and pandas.
Respond directly to the question once you have enough information to answer it."""
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system,
        ),
        ("human", "{question}"),
        MessagesPlaceholder("chat_history", optional=True),
    ]
)


def _get_chat_history(x: dict) -> list:
    """Parse the chain output up to this point into a list of chat history messages to insert in the prompt."""
    ai_msg = x["ai_msg"]
    tool_call_id = x["ai_msg"].additional_kwargs["tool_calls"][0]["id"]
    tool_msg = ToolMessage(tool_call_id=tool_call_id, content=str(x["tool_output"]))
    return [ai_msg, tool_msg]


chain = (
    RunnablePassthrough.assign(ai_msg=prompt | llm_with_tools)
    .assign(tool_output=itemgetter("ai_msg") | parser | tool)
    .assign(chat_history=_get_chat_history)
    .assign(response=prompt | llm | StrOutputParser())
    .pick(["tool_output", "response"])
)

chain.invoke({"question": "What's the correlation between age and fare"})

"""
Here's the LangSmith trace for this run: https://smith.langchain.com/public/14e38d70-45b1-4b81-8477-9fd2b7c07ea6/r

### Agent

For complex questions it can be helpful for an LLM to be able to iteratively execute code while maintaining the inputs and outputs of its previous executions. This is where Agents come into play. They allow an LLM to decide how many times a tool needs to be invoked and keep track of the executions it's made so far. The [create_pandas_dataframe_agent](https://python.langchain.com/api_reference/experimental/agents/langchain_experimental.agents.agent_toolkits.pandas.base.create_pandas_dataframe_agent.html) is a built-in agent that makes it easy to work with dataframes:
"""
logger.info("### Agent")


agent = create_pandas_dataframe_agent(
    llm, df, agent_type="ollama-tools", verbose=True, allow_dangerous_code=True
)
agent.invoke(
    {
        "input": "What's the correlation between age and fare? is that greater than the correlation between fare and survival?"
    }
)

"""
Here's the LangSmith trace for this run: https://smith.langchain.com/public/6a86aee2-4f22-474a-9264-bd4c7283e665/r

### Multiple CSVs {#multiple-csvs}

To handle multiple CSVs (or dataframes) we just need to pass multiple dataframes to our Python tool. Our `create_pandas_dataframe_agent` constructor can do this out of the box, we can pass in a list of dataframes instead of just one. If we're constructing a chain ourselves, we can do something like:
"""
logger.info("### Multiple CSVs {#multiple-csvs}")

df_1 = df[["Age", "Fare"]]
df_2 = df[["Fare", "Survived"]]

tool = PythonAstREPLTool(locals={"df_1": df_1, "df_2": df_2})
llm_with_tool = llm.bind_tools(tools=[tool], tool_choice=tool.name)
df_template = """```python
{df_name}.head().to_markdown()
>>> {df_head}
```"""
df_context = "\n\n".join(
    df_template.format(df_head=_df.head().to_markdown(), df_name=df_name)
    for _df, df_name in [(df_1, "df_1"), (df_2, "df_2")]
)

system = f"""You have access to a number of pandas dataframes. \
Here is a sample of rows from each dataframe and the python code that was used to generate the sample:

{df_context}

Given a user question about the dataframes, write the Python code to answer it. \
Don't assume you have access to any libraries other than built-in Python ones and pandas. \
Make sure to refer only to the variables mentioned above."""
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])

chain = prompt | llm_with_tool | parser | tool
chain.invoke(
    {
        "question": "return the difference in the correlation between age and fare and the correlation between fare and survival"
    }
)

"""
Here's the LangSmith trace for this run: https://smith.langchain.com/public/cc2a7d7f-7c5a-4e77-a10c-7b5420fcd07f/r

### Sandboxed code execution

There are a number of tools like [E2B](/docs/integrations/tools/e2b_data_analysis) and [Bearly](/docs/integrations/tools/bearly) that provide sandboxed environments for Python code execution, to allow for safer code-executing chains and agents.

## Next steps

For more advanced data analysis applications we recommend checking out:

* [SQL tutorial](/docs/tutorials/sql_qa): Many of the challenges of working with SQL db's and CSV's are generic to any structured data type, so it's useful to read the SQL techniques even if you're using Pandas for CSV data analysis.
* [Tool use](/docs/how_to/tool_calling): Guides on general best practices when working with chains and agents that invoke tools
* [Agents](/docs/tutorials/agents): Understand the fundamentals of building LLM agents.
* Integrations: Sandboxed envs like [E2B](/docs/integrations/tools/e2b_data_analysis) and [Bearly](/docs/integrations/tools/bearly), utilities like [SQLDatabase](https://python.langchain.com/api_reference/community/utilities/langchain_community.utilities.sql_database.SQLDatabase.html#langchain_community.utilities.sql_database.SQLDatabase), related agents like [Spark DataFrame agent](/docs/integrations/tools/spark_sql).
"""
logger.info("### Sandboxed code execution")

logger.info("\n\n[DONE]", bright=True)