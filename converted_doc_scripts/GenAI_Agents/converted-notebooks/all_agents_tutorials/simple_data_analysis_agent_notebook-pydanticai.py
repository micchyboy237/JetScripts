from dataclasses import dataclass
from datetime import datetime, timedelta
from dotenv import load_dotenv
from jet.logger import CustomLogger
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.messages import Message, MessagesTypeAdapter
from pydantic_ai.result import RunResult
from typing import Any
import numpy as np
import os
import pandas as pd
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# Data Analysis Simple Agent with PydanticAI

**This tutorial is based on the LangChain tutorial: `Data Analysis Simple Agent`. It demonstrates the same concept using PydanticAI as the agent framework.**

**You don’t need to be familiar with the LangChain notebook to follow along—this tutorial stands on its own and explains everything you need to know.** For more information about PydanticAI, visit their [official website](https://ai.pydantic.dev/), or check out the PydanticAI Overview in [this notebook](https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/simple_conversational_agent-pydanticai.ipynb).

In this version of the notebook, we replicate the [Data Analysis Simple Agent](https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/simple_data_analysis_agent_notebook.ipynb) workflow using **PydanticAI**. The primary difference is that LangChain includes a built-in agent designed to handle Pandas DataFrames and perform actions on them directly. PydanticAI, being a newer framework, does not yet include such a built-in tool. As a result, we’ll create the tool ourselves, providing an opportunity to explore how to build custom tools and implement retry logic with PydanticAI.

## Overview
This tutorial guides you through creating an AI-powered data analysis agent that can interpret and answer questions about a dataset using natural language. It combines language models with data manipulation tools to enable intuitive data exploration.

## Motivation
Data analysis often requires specialized knowledge, limiting access to insights for non-technical users. By creating an AI agent that understands natural language queries, we can democratize data analysis, allowing anyone to extract valuable information from complex datasets without needing to know programming or statistical tools.

## Key Components
1. Language Model: Processes natural language queries and generates human-like responses
2. Data Manipulation Framework: Handles dataset operations and analysis
3. Agent Framework: Connects the language model with data manipulation tools
4. Synthetic Dataset: Represents real-world data for demonstration purposes

## Method
1. Create a synthetic dataset representing car sales data
2. Construct an agent that combines the language model with data analysis capabilities
3. Create a tool that the agent can use to query our dataset.
4. Implement a query processing function to handle natural language questions
5. Demonstrate the agent's abilities with example queries

## Conclusion
This approach to data analysis offers significant benefits:
- Accessibility for non-technical users
- Flexibility in handling various query types
- Efficient ad-hoc data exploration

By making data insights more accessible, this method has the potential to transform how organizations leverage their data for decision-making across various fields and industries.

## Import libraries and set environment variables
"""
logger.info("# Data Analysis Simple Agent with PydanticAI")






np.random.seed(42)

# import nest_asyncio
# nest_asyncio.apply()

load_dotenv()
# os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LOGFIRE_IGNORE_NO_CONFIG'] = '1'

"""
## Generate Sample Data

In this section, we create a sample dataset of car sales. This includes generating dates, car makes, models, colors, and other relevant information.
"""
logger.info("## Generate Sample Data")

n_rows = 1000

start_date = datetime(2022, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(n_rows)]

makes = ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'Nissan', 'BMW', 'Mercedes', 'Audi', 'Hyundai', 'Kia']
models = ['Sedan', 'SUV', 'Truck', 'Hatchback', 'Coupe', 'Van']
colors = ['Red', 'Blue', 'Black', 'White', 'Silver', 'Gray', 'Green']

data = {
    'Date': dates,
    'Make': np.random.choice(makes, n_rows),
    'Model': np.random.choice(models, n_rows),
    'Color': np.random.choice(colors, n_rows),
    'Year': np.random.randint(2015, 2023, n_rows),
    'Price': np.random.uniform(20000, 80000, n_rows).round(2),
    'Mileage': np.random.uniform(0, 100000, n_rows).round(0),
    'EngineSize': np.random.choice([1.6, 2.0, 2.5, 3.0, 3.5, 4.0], n_rows),
    'FuelEfficiency': np.random.uniform(20, 40, n_rows).round(1),
    'SalesPerson': np.random.choice(['Alice', 'Bob', 'Charlie', 'David', 'Eva'], n_rows)
}

df = pd.DataFrame(data).sort_values('Date')

logger.debug("\nFirst few rows of the generated data:")
logger.debug(df.head())

logger.debug("\nDataFrame info:")
df.info()

logger.debug("\nSummary statistics:")
logger.debug(df.describe())

"""
## Create Data Analysis Agent

Unlike the [LangChain notebook](https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/simple_data_analysis_agent_notebook.ipynb) on which this example is based, PydanticAI does not (yet?) have a built-in tool for processing Pandas DataFrames.

To address this, we’ll create a custom tool that implements the required functionality for our example.

We’ll begin by defining the agent itself, along with the dependencies that the tool will need. The tool implementation will follow in the next section.

#### Dependencies

PydanticAI uses a [dependency injection system](https://ai.pydantic.dev/dependencies/) to provide data and services to an agent’s system prompts, tools, and result validators.

We’ll use this system to define the DataFrame as a dependency, allowing us to reference it inside the tool.
"""
logger.info("## Create Data Analysis Agent")

@dataclass
class Deps:
    """The only dependency we need is the DataFrame we'll be working with."""
    df: pd.DataFrame

agent = Agent(
    model='ollama:llama3.2',
    system_prompt="""You are an AI assistant that helps extract information from a pandas DataFrame.
    If asked about columns, be sure to check the column names first.
    Be concise in your answers.""",
    deps_type=Deps,

    retries=10,
)

"""
## Create a Tool to Query the DataFrame

Our tool is straightforward. Unlike the LangChain function `create_pandas_dataframe_agent`, which you can see in the [LangChain notebook](https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/simple_data_analysis_agent_notebook.ipynb) and uses the Python REPL and can be dangerous, we run our queries using `pd.eval`.

`pd.eval` allows execution of only a subset of Pandas commands, limiting the potential for malicious code execution.

The downside is that this approach supports only a subset of the regular Pandas syntax, which means the agent may occasionally make mistakes by using unsupported syntax. To handle such cases, we enable retries. During the agent’s definition, we set the number of allowed retries to 10. If an error occurs during tool execution, we raise a `ModelRetry` exception to prompt the agent to retry its query.
"""
logger.info("## Create a Tool to Query the DataFrame")

@agent.tool
async def df_query(ctx: RunContext[Deps], query: str) -> str:
    """A tool for running queries on the `pandas.DataFrame`. Use this tool to interact with the DataFrame.

    `query` will be executed using `pd.eval(query, target=df)`, so it must contain syntax compatible with
    `pandas.eval`.
    """

    logger.debug(f'Running query: `{query}`')
    try:
        return str(pd.eval(query, target=ctx.deps.df))
    except Exception as e:
        raise ModelRetry(f'query: `{query}` is not a valid query. Reason: `{e}`') from e

"""
## Define Question-Asking Function

This function allows us to easily ask questions to our data analysis agent and display the results.
"""
logger.info("## Define Question-Asking Function")

def ask_agent(question):
    """Function to ask questions to the agent and display the response"""
    deps = Deps(df=df)
    logger.debug(f"Question: {question}")
    response = agent.run_sync(question, deps=deps)
    logger.debug(f"Answer: {response.new_messages()[-1].content}")
    logger.debug("---")

"""
## Example Questions

Here are some example questions you can ask the data analysis agent. You can modify these or add your own questions to analyze the dataset.
"""
logger.info("## Example Questions")

ask_agent("What are the column names in this dataset?")
ask_agent("How many rows are in this dataset?")
ask_agent("What is the average price of cars sold?")

"""
#### Analysis of Examples

As you can see in the above example, the agent got the column names right away but needed to retry a few times before arriving at the correct syntax to query the number of rows and the average price.

The primary issue was that the `Price` column name starts with a capital `P`, which caused some retries when querying the average price. We could improve the agent’s performance by including additional context in the prompt, such as column names, types, or usage examples, to help the agent arrive at correct answers more efficiently.
"""
logger.info("#### Analysis of Examples")


logger.info("\n\n[DONE]", bright=True)