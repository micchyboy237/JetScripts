from jet.logger import CustomLogger
from jet.llm.ollama.base import initialize_ollama_settings
import os
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import AgentType
from jet.adapters.langchain.chat_ollama import ChatOllama
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os


logger = CustomLogger(
    f"{os.path.splitext(os.path.basename(__file__))[0]}.log", overwrite=True)

initialize_ollama_settings()

"""
# Data Analysis Simple Agent

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
3. Implement a query processing function to handle natural language questions
4. Demonstrate the agent's abilities with example queries

## Conclusion
This approach to data analysis offers significant benefits:
- Accessibility for non-technical users
- Flexibility in handling various query types
- Efficient ad-hoc data exploration

By making data insights more accessible, this method has the potential to transform how organizations leverage their data for decision-making across various fields and industries.

## Import libraries and set environment variables
"""


load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

np.random.seed(42)

"""
## Generate Sample Data

In this section, we create a sample dataset of car sales. This includes generating dates, car makes, models, colors, and other relevant information.
"""

n_rows = 1000

start_date = datetime(2022, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(n_rows)]

makes = ['Toyota', 'Honda', 'Ford', 'Chevrolet',
         'Nissan', 'BMW', 'Mercedes', 'Audi', 'Hyundai', 'Kia']
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
logger.info(df.info())

logger.debug("\nSummary statistics:")
logger.orange(df.describe())

"""
## Create Data Analysis Agent

Here, we create a Pandas DataFrame agent using LangChain. This agent will be capable of analyzing our dataset and answering questions about it.
"""

agent = create_pandas_dataframe_agent(
    ChatOllama(model="llama3.2"),
    df,
    verbose=True,
    allow_dangerous_code=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)
logger.debug(
    "Data Analysis Agent is ready. You can now ask questions about the data.")

"""
## Define Question-Asking Function

This function allows us to easily ask questions to our data analysis agent and display the results.
"""


def ask_agent(question):
    """Function to ask questions to the agent and display the response"""
    response = agent.run({
        "input": question,
        "agent_scratchpad": f"Human: {question}\nAI: To answer this question, I need to use Python to analyze the dataframe. I'll use the python_repl_ast tool.\n\nAction: python_repl_ast\nAction Input: ",
    })
    logger.debug(f"Question: {question}")
    logger.success(f"Answer: {response}")
    logger.debug("---")


"""
## Example Questions

Here are some example questions you can ask the data analysis agent. You can modify these or add your own questions to analyze the dataset.
"""

ask_agent("What are the column names in this dataset?")
ask_agent("How many rows are in this dataset?")
ask_agent("What is the average price of cars sold?")

logger.info("\n\n[DONE]", bright=True)
