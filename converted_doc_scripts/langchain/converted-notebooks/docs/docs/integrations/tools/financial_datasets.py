from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.agent_toolkits.financial_datasets.toolkit import (
FinancialDatasetsToolkit,
)
from langchain_community.utilities.financial_datasets import FinancialDatasetsAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
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
# FinancialDatasets Toolkit

The [financial datasets](https://financialdatasets.ai/) stock market API provides REST endpoints that let you get financial data for 16,000+ tickers spanning 30+ years.

## Setup

To use this toolkit, you need two API keys:

`FINANCIAL_DATASETS_API_KEY`: Get it from [financialdatasets.ai](https://financialdatasets.ai/).
# `OPENAI_API_KEY`: Get it from [Ollama](https://platform.ollama.com/).
"""
logger.info("# FinancialDatasets Toolkit")

# import getpass

# os.environ["FINANCIAL_DATASETS_API_KEY"] = getpass.getpass()

# os.environ["OPENAI_API_KEY"] = getpass.getpass()

"""
### Installation

This toolkit lives in the `langchain-community` package.
"""
logger.info("### Installation")

# %pip install -qU langchain-community

"""
## Instantiation

Now we can instantiate our toolkit:
"""
logger.info("## Instantiation")


api_wrapper = FinancialDatasetsAPIWrapper(
    financial_datasets_api_key=os.environ["FINANCIAL_DATASETS_API_KEY"]
)
toolkit = FinancialDatasetsToolkit(api_wrapper=api_wrapper)

"""
## Tools

View available tools:
"""
logger.info("## Tools")

tools = toolkit.get_tools()

"""
## Use within an agent

Let's equip our agent with the FinancialDatasetsToolkit and ask financial questions.
"""
logger.info("## Use within an agent")

system_prompt = """
You are an advanced financial analysis AI assistant equipped with specialized tools
to access and analyze financial data. Your primary function is to help users with
financial analysis by retrieving and interpreting income statements, balance sheets,
and cash flow statements for publicly traded companies.

You have access to the following tools from the FinancialDatasetsToolkit:

1. Balance Sheets: Retrieves balance sheet data for a given ticker symbol.
2. Income Statements: Fetches income statement data for a specified company.
3. Cash Flow Statements: Accesses cash flow statement information for a particular ticker.

Your capabilities include:

1. Retrieving financial statements for any publicly traded company using its ticker symbol.
2. Analyzing financial ratios and metrics based on the data from these statements.
3. Comparing financial performance across different time periods (e.g., year-over-year or quarter-over-quarter).
4. Identifying trends in a company's financial health and performance.
5. Providing insights on a company's liquidity, solvency, profitability, and efficiency.
6. Explaining complex financial concepts in simple terms.

When responding to queries:

1. Always specify which financial statement(s) you're using for your analysis.
2. Provide context for the numbers you're referencing (e.g., fiscal year, quarter).
3. Explain your reasoning and calculations clearly.
4. If you need more information to provide a complete answer, ask for clarification.
5. When appropriate, suggest additional analyses that might be helpful.

Remember, your goal is to provide accurate, insightful financial analysis to
help users make informed decisions. Always maintain a professional and objective tone in your responses.
"""

"""
Instantiate the LLM.
"""
logger.info("Instantiate the LLM.")


model = ChatOllama(model="llama3.2")

"""
Define a user query.
"""
logger.info("Define a user query.")

query = "What was AAPL's revenue in 2023? What about it's total debt in Q1 2024?"

"""
Create the agent.
"""
logger.info("Create the agent.")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

"""
Query the agent.
"""
logger.info("Query the agent.")

agent_executor.invoke({"input": query})

"""
## API reference

For detailed documentation of all `FinancialDatasetsToolkit` features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/agent_toolkits/langchain_community.agent_toolkits.financial_datasets.toolkit.FinancialDatasetsToolkit.html).


"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)