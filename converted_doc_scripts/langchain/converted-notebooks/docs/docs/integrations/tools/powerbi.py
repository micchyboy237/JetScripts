from azure.identity import DefaultAzureCredential
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.agent_toolkits import PowerBIToolkit, create_pbi_agent
from langchain_community.utilities.powerbi import PowerBIDataset
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
# PowerBI Toolkit

This notebook showcases an agent interacting with a `Power BI Dataset`. The agent is answering more general questions about a dataset, as well as recover from errors.

Note that, as this agent is in active development, all answers might not be correct. It runs against the [executequery endpoint](https://learn.microsoft.com/en-us/rest/api/power-bi/datasets/execute-queries), which does not allow deletes.

### Notes:
- It relies on authentication with the azure.identity package, which can be installed with `pip install azure-identity`. Alternatively you can create the powerbi dataset with a token as a string without supplying the credentials.
- You can also supply a username to impersonate for use with datasets that have RLS enabled. 
- The toolkit uses a LLM to create the query from the question, the agent uses the LLM for the overall execution.
- Testing was done mostly with a `gpt-3.5-turbo-instruct` model, codex models did not seem to perform ver well.

## Initialization
"""
logger.info("# PowerBI Toolkit")


fast_llm = ChatOllama(
    temperature=0.5, max_tokens=1000, model_name="gpt-3.5-turbo", verbose=True
)
smart_llm = ChatOllama(model="llama3.2")

toolkit = PowerBIToolkit(
    powerbi=PowerBIDataset(
        dataset_id="<dataset_id>",
        table_names=["table1", "table2"],
        credential=DefaultAzureCredential(),
    ),
    llm=smart_llm,
)

agent_executor = create_pbi_agent(
    llm=fast_llm,
    toolkit=toolkit,
    verbose=True,
)

"""
## Example: describing a table
"""
logger.info("## Example: describing a table")

agent_executor.run("Describe table1")

"""
## Example: simple query on a table
In this example, the agent actually figures out the correct query to get a row count of the table.
"""
logger.info("## Example: simple query on a table")

agent_executor.run("How many records are in table1?")

"""
## Example: running queries
"""
logger.info("## Example: running queries")

agent_executor.run("How many records are there by dimension1 in table2?")

agent_executor.run("What unique values are there for dimensions2 in table2")

"""
## Example: add your own few-shot prompts
"""
logger.info("## Example: add your own few-shot prompts")

few_shots = """
Question: How many rows are in the table revenue?
DAX: EVALUATE ROW("Number of rows", COUNTROWS(revenue_details))
----
Question: How many rows are in the table revenue where year is not empty?
DAX: EVALUATE ROW("Number of rows", COUNTROWS(FILTER(revenue_details, revenue_details[year] <> "")))
----
Question: What was the average of value in revenue in dollars?
DAX: EVALUATE ROW("Average", AVERAGE(revenue_details[dollar_value]))
----
"""
toolkit = PowerBIToolkit(
    powerbi=PowerBIDataset(
        dataset_id="<dataset_id>",
        table_names=["table1", "table2"],
        credential=DefaultAzureCredential(),
    ),
    llm=smart_llm,
    examples=few_shots,
)
agent_executor = create_pbi_agent(
    llm=fast_llm,
    toolkit=toolkit,
    verbose=True,
)

agent_executor.run("What was the maximum of value in revenue in dollars in 2022?")

logger.info("\n\n[DONE]", bright=True)