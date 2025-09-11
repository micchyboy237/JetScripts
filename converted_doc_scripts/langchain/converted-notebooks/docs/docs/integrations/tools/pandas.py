from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import Ollama
from jet.logger import logger
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
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
# Pandas Dataframe

This notebook shows how to use agents to interact with a `Pandas DataFrame`. It is mostly optimized for question answering.

**NOTE: this agent calls the `Python` agent under the hood, which executes LLM generated Python code - this can be bad if the LLM generated Python code is harmful. Use cautiously.**

**NOTE: Since langchain migrated to v0.3 you should upgrade jet.adapters.langchain.chat_ollama and langchain.   This would avoid import errors.**


pip install --upgrade jet.adapters.langchain.chat_ollama
pip install --upgrade langchain
"""
logger.info("# Pandas Dataframe")



df = pd.read_csv(
    "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv"
)

"""
## Using `ZERO_SHOT_REACT_DESCRIPTION`

This shows how to initialize the agent using the `ZERO_SHOT_REACT_DESCRIPTION` agent type.
"""
logger.info("## Using `ZERO_SHOT_REACT_DESCRIPTION`")

agent = create_pandas_dataframe_agent(Ollama(temperature=0), df, verbose=True)

"""
## Using Ollama Functions

This shows how to initialize the agent using the OPENAI_FUNCTIONS agent type. Note that this is an alternative to the above.
"""
logger.info("## Using Ollama Functions")

agent = create_pandas_dataframe_agent(
    ChatOllama(model="llama3.2"),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

agent.invoke("how many rows are there?")

agent.invoke("how many people have more than 3 siblings")

agent.invoke("whats the square root of the average age?")

"""
## Multi DataFrame Example

This next part shows how the agent can interact with multiple dataframes passed in as a list.
"""
logger.info("## Multi DataFrame Example")

df1 = df.copy()
df1["Age"] = df1["Age"].fillna(df1["Age"].mean())

agent = create_pandas_dataframe_agent(Ollama(temperature=0), [df, df1], verbose=True)
agent.invoke("how many rows in the age column are different?")

logger.info("\n\n[DONE]", bright=True)