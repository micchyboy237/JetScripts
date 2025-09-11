from jet.logger import logger
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_community.chat_models.human import HumanInputChatModel
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
# Human input chat model

Along with HumanInputLLM, LangChain also provides a pseudo chat model class that can be used for testing, debugging, or educational purposes. This allows you to mock out calls to the chat model and simulate how a human would respond if they received the messages.

In this notebook, we go over how to use this.

We start this with using the HumanInputChatModel in an agent.
"""
logger.info("# Human input chat model")


"""
Since we will use the `WikipediaQueryRun` tool in this notebook, you might need to install the `wikipedia` package if you haven't done so already.
"""
logger.info("Since we will use the `WikipediaQueryRun` tool in this notebook, you might need to install the `wikipedia` package if you haven't done so already.")

# %pip install wikipedia


tools = load_tools(["wikipedia"])
llm = HumanInputChatModel()

agent = initialize_agent(
    tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent("What is Bocchi the Rock?")

logger.info("\n\n[DONE]", bright=True)