from jet.logger import logger
from langchain.agents import AgentExecutor, AgentType, initialize_agent, load_tools
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import GradientLLM
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
# Memorize

Fine-tuning LLM itself to memorize information using unsupervised learning.

This tool requires LLMs that support fine-tuning. Currently, only `langchain.llms import GradientLLM` is supported.

## Imports
"""
logger.info("# Memorize")



"""
## Set the Environment API Key
Make sure to get your API key from Gradient AI. You are given $10 in free credits to test and fine-tune different models.
"""
logger.info("## Set the Environment API Key")

# from getpass import getpass

if not os.environ.get("GRADIENT_ACCESS_TOKEN", None):
#     os.environ["GRADIENT_ACCESS_TOKEN"] = getpass("gradient.ai access token:")
if not os.environ.get("GRADIENT_WORKSPACE_ID", None):
#     os.environ["GRADIENT_WORKSPACE_ID"] = getpass("gradient.ai workspace id:")
if not os.environ.get("GRADIENT_MODEL_ADAPTER_ID", None):
#     os.environ["GRADIENT_MODEL_ID"] = getpass("gradient.ai model id:")

"""
Optional: Validate your Environment variables ```GRADIENT_ACCESS_TOKEN``` and ```GRADIENT_WORKSPACE_ID``` to get currently deployed models.

## Create the `GradientLLM` instance
You can specify different parameters such as the model name, max tokens generated, temperature, etc.
"""
logger.info("## Create the `GradientLLM` instance")

llm = GradientLLM(
    model_id=os.environ["GRADIENT_MODEL_ID"],
)

"""
## Load tools
"""
logger.info("## Load tools")

tools = load_tools(["memorize"], llm=llm)

"""
## Initiate the Agent
"""
logger.info("## Initiate the Agent")

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

"""
## Run the agent
Ask the agent to memorize a piece of text.
"""
logger.info("## Run the agent")

agent.run(
    "Please remember the fact in detail:\nWith astonishing dexterity, Zara Tubikova set a world record by solving a 4x4 Rubik's Cube variation blindfolded in under 20 seconds, employing only their feet."
)

logger.info("\n\n[DONE]", bright=True)