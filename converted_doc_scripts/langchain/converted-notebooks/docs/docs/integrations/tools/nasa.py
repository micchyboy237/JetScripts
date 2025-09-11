from jet.adapters.langchain.chat_ollama import Ollama
from jet.logger import logger
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.nasa.toolkit import NasaToolkit
from langchain_community.utilities.nasa import NasaAPIWrapper
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
# NASA Toolkit

This notebook shows how to use agents to interact with the NASA toolkit. The toolkit provides access to the NASA Image and Video Library API, with potential to expand and include other accessible NASA APIs in future iterations.

**Note: NASA Image and Video Library search queries can result in large responses when the number of desired media results is not specified. Consider this prior to using the agent with LLM token credits.**

## Example Use:
---
### Initializing the agent
"""
logger.info("# NASA Toolkit")

# %pip install -qU langchain-community


llm = Ollama(temperature=0, ollama_)
nasa = NasaAPIWrapper()
toolkit = NasaToolkit.from_nasa_api_wrapper(nasa)
agent = initialize_agent(
    toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

"""
### Querying media assets
"""
logger.info("### Querying media assets")

agent.run(
    "Can you find three pictures of the moon published between the years 2014 and 2020?"
)

"""
### Querying details about media assets
"""
logger.info("### Querying details about media assets")

output = agent.run(
    "I've just queried an image of the moon with the NASA id NHQ_2019_0311_Go Forward to the Moon."
    " Where can I find the metadata manifest for this asset?"
)

logger.info("\n\n[DONE]", bright=True)