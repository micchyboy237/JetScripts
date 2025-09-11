from jet.logger import logger
from langchain_community.agent_toolkits.steam.toolkit import SteamToolkit
from langchain_community.utilities.steam import SteamWebAPIWrapper
from langgraph.prebuilt import create_react_agent
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
# Steam Toolkit

>[Steam (Wikipedia)](https://en.wikipedia.org/wiki/Steam_(service)) is a video game digital distribution service and storefront developed by `Valve Corporation`. It provides game updates automatically for Valve's games, and expanded to distributing third-party titles. `Steam` offers various features, like game server matchmaking with Valve Anti-Cheat measures, social networking, and game streaming services.

>[Steam](https://store.steampowered.com/about/) is the ultimate destination for playing, discussing, and creating games.

Steam toolkit has two tools:
- `Game Details`
- `Recommended Games`

This notebook provides a walkthrough of using Steam API with LangChain to retrieve Steam game recommendations based on your current Steam Game Inventory or to gather information regarding some Steam Games which you provide.

## Setting up

We have to install two python libraries.

## Imports
"""
logger.info("# Steam Toolkit")

# %pip install --upgrade --quiet python-steam-api python-decouple steamspypi

"""
## Assign Environmental Variables
To use this toolkit, please have your Ollama API Key, Steam API key (from [here](https://steamcommunity.com/dev/apikey)) and your own SteamID handy. Once you have received a Steam API Key, you can input it as an environmental variable below.
# The toolkit will read the "STEAM_KEY" API Key as an environmental variable to authenticate you so please set them here. You will also need to set your "OPENAI_API_KEY" and your "STEAM_ID".
"""
logger.info("## Assign Environmental Variables")


os.environ["STEAM_KEY"] = ""
os.environ["STEAM_ID"] = ""
# os.environ["OPENAI_API_KEY"] = ""

"""
## Initialization: 
Initialize the LLM, SteamWebAPIWrapper, SteamToolkit and most importantly the langchain agent to process your query!
## Example
"""
logger.info("## Initialization:")


steam = SteamWebAPIWrapper()
tools = [steam.run]


agent = create_react_agent("ollama:gpt-4.1-mini", tools)

events = agent.stream(
    {"messages": [("user", "can you give the information about the game Terraria?")]},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_logger.debug()

logger.info("\n\n[DONE]", bright=True)