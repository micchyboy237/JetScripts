from jet.adapters.langchain.chat_ollama import Ollama
from jet.logger import logger
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import SceneXplainTool
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
# SceneXplain


[SceneXplain](https://scenex.jina.ai/) is an ImageCaptioning service accessible through the SceneXplain Tool.

To use this tool, you'll need to make an account and fetch your API Token [from the website](https://scenex.jina.ai/api). Then you can instantiate the tool.
"""
logger.info("# SceneXplain")


os.environ["SCENEX_API_KEY"] = "<YOUR_API_KEY>"


tools = load_tools(["sceneXplain"])

"""
Or directly instantiate the tool.
"""
logger.info("Or directly instantiate the tool.")


tool = SceneXplainTool()

"""
## Usage in an Agent

The tool can be used in any LangChain agent as follows:
"""
logger.info("## Usage in an Agent")


llm = Ollama(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history")
agent = initialize_agent(
    tools, llm, memory=memory, agent="conversational-react-description", verbose=True
)
output = agent.run(
    input=(
        "What is in this image https://storage.googleapis.com/causal-diffusion.appspot.com/imagePrompts%2F0rw369i5h9t%2Foriginal.png. "
        "Is it movie or a game? If it is a movie, what is the name of the movie?"
    )
)

logger.debug(output)

logger.info("\n\n[DONE]", bright=True)