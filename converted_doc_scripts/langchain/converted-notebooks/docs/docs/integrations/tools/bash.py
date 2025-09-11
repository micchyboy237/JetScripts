from jet.logger import logger
from langchain_community.tools import ShellTool
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
# Shell (bash)

Giving agents access to the shell is powerful (though risky outside a sandboxed environment).

The LLM can use it to execute any shell commands. A common use case for this is letting the LLM interact with your local file system.

**Note:** Shell tool does not work with Windows OS.
"""
logger.info("# Shell (bash)")

# %pip install --upgrade --quiet langchain-community


shell_tool = ShellTool()

logger.debug(shell_tool.run({"commands": ["echo 'Hello World!'", "time"]}))

"""
### Use with Agents

As with all tools, these can be given to an agent to accomplish more complex tasks. Let's have the agent fetch some links from a web page.
"""
logger.info("### Use with Agents")


tools = [shell_tool]
agent = create_react_agent("ollama:gpt-4.1-mini", tools)

input_message = {
    "role": "user",
    "content": (
        "Download the README here and identify the link for LangChain tutorials: "
        "https://raw.githubusercontent.com/langchain-ai/langchain/master/README.md"
    ),
}

for step in agent.stream(
    {"messages": [input_message]},
    stream_mode="values",
):
    step["messages"][-1].pretty_logger.debug()

logger.info("\n\n[DONE]", bright=True)