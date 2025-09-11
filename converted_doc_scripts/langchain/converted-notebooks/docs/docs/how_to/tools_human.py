from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import tool
from typing import Dict, List
import ChatModelTabs from "@theme/ChatModelTabs";
import json
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
# How to add a human-in-the-loop for tools

There are certain tools that we don't trust a model to execute on its own. One thing we can do in such situations is require human approval before the tool is invoked.

:::info

This how-to guide shows a simple way to add human-in-the-loop for code running in a jupyter notebook or in a terminal.

To build a production application, you will need to do more work to keep track of application state appropriately.

We recommend using `langgraph` for powering such a capability. For more details, please see this [guide](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/).
:::

## Setup

We'll need to install the following packages:
"""
logger.info("# How to add a human-in-the-loop for tools")

# %pip install --upgrade --quiet langchain

"""
And set these environment variables:
"""
logger.info("And set these environment variables:")

# import getpass

"""
## Chain

Let's create a few simple (dummy) tools and a tool-calling chain:


<ChatModelTabs customVarName="llm"/>
"""
logger.info("## Chain")


llm = ChatOllama(model="llama3.2")




@tool
def count_emails(last_n_days: int) -> int:
    """Dummy function to count number of e-mails. Returns 2 * last_n_days."""
    return last_n_days * 2


@tool
def send_email(message: str, recipient: str) -> str:
    """Dummy function for sending an e-mail."""
    return f"Successfully sent email to {recipient}."


tools = [count_emails, send_email]
llm_with_tools = llm.bind_tools(tools)


def call_tools(msg: AIMessage) -> List[Dict]:
    """Simple sequential tool calling helper."""
    tool_map = {tool.name: tool for tool in tools}
    tool_calls = msg.tool_calls.copy()
    for tool_call in tool_calls:
        tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
    return tool_calls


chain = llm_with_tools | call_tools
chain.invoke("how many emails did i get in the last 5 days?")

"""
## Adding human approval

Let's add a step in the chain that will ask a person to approve or reject the tool call request.

On rejection, the step will raise an exception which will stop execution of the rest of the chain.
"""
logger.info("## Adding human approval")



class NotApproved(Exception):
    """Custom exception."""


def human_approval(msg: AIMessage) -> AIMessage:
    """Responsible for passing through its input or raising an exception.

    Args:
        msg: output from the chat model

    Returns:
        msg: original output from the msg
    """
    tool_strs = "\n\n".join(
        json.dumps(tool_call, indent=2) for tool_call in msg.tool_calls
    )
    input_msg = (
        f"Do you approve of the following tool invocations\n\n{tool_strs}\n\n"
        "Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no.\n >>>"
    )
    resp = input(input_msg)
    if resp.lower() not in ("yes", "y"):
        raise NotApproved(f"Tool invocations not approved:\n\n{tool_strs}")
    return msg

chain = llm_with_tools | human_approval | call_tools
chain.invoke("how many emails did i get in the last 5 days?")

try:
    chain.invoke("Send sally@gmail.com an email saying 'What's up homie'")
except NotApproved as e:
    logger.debug()
    logger.debug(e)

logger.info("\n\n[DONE]", bright=True)