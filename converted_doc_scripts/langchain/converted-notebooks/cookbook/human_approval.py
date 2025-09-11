from jet.adapters.langchain.chat_ollama import Ollama
from jet.logger import logger
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import HumanApprovalCallbackHandler
from langchain.tools import ShellTool
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
# Human-in-the-loop Tool Validation

This walkthrough demonstrates how to add human validation to any Tool. We'll do this using the `HumanApprovalCallbackhandler`.

Let's suppose we need to make use of the `ShellTool`. Adding this tool to an automated flow poses obvious risks. Let's see how we could enforce manual human approval of inputs going into this tool.

**Note**: We generally recommend against using the `ShellTool`. There's a lot of ways to misuse it, and it's not required for most use cases. We employ it here only for demonstration purposes.
"""
logger.info("# Human-in-the-loop Tool Validation")


tool = ShellTool()

logger.debug(tool.run("echo Hello World!"))

"""
## Adding Human Approval
Adding the default `HumanApprovalCallbackHandler` to the tool will make it so that a user has to manually approve every input to the tool before the command is actually executed.
"""
logger.info("## Adding Human Approval")

tool = ShellTool(callbacks=[HumanApprovalCallbackHandler()])

logger.debug(tool.run("ls /usr"))

logger.debug(tool.run("ls /private"))

"""
## Configuring Human Approval

Let's suppose we have an agent that takes in multiple tools, and we want it to only trigger human approval requests on certain tools and certain inputs. We can configure out callback handler to do just this.
"""
logger.info("## Configuring Human Approval")


def _should_check(serialized_obj: dict) -> bool:
    return serialized_obj.get("name") == "terminal"


def _approve(_input: str) -> bool:
    if _input == "echo 'Hello World'":
        return True
    msg = (
        "Do you approve of the following input? "
        "Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no."
    )
    msg += "\n\n" + _input + "\n"
    resp = input(msg)
    return resp.lower() in ("yes", "y")


callbacks = [HumanApprovalCallbackHandler(should_check=_should_check, approve=_approve)]

llm = Ollama(temperature=0)
tools = load_tools(["wikipedia", "llm-math", "terminal"], llm=llm)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

agent.run(
    "It's 2023 now. How many years ago did Konrad Adenauer become Chancellor of Germany.",
    callbacks=callbacks,
)

agent.run("print 'Hello World' in the terminal", callbacks=callbacks)

agent.run("list all directories in /private", callbacks=callbacks)

logger.info("\n\n[DONE]", bright=True)