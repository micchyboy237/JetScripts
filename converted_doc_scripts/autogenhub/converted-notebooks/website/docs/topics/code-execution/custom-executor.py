from IPython import get_ipython
from autogen import ConversableAgent
from autogen.coding import CodeBlock, CodeExecutor, CodeExtractor, CodeResult, MarkdownCodeExtractor
from jet.logger import CustomLogger
from typing import List
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Custom Code Executor

In this guide we will show you how to create a custom code executor that runs
code inside the same Jupyter notebook as this one.

First, let's install the required dependencies:
"""
logger.info("# Custom Code Executor")

# ! pip -qqq install pyautogen matplotlib yfinance




"""
Now we can create the custom code executor class by subclassing the
`CodeExecutor` protocol and implementing the `execute_code_blocks` method.
"""
logger.info("Now we can create the custom code executor class by subclassing the")

class NotebookExecutor(CodeExecutor):

    @property
    def code_extractor(self) -> CodeExtractor:
        return MarkdownCodeExtractor()

    def __init__(self) -> None:
        self._ipython = get_ipython()

    def execute_code_blocks(self, code_blocks: List[CodeBlock]) -> CodeResult:
        log = ""
        for code_block in code_blocks:
            result = self._ipython.run_cell("%%capture --no-display cap\n" + code_block.code)
            log += self._ipython.ev("cap.stdout")
            log += self._ipython.ev("cap.stderr")
            if result.result is not None:
                log += str(result.result)
            exitcode = 0 if result.success else 1
            if result.error_before_exec is not None:
                log += f"\n{result.error_before_exec}"
                exitcode = 1
            if result.error_in_exec is not None:
                log += f"\n{result.error_in_exec}"
                exitcode = 1
            if exitcode != 0:
                break
        return CodeResult(exit_code=exitcode, output=log)

"""
Now we can use the new custom code executor in our agents.
"""
logger.info("Now we can use the new custom code executor in our agents.")

code_writer_agent = ConversableAgent(
    name="CodeWriter",
    system_message="You are a helpful AI assistant.\n"
    "You use your coding skill to solve problems.\n"
    "You have access to a IPython kernel to execute Python code.\n"
    "You can suggest Python code in Markdown blocks, each block is a cell.\n"
    "The code blocks will be executed in the IPython kernel in the order you suggest them.\n"
    "All necessary libraries have already been installed.\n"
    "Once the task is done, returns 'TERMINATE'.",
#     llm_config={"config_list": [{"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY")}]},
)

code_executor_agent = ConversableAgent(
    name="CodeExecutor",
    llm_config=False,
    code_execution_config={"executor": NotebookExecutor()},
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", "").strip().upper(),
)

"""
Let's use the agents to complete a simple task of drawing a plot showing
the market caps of the top 7 publicly listed companies.
"""
logger.info("Let's use the agents to complete a simple task of drawing a plot showing")

chat_result = code_executor_agent.initiate_chat(
    code_writer_agent,
    message="Create a plot showing the market caps of the top 7 publicly listed companies using data from Yahoo Finance.",
)

"""
You can see the plots are now displayed in the current notebook.
"""
logger.info("You can see the plots are now displayed in the current notebook.")

logger.info("\n\n[DONE]", bright=True)