from jet.logger import logger
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
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
# Python REPL

Sometimes, for complex calculations, rather than have an LLM generate the answer directly, it can be better to have the LLM generate code to calculate the answer, and then run that code to get the answer. In order to easily do that, we provide a simple Python REPL to execute commands in.

This interface will only return things that are printed - therefore, if you want to use it to calculate an answer, make sure to have it print out the answer.


:::caution
Python REPL can execute arbitrary code on the host machine (e.g., delete files, make network requests). Use with caution.

For more information general security guidelines, please see https://python.langchain.com/docs/security/.
:::
"""
logger.info("# Python REPL")


python_repl = PythonREPL()

python_repl.run("logger.debug(1+1)")

repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `logger.debug(...)`.",
    func=python_repl.run,
)

logger.info("\n\n[DONE]", bright=True)