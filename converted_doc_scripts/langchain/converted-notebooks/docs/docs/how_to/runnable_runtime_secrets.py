from jet.logger import logger
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
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
# How to pass runtime secrets to runnables

:::info Requires `langchain-core >= 0.2.22`

:::

We can pass in secrets to our [runnables](/docs/concepts/runnables/) at runtime using the `RunnableConfig`. Specifically we can pass in secrets with a `__` prefix to the `configurable` field. This will ensure that these secrets aren't traced as part of the invocation:
"""
logger.info("# How to pass runtime secrets to runnables")



@tool
def foo(x: int, config: RunnableConfig) -> int:
    """Sum x and a secret int"""
    return x + config["configurable"]["__top_secret_int"]


foo.invoke({"x": 5}, {"configurable": {"__top_secret_int": 2, "traced_key": "bar"}})

"""
Looking at the LangSmith trace for this run, we can see that "traced_key" was recorded (as part of Metadata) while our secret int was not: https://smith.langchain.com/public/aa7e3289-49ca-422d-a408-f6b927210170/r
"""
logger.info("Looking at the LangSmith trace for this run, we can see that "traced_key" was recorded (as part of Metadata) while our secret int was not: https://smith.langchain.com/public/aa7e3289-49ca-422d-a408-f6b927210170/r")

logger.info("\n\n[DONE]", bright=True)