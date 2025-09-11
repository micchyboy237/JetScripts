from jet.logger import logger
from langchain_robocorp import ActionServerToolkit
from langchain_robocorp.toolkits import ActionServerRequestTool
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
# Sema4 (fka Robocorp)

>[Robocorp](https://robocorp.com/) helps build and operate Python workers that run seamlessly anywhere at any scale


## Installation and Setup

You need to install `langchain-robocorp` python package:
"""
logger.info("# Sema4 (fka Robocorp)")

pip install langchain-robocorp

"""
You will need a running instance of `Action Server` to communicate with from your agent application.
See the [Robocorp Quickstart](https://github.com/robocorp/robocorp#quickstart) on how to setup Action Server and create your Actions.

You can bootstrap a new project using Action Server `new` command.
"""
logger.info("You will need a running instance of `Action Server` to communicate with from your agent application.")

action-server new
cd ./your-project-name
action-server start

"""
## Tool
"""
logger.info("## Tool")


"""
## Toolkit

See a [usage example](/docs/integrations/tools/robocorp).
"""
logger.info("## Toolkit")


logger.info("\n\n[DONE]", bright=True)