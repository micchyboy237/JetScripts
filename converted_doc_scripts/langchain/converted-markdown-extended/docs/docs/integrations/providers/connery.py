from jet.logger import logger
from langchain_community.agent_toolkits.connery import ConneryToolkit
from langchain_community.tools.connery import ConneryService
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
# Connery

>[Connery SDK](https://github.com/connery-io/connery-sdk) is an NPM package that
> includes both an SDK and a CLI, designed for the development of plugins and actions.
>
>The CLI automates many things in the development process. The SDK
> offers a JavaScript API for defining plugins and actions and packaging them
> into a plugin server with a standardized REST API generated from the metadata.
> The plugin server handles authorization, input validation, and logging.
> So you can focus on the logic of your actions.
>
> See the use cases and examples in the [Connery SDK documentation](https://sdk.connery.io/docs/use-cases/)

## Toolkit

See [usage example](/docs/integrations/tools/connery).
"""
logger.info("# Connery")


"""
## Tools

### ConneryAction
"""
logger.info("## Tools")


logger.info("\n\n[DONE]", bright=True)