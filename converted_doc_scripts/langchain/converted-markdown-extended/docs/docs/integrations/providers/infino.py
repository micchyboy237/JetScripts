from jet.logger import logger
from langchain.callbacks import InfinoCallbackHandler
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
# Infino

>[Infino](https://github.com/infinohq/infino) is an open-source observability platform that stores both metrics and application logs together.

Key features of `Infino` include:
- **Metrics Tracking**: Capture time taken by LLM model to handle request, errors, number of tokens, and costing indication for the particular LLM.
- **Data Tracking**: Log and store prompt, request, and response data for each LangChain interaction.
- **Graph Visualization**: Generate basic graphs over time, depicting metrics such as request duration, error occurrences, token count, and cost.

## Installation and Setup

First, you'll need to install the  `infinopy` Python package as follows:
"""
logger.info("# Infino")

pip install infinopy

"""
If you already have an `Infino Server` running, then you're good to go; but if
you don't, follow the next steps to start it:

- Make sure you have Docker installed
- Run the following in your terminal:
    ```
    docker run --rm --detach --name infino-example -p 3000:3000 infinohq/infino:latest
    ```



## Using Infino

See a [usage example of `InfinoCallbackHandler`](/docs/integrations/callbacks/infino).
"""
logger.info("## Using Infino")


logger.info("\n\n[DONE]", bright=True)