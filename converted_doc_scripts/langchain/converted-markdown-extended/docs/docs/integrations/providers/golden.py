from jet.logger import logger
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.utilities.golden_query import GoldenQueryAPIWrapper
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
# Golden

>[Golden](https://golden.com) provides a set of natural language APIs for querying and enrichment using the Golden Knowledge Graph e.g. queries such as: `Products from Ollama`, `Generative ai companies with series a funding`, and `rappers who invest` can be used to retrieve structured data about relevant entities.
>
>The `golden-query` langchain tool is a wrapper on top of the [Golden Query API](https://docs.golden.com/reference/query-api) which enables programmatic access to these results.
>See the [Golden Query API docs](https://docs.golden.com/reference/query-api) for more information.

## Installation and Setup
- Go to the [Golden API docs](https://docs.golden.com/) to get an overview about the Golden API.
- Get your API key from the [Golden API Settings](https://golden.com/settings/api) page.
- Save your API key into GOLDEN_API_KEY env variable

## Wrappers

### Utility

There exists a GoldenQueryAPIWrapper utility which wraps this API. To import this utility:
"""
logger.info("# Golden")


"""
For a more detailed walkthrough of this wrapper, see [this notebook](/docs/integrations/tools/golden_query).

### Tool

You can also easily load this wrapper as a Tool (to use with an Agent).
You can do this with:
"""
logger.info("### Tool")

tools = load_tools(["golden-query"])

"""
For more information on tools, see [this page](/docs/how_to/tools_builtin).
"""
logger.info("For more information on tools, see [this page](/docs/how_to/tools_builtin).")

logger.info("\n\n[DONE]", bright=True)