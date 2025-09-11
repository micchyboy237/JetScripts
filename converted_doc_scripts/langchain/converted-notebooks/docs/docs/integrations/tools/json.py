from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.agent_toolkits import JsonToolkit, create_json_agent
from langchain_community.tools.json.tool import JsonSpec
import os
import shutil
import yaml


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
# JSON Toolkit

This notebook showcases an agent interacting with large `JSON/dict` objects. 
This is useful when you want to answer questions about a JSON blob that's too large to fit in the context window of an LLM. The agent is able to iteratively explore the blob to find what it needs to answer the user's question.

In the below example, we are using the OpenAPI spec for the Ollama API, which you can find [here](https://github.com/ollama/ollama-openapi/blob/master/openapi.yaml).

We will use the JSON agent to answer some questions about the API spec.
"""
logger.info("# JSON Toolkit")

# %pip install -qU langchain-community

"""
## Initialization
"""
logger.info("## Initialization")


with open("ollama_openapi.yml") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
json_spec = JsonSpec(dict_={}, max_value_length=4000)
json_toolkit = JsonToolkit(spec=json_spec)

json_agent_executor = create_json_agent(
    llm=Ollama(temperature=0), toolkit=json_toolkit, verbose=True
)

"""
## Individual tools

Let's see what individual tools are inside the Jira toolkit.
"""
logger.info("## Individual tools")

[(el.name, el.description) for el in json_toolkit.get_tools()]

"""
## Example: getting the required POST parameters for a request
"""
logger.info("## Example: getting the required POST parameters for a request")

json_agent_executor.run(
    "What are the required parameters in the request body to the /completions endpoint?"
)

logger.info("\n\n[DONE]", bright=True)
