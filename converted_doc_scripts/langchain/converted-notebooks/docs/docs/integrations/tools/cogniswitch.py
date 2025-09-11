from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain_community.agent_toolkits import CogniswitchToolkit
import os
import shutil
import warnings


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
# Cogniswitch Toolkit

CogniSwitch is used to build production ready applications that can consume, organize and retrieve knowledge flawlessly. Using the framework of your choice, in this case Langchain, CogniSwitch helps alleviate the stress of decision making when it comes to, choosing the right storage and retrieval formats. It also eradicates reliability issues and hallucinations when it comes to responses that are generated. 

## Setup

Visit [this page](https://www.cogniswitch.ai/developer?utm_source=langchain&utm_medium=langchainbuild&utm_id=dev) to register a Cogniswitch account.

- Signup with your email and verify your registration 

- You will get a mail with a platform token and oauth token for using the services.
"""
logger.info("# Cogniswitch Toolkit")

# %pip install -qU langchain-community

"""
## Import necessary libraries
"""
logger.info("## Import necessary libraries")


warnings.filterwarnings("ignore")



"""
## Cogniswitch platform token, OAuth token and Ollama API key
"""
logger.info("## Cogniswitch platform token, OAuth token and Ollama API key")

cs_token = "Your CogniSwitch token"
OAI_token = "Your Ollama API token"
oauth_token = "Your CogniSwitch authentication token"

# os.environ["OPENAI_API_KEY"] = OAI_token

"""
## Instantiate the cogniswitch toolkit with the credentials
"""
logger.info("## Instantiate the cogniswitch toolkit with the credentials")

cogniswitch_toolkit = CogniswitchToolkit(
    cs_token=cs_token, OAI_token=OAI_token, apiKey=oauth_token
)

"""
### Get the list of cogniswitch tools
"""
logger.info("### Get the list of cogniswitch tools")

tool_lst = cogniswitch_toolkit.get_tools()

"""
## Instantiate the LLM
"""
logger.info("## Instantiate the LLM")

llm = ChatOllama(
    temperature=0,
    ollama_api_key=OAI_token,
    max_tokens=1500,
    model_name="gpt-3.5-turbo-0613",
)

"""
## Use the LLM with the Toolkit

### Create an agent with the LLM and Toolkit
"""
logger.info("## Use the LLM with the Toolkit")

agent_executor = create_conversational_retrieval_agent(llm, tool_lst, verbose=False)

"""
### Invoke the agent to upload a URL
"""
logger.info("### Invoke the agent to upload a URL")

response = agent_executor.invoke("upload this url https://cogniswitch.ai/developer")

logger.debug(response["output"])

"""
### Invoke the agent to upload a File
"""
logger.info("### Invoke the agent to upload a File")

response = agent_executor.invoke("upload this file example_file.txt")

logger.debug(response["output"])

"""
### Invoke the agent to get the status of a document
"""
logger.info("### Invoke the agent to get the status of a document")

response = agent_executor.invoke("Tell me the status of this document example_file.txt")

logger.debug(response["output"])

"""
### Invoke the agent with query and get the answer
"""
logger.info("### Invoke the agent with query and get the answer")

response = agent_executor.invoke("How can cogniswitch help develop GenAI applications?")

logger.debug(response["output"])

logger.info("\n\n[DONE]", bright=True)