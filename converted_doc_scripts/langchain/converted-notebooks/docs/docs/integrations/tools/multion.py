from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain import hub
from langchain.agents import AgentExecutor, create_ollama_functions_agent
from langchain_community.agent_toolkits import MultionToolkit
import multion
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
# MultiOn Toolkit
 
[MultiON](https://www.multion.ai/blog/multion-building-a-brighter-future-for-humanity-with-ai-agents) has built an AI Agent that can interact with a broad array of web services and applications. 

This notebook walks you through connecting LangChain to the `MultiOn` Client in your browser. 

This enables custom agentic workflow that utilize the power of MultiON agents.
 
To use this toolkit, you will need to add `MultiOn Extension` to your browser: 

* Create a [MultiON account](https://app.multion.ai/login?callbackUrl=%2Fprofile). 
* Add  [MultiOn extension for Chrome](https://multion.notion.site/Download-MultiOn-ddddcfe719f94ab182107ca2612c07a5).
"""
logger.info("# MultiOn Toolkit")

# %pip install --upgrade --quiet  multion langchain -q

# %pip install -qU langchain-community


toolkit = MultionToolkit()
toolkit

tools = toolkit.get_tools()
tools

"""
## MultiOn Setup

Once you have created an account, create an API key at https://app.multion.ai/. 

Login to establish connection with your extension.
"""
logger.info("## MultiOn Setup")


multion.login()

"""
## Use Multion Toolkit within an Agent

This will use MultiON chrome extension to perform the desired actions.

We can run the below, and view the [trace](https://smith.langchain.com/public/34aaf36d-204a-4ce3-a54e-4a0976f09670/r) to see:

* The agent uses the `create_multion_session` tool
* It then uses MultiON to execute the query
"""
logger.info("## Use Multion Toolkit within an Agent")


instructions = """You are an assistant."""
base_prompt = hub.pull("langchain-ai/ollama-functions-template")
prompt = base_prompt.partial(instructions=instructions)

llm = ChatOllama(model="llama3.2")

agent = create_ollama_functions_agent(llm, toolkit.get_tools(), prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=False,
)

agent_executor.invoke(
    {
        "input": "Use multion to explain how AlphaCodium works, a recently released code language model."
    }
)

logger.info("\n\n[DONE]", bright=True)