from ionic_langchain.tool import Ionic, IonicTool
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_react_agent
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
# Ionic Shopping Tool

[Ionic](https://www.ioniccommerce.com/) is a plug and play ecommerce marketplace for AI Assistants. By including the [Ionic Tool](https://github.com/ioniccommerce/ionic_langchain) in your agent, you are effortlessly providing your users with the ability to shop and transact directly within your agent, and you'll get a cut of the transaction.


This is a basic jupyter notebook demonstrating how to integrate the Ionic Tool into your agent. For more information on setting up your Agent with Ionic, see the Ionic [documentation](https://docs.ioniccommerce.com/introduction).

This Jupyter Notebook demonstrates how to use the Ionic tool with an Agent.

**Note: The ionic-langchain package is maintained by the Ionic Commerce team, not the LangChain maintainers.**



---

## Setup
"""
logger.info("# Ionic Shopping Tool")

pip install langchain jet.adapters.langchain.chat_ollama langchainhub

pip install ionic-langchain

"""
## Setup Agent
"""
logger.info("## Setup Agent")


ollama_key = "YOUR KEY HERE"
model = "gpt-3.5-turbo-instruct"
temperature = 0.6

llm = ChatOllama(ollama_api_key=ollama_key,
                 model_name=model, temperature=temperature)


ionic_tool = IonicTool().tool()


ionic_tool.description = str(
    """
Ionic is an e-commerce shopping tool. Assistant uses the Ionic Commerce Shopping Tool to find, discover, and compare products from thousands of online retailers. Assistant should use the tool when the user is looking for a product recommendation or trying to find a specific product.

The user may specify the number of results, minimum price, and maximum price for which they want to see results.
Ionic Tool input is a comma-separated string of values:
  - query string (required, must not include commas)
  - number of results (default to 4, no more than 10)
  - minimum price in cents ($5 becomes 500)
  - maximum price in cents
For example, if looking for coffee beans between 5 and 10 dollars, the tool input would be `coffee beans, 5, 500, 1000`.

Return them as a markdown formatted list with each recommendation from tool results, being sure to include the full PDP URL. For example:

1. Product 1: [Price] -- link
2. Product 2: [Price] -- link
3. Product 3: [Price] -- link
4. Product 4: [Price] -- link
"""
)

tools = [ionic_tool]

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm,
    tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent, tools=tools, handle_parsing_errors=True, verbose=True, max_iterations=5
)

"""
## Run
"""
logger.info("## Run")

input = (
    "I'm looking for a new 4k monitor can you find me some options for less than $1000"
)
agent_executor.invoke({"input": input})

logger.info("\n\n[DONE]", bright=True)
