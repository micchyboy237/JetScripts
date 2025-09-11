from jet.transformers.formatters import format_json
from jet.logger import logger
from langchain import hub
from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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
---
sidebar_label: ZHIPU AI
---

# ZHIPU AI

This notebook shows how to use [ZHIPU AI API](https://open.bigmodel.cn/dev/api) in LangChain with the langchain.chat_models.ChatZhipuAI.

>[*GLM-4*](https://open.bigmodel.cn/) is a multi-lingual large language model aligned with human intent, featuring capabilities in Q&A, multi-turn dialogue, and code generation. The overall performance of the new generation base model GLM-4 has been significantly improved compared to the previous generation, supporting longer contexts; Stronger multimodality; Support faster inference speed, more concurrency, greatly reducing inference costs; Meanwhile, GLM-4 enhances the capabilities of intelligent agents.

## Getting started
### Installation
First, ensure the zhipuai package is installed in your Python environment. Run the following command:
"""
logger.info("# ZHIPU AI")



"""
### Importing the Required Modules
After installation, import the necessary modules to your Python script:
"""
logger.info("### Importing the Required Modules")


"""
### Setting Up Your API Key
Sign in to [ZHIPU AI](https://open.bigmodel.cn/login?redirect=%2Fusercenter%2Fapikeys) for an API Key to access our models.
"""
logger.info("### Setting Up Your API Key")


os.environ["ZHIPUAI_API_KEY"] = "zhipuai_api_key"

"""
### Initialize the ZHIPU AI Chat Model
Here's how to initialize the chat model:
"""
logger.info("### Initialize the ZHIPU AI Chat Model")

chat = ChatZhipuAI(
    model="glm-4",
    temperature=0.5,
)

"""
### Basic Usage
Invoke the model with system and human messages like this:
"""
logger.info("### Basic Usage")

messages = [
    AIMessage(content="Hi."),
    SystemMessage(content="Your role is a poet."),
    HumanMessage(content="Write a short poem about AI in four lines."),
]

response = chat.invoke(messages)
logger.debug(response.content)  # Displays the AI-generated poem

"""
## Advanced Features
### Streaming Support
For continuous interaction, use the streaming feature:
"""
logger.info("## Advanced Features")


streaming_chat = ChatZhipuAI(
    model="glm-4",
    temperature=0.5,
    streaming=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

streaming_chat(messages)

"""
### Asynchronous Calls
For non-blocking calls, use the asynchronous approach:
"""
logger.info("### Asynchronous Calls")

async_chat = ChatZhipuAI(
    model="glm-4",
    temperature=0.5,
)

response = await async_chat.agenerate([messages])
logger.success(format_json(response))
logger.debug(response)

"""
### Using With Functions Call

GLM-4 Model can be used with the function call as well，use the following code to run a simple LangChain json_chat_agent.
"""
logger.info("### Using With Functions Call")

os.environ["TAVILY_API_KEY"] = "tavily_api_key"


tools = [TavilySearchResults(max_results=1)]
prompt = hub.pull("hwchase17/react-chat-json")
llm = ChatZhipuAI(temperature=0.01, model="glm-4")

agent = create_json_chat_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

agent_executor.invoke({"input": "what is LangChain?"})

logger.info("\n\n[DONE]", bright=True)