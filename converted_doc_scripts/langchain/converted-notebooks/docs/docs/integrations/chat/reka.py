from jet.logger import logger
from langchain_community.chat_models import ChatReka
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
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
sidebar_label: Reka
---

# ChatReka

This notebook provides a quick overview for getting started with Reka [chat models](../../concepts/chat_models.mdx). 

Reka has several chat models. You can find information about their latest models and their costs, context windows, and supported input types in the [Reka docs](https://docs.reka.ai/available-models).




## Overview
### Integration details

| Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatReka] | [langchain_community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_community?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_community?style=flat-square&label=%20) |

### Model features
| [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | 

## Setup

To access Reka models you'll need to create a Reka developer account, get an API key, and install the `langchain_community` integration package and the reka python package via 'pip install reka-api'.

### Credentials

Head to https://platform.reka.ai/ to sign up for Reka and generate an API key. Once you've done this set the REKA_API_KEY environment variable:

### Installation

The LangChain __ModuleName__ integration lives in the `langchain_community` package:
"""
logger.info("# ChatReka")

# %pip install -qU langchain_community reka-api

"""
## Instantiation
"""
logger.info("## Instantiation")

# import getpass

# os.environ["REKA_API_KEY"] = getpass.getpass("Enter your Reka API key: ")

"""
Optional: use Langsmith to trace the execution of the model
"""
logger.info("Optional: use Langsmith to trace the execution of the model")

# import getpass

os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your Langsmith API key: ")


model = ChatReka()

"""
## Invocation
"""
logger.info("## Invocation")

model.invoke("hi")

"""
# Images input
"""
logger.info("# Images input")


image_url = "https://v0.docs.reka.ai/_images/000000245576.jpg"

message = HumanMessage(
    content=[
        {"type": "text", "text": "describe the weather in this image"},
        {
            "type": "image_url",
            "image_url": {"url": image_url},
        },
    ],
)
response = model.invoke([message])
logger.debug(response.content)

"""
# Multiple images as input
"""
logger.info("# Multiple images as input")

message = HumanMessage(
    content=[
        {"type": "text", "text": "What are the difference between the two images? "},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://cdn.pixabay.com/photo/2019/07/23/13/51/shepherd-dog-4357790_1280.jpg"
            },
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "https://cdn.pixabay.com/photo/2024/02/17/00/18/cat-8578562_1280.jpg"
            },
        },
    ],
)
response = model.invoke([message])
logger.debug(response.content)

"""
## Chaining
"""
logger.info("## Chaining")


prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | model
chain.invoke(
    {
        "input_language": "English",
        "output_language": "German",
        "input": "I love programming.",
    }
)

"""
Use with Tavily api search

# Tool use and agent creation

## Define the tools

We first need to create the tools we want to use. Our main tool of choice will be Tavily - a search engine. We have a built-in tool in LangChain to easily use Tavily search engine as tool.
"""
logger.info("# Tool use and agent creation")

# import getpass

# os.environ["TAVILY_API_KEY"] = getpass.getpass("Enter your Tavily API key: ")


search = TavilySearchResults(max_results=2)
search_results = search.invoke("what is the weather in SF")
logger.debug(search_results)
tools = [search]

"""
We can now see what it is like to enable this model to do tool calling. In order to enable that we use .bind_tools to give the language model knowledge of these tools
"""
logger.info("We can now see what it is like to enable this model to do tool calling. In order to enable that we use .bind_tools to give the language model knowledge of these tools")

model_with_tools = model.bind_tools(tools)

"""
We can now call the model. Let's first call it with a normal message, and see how it responds. We can look at both the content field as well as the tool_calls field.
"""
logger.info("We can now call the model. Let's first call it with a normal message, and see how it responds. We can look at both the content field as well as the tool_calls field.")


response = model_with_tools.invoke([HumanMessage(content="Hi!")])

logger.debug(f"ContentString: {response.content}")
logger.debug(f"ToolCalls: {response.tool_calls}")

"""
Now, let's try calling it with some input that would expect a tool to be called.
"""
logger.info("Now, let's try calling it with some input that would expect a tool to be called.")

response = model_with_tools.invoke([HumanMessage(content="What's the weather in SF?")])

logger.debug(f"ContentString: {response.content}")
logger.debug(f"ToolCalls: {response.tool_calls}")

"""
We can see that there's now no text content, but there is a tool call! It wants us to call the Tavily Search tool.

This isn't calling that tool yet - it's just telling us to. In order to actually call it, we'll want to create our agent.

# Create the agent

Now that we have defined the tools and the LLM, we can create the agent. We will be using LangGraph to construct the agent. Currently, we are using a high level interface to construct the agent, but the nice thing about LangGraph is that this high-level interface is backed by a low-level, highly controllable API in case you want to modify the agent logic.

Now, we can initialize the agent with the LLM and the tools.

Note that we are passing in the model, not model_with_tools. That is because `create_react_agent` will call `.bind_tools` for us under the hood.
"""
logger.info("# Create the agent")


agent_executor = create_react_agent(model, tools)

"""
Let's now try it out on an example where it should be invoking the tool
"""
logger.info("Let's now try it out on an example where it should be invoking the tool")

response = agent_executor.invoke({"messages": [HumanMessage(content="hi!")]})

response["messages"]

"""
In order to see exactly what is happening under the hood (and to make sure it's not calling a tool) we can take a look at the LangSmith trace: https://smith.langchain.com/public/2372d9c5-855a-45ee-80f2-94b63493563d/r
"""
logger.info("In order to see exactly what is happening under the hood (and to make sure it's not calling a tool) we can take a look at the LangSmith trace: https://smith.langchain.com/public/2372d9c5-855a-45ee-80f2-94b63493563d/r")

response = agent_executor.invoke(
    {"messages": [HumanMessage(content="whats the weather in sf?")]}
)
response["messages"]

"""
We can check out the LangSmith trace to make sure it's calling the search tool effectively.

https://smith.langchain.com/public/013ef704-654b-4447-8428-637b343d646e/r

We've seen how the agent can be called with `.invoke` to get a final response. If the agent executes multiple steps, this may take a while. To show intermediate progress, we can stream back messages as they occur.
"""
logger.info("We can check out the LangSmith trace to make sure it's calling the search tool effectively.")

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather in sf?")]}
):
    logger.debug(chunk)
    logger.debug("----")

"""
## API reference

https://docs.reka.ai/quick-start
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)