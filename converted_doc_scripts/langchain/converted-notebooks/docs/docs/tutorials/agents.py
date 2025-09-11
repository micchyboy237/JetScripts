from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import ChatModelTabs from "@theme/ChatModelTabs";
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
keywords: [agent, agents]
---

# Build an Agent

LangChain supports the creation of [agents](/docs/concepts/agents), or systems that use [LLMs](/docs/concepts/chat_models) as reasoning engines to determine which actions to take and the inputs necessary to perform the action.
After executing actions, the results can be fed back into the LLM to determine whether more actions are needed, or whether it is okay to finish. This is often achieved via [tool-calling](/docs/concepts/tool_calling).

In this tutorial we will build an agent that can interact with a search engine. You will be able to ask this agent questions, watch it call the search tool, and have conversations with it.

## End-to-end agent

The code snippet below represents a fully functional agent that uses an LLM to decide which tools to use. It is equipped with a generic search tool. It has conversational memory - meaning that it can be used as a multi-turn chatbot.

In the rest of the guide, we will walk through the individual components and what each part does - but if you want to just grab some code and get started, feel free to use this!
"""
logger.info("# Build an Agent")


memory = MemorySaver()
model = init_chat_model("ollama:llama3.2")
search = TavilySearch(max_results=2)
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

input_message = {
    "role": "user",
    "content": "Hi, I'm Bob and I live in SF.",
}
for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_logger.debug()

input_message = {
    "role": "user",
    "content": "What's the weather where I live?",
}

for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_logger.debug()

"""
## Setup

### Jupyter Notebook

This guide (and most of the other guides in the documentation) uses [Jupyter notebooks](https://jupyter.org/) and assumes the reader is as well. Jupyter notebooks are perfect interactive environments for learning how to work with LLM systems because oftentimes things can go wrong (unexpected output, API down, etc), and observing these cases is a great way to better understand building with LLMs.

This and other tutorials are perhaps most conveniently run in a Jupyter notebook. See [here](https://jupyter.org/install) for instructions on how to install.

### Installation

To install LangChain run:
"""
logger.info("## Setup")

# %pip install -U langgraph langchain-tavily langgraph-checkpoint-sqlite

"""
For more details, see our [Installation guide](/docs/how_to/installation).

### LangSmith

Many of the applications you build with LangChain will contain multiple steps with multiple invocations of LLM calls.
As these applications get more and more complex, it becomes crucial to be able to inspect what exactly is going on inside your chain or agent.
The best way to do this is with [LangSmith](https://smith.langchain.com).

After you sign up at the link above, make sure to set your environment variables to start logging traces:

```shell
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."
```

Or, if in a notebook, you can set them with:

```python
# import getpass

os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
```

### Tavily

We will be using [Tavily](/docs/integrations/tools/tavily_search) (a search engine) as a tool.
In order to use it, you will need to get and set an API key:

```bash
export TAVILY_API_KEY="..."
```

Or, if in a notebook, you can set it with:

```python
# import getpass

# os.environ["TAVILY_API_KEY"] = getpass.getpass()
```

## Define tools

We first need to create the tools we want to use. Our main tool of choice will be [Tavily](/docs/integrations/tools/tavily_search) - a search engine. We can use the dedicated [langchain-tavily](https://pypi.org/project/langchain-tavily/) [integration package](/docs/concepts/architecture/#integration-packages) to easily use Tavily search engine as tool with LangChain.
"""
logger.info("### LangSmith")


search = TavilySearch(max_results=2)
search_results = search.invoke("What is the weather in SF")
logger.debug(search_results)
tools = [search]

"""
:::tip

In many applications, you may want to define custom tools. LangChain supports custom
tool creation via Python functions and other means. Refer to the
[How to create tools](/docs/how_to/custom_tools/) guide for details.

:::

## Using Language Models

Next, let's learn how to use a language model to call tools. LangChain supports many different language models that you can use interchangably - select the one you want to use below!


<ChatModelTabs overrideParams={{ollama: {model: "gpt-4.1"}}} />
"""
logger.info("## Using Language Models")


model = ChatOllama(model="llama3.2")

"""
You can call the language model by passing in a list of messages. By default, the response is a `content` string.
"""
logger.info("You can call the language model by passing in a list of messages. By default, the response is a `content` string.")

query = "Hi!"
response = model.invoke([{"role": "user", "content": query}])
response.text()

"""
We can now see what it is like to enable this model to do tool calling. In order to enable that we use `.bind_tools` to give the language model knowledge of these tools
"""
logger.info("We can now see what it is like to enable this model to do tool calling. In order to enable that we use `.bind_tools` to give the language model knowledge of these tools")

model_with_tools = model.bind_tools(tools)

"""
We can now call the model. Let's first call it with a normal message, and see how it responds. We can look at both the `content` field as well as the `tool_calls` field.
"""
logger.info("We can now call the model. Let's first call it with a normal message, and see how it responds. We can look at both the `content` field as well as the `tool_calls` field.")

query = "Hi!"
response = model_with_tools.invoke([{"role": "user", "content": query}])

logger.debug(f"Message content: {response.text()}\n")
logger.debug(f"Tool calls: {response.tool_calls}")

"""
Now, let's try calling it with some input that would expect a tool to be called.
"""
logger.info("Now, let's try calling it with some input that would expect a tool to be called.")

query = "Search for the weather in SF"
response = model_with_tools.invoke([{"role": "user", "content": query}])

logger.debug(f"Message content: {response.text()}\n")
logger.debug(f"Tool calls: {response.tool_calls}")

"""
We can see that there's now no text content, but there is a tool call! It wants us to call the Tavily Search tool.

This isn't calling that tool yet - it's just telling us to. In order to actually call it, we'll want to create our agent.

## Create the agent

Now that we have defined the tools and the LLM, we can create the agent. We will be using [LangGraph](/docs/concepts/architecture/#langgraph) to construct the agent. 
Currently, we are using a high level interface to construct the agent, but the nice thing about LangGraph is that this high-level interface is backed by a low-level, highly controllable API in case you want to modify the agent logic.

Now, we can initialize the agent with the LLM and the tools.

Note that we are passing in the `model`, not `model_with_tools`. That is because `create_react_agent` will call `.bind_tools` for us under the hood.
"""
logger.info("## Create the agent")


agent_executor = create_react_agent(model, tools)

"""
## Run the agent

We can now run the agent with a few queries! Note that for now, these are all **stateless** queries (it won't remember previous interactions). Note that the agent will return the **final** state at the end of the interaction (which includes any inputs, we will see later on how to get only the outputs).

First up, let's see how it responds when there's no need to call a tool:
"""
logger.info("## Run the agent")

input_message = {"role": "user", "content": "Hi!"}
response = agent_executor.invoke({"messages": [input_message]})

for message in response["messages"]:
    message.pretty_logger.debug()

"""
In order to see exactly what is happening under the hood (and to make sure it's not calling a tool) we can take a look at the [LangSmith trace](https://smith.langchain.com/public/28311faa-e135-4d6a-ab6b-caecf6482aaa/r)

Let's now try it out on an example where it should be invoking the tool
"""
logger.info("In order to see exactly what is happening under the hood (and to make sure it's not calling a tool) we can take a look at the [LangSmith trace](https://smith.langchain.com/public/28311faa-e135-4d6a-ab6b-caecf6482aaa/r)")

input_message = {"role": "user", "content": "Search for the weather in SF"}
response = agent_executor.invoke({"messages": [input_message]})

for message in response["messages"]:
    message.pretty_logger.debug()

"""
We can check out the [LangSmith trace](https://smith.langchain.com/public/f520839d-cd4d-4495-8764-e32b548e235d/r) to make sure it's calling the search tool effectively.

## Streaming Messages

We've seen how the agent can be called with `.invoke` to get  a final response. If the agent executes multiple steps, this may take a while. To show intermediate progress, we can stream back messages as they occur.
"""
logger.info("## Streaming Messages")

for step in agent_executor.stream({"messages": [input_message]}, stream_mode="values"):
    step["messages"][-1].pretty_logger.debug()

"""
## Streaming tokens

In addition to streaming back messages, it is also useful to stream back tokens.
We can do this by specifying `stream_mode="messages"`.


::: note

Below we use `message.text()`, which requires `langchain-core>=0.3.37`.

:::
"""
logger.info("## Streaming tokens")

for step, metadata in agent_executor.stream(
    {"messages": [input_message]}, stream_mode="messages"
):
    if metadata["langgraph_node"] == "agent" and (text := step.text()):
        logger.debug(text, end="|")

"""
## Adding in memory

As mentioned earlier, this agent is stateless. This means it does not remember previous interactions. To give it memory we need to pass in a checkpointer. When passing in a checkpointer, we also have to pass in a `thread_id` when invoking the agent (so it knows which thread/conversation to resume from).
"""
logger.info("## Adding in memory")


memory = MemorySaver()

agent_executor = create_react_agent(model, tools, checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

input_message = {"role": "user", "content": "Hi, I'm Bob!"}
for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_logger.debug()

input_message = {"role": "user", "content": "What's my name?"}
for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_logger.debug()

"""
Example [LangSmith trace](https://smith.langchain.com/public/fa73960b-0f7d-4910-b73d-757a12f33b2b/r)

If you want to start a new conversation, all you have to do is change the `thread_id` used
"""
logger.info("Example [LangSmith trace](https://smith.langchain.com/public/fa73960b-0f7d-4910-b73d-757a12f33b2b/r)")

config = {"configurable": {"thread_id": "xyz123"}}

input_message = {"role": "user", "content": "What's my name?"}
for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_logger.debug()

"""
## Conclusion

That's a wrap! In this quick start we covered how to create a simple agent. 
We've then shown how to stream back a response - not only with the intermediate steps, but also tokens!
We've also added in memory so you can have a conversation with them.
Agents are a complex topic with lots to learn! 

For more information on Agents, please check out the [LangGraph](/docs/concepts/architecture/#langgraph) documentation. This has it's own set of concepts, tutorials, and how-to guides.
"""
logger.info("## Conclusion")


logger.info("\n\n[DONE]", bright=True)