from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from jet.llm.ollama.base_langchain import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

initialize_ollama_settings()

"""
# How to add tools to chatbots

:::info Prerequisites

This guide assumes familiarity with the following concepts:

- [Chatbots](/docs/concepts/messages)
- [Agents](/docs/tutorials/agents)
- [Chat history](/docs/concepts/chat_history)

:::

This section will cover how to create conversational agents: chatbots that can interact with other systems and APIs using tools.

:::note

This how-to guide previously built a chatbot using [RunnableWithMessageHistory](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html). You can access this version of the guide in the [v0.2 docs](https://python.langchain.com/v0.2/docs/how_to/chatbots_tools/).

As of the v0.3 release of LangChain, we recommend that LangChain users take advantage of [LangGraph persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/) to incorporate `memory` into new LangChain applications.

If your code is already relying on `RunnableWithMessageHistory` or `BaseChatMessageHistory`, you do **not** need to make any changes. We do not plan on deprecating this functionality in the near future as it works for simple chat applications and any code that uses `RunnableWithMessageHistory` will continue to work as expected.

Please see [How to migrate to LangGraph Memory](/docs/versions/migrating_memory/) for more details.
:::

## Setup

For this guide, we'll be using a [tool calling agent](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#tool-calling-agent) with a single tool for searching the web. The default will be powered by [Tavily](/docs/integrations/tools/tavily_search), but you can switch it out for any similar tool. The rest of this section will assume you're using Tavily.

You'll need to [sign up for an account](https://tavily.com/) on the Tavily website, and install the following packages:
"""

# %pip install --upgrade --quiet langchain-community langchain-openai tavily-python langgraph

# import getpass

# if not os.environ.get("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")

if not os.environ.get("TAVILY_API_KEY"):
#     os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API Key:")

"""
# You will also need your Ollama key set as `OPENAI_API_KEY` and your Tavily API key set as `TAVILY_API_KEY`.
"""

"""
## Creating an agent

Our end goal is to create an agent that can respond conversationally to user questions while looking up information as needed.

First, let's initialize Tavily and an Ollama [chat model](/docs/concepts/chat_models/) capable of tool calling:
"""


tools = [TavilySearchResults(max_results=1)]

model = ChatOllama(model="llama3.1")

"""
To make our agent conversational, we can also specify a prompt. Here's an example:
"""

prompt = (
    "You are a helpful assistant. "
    "You may not need to use tools for every query - the user may just want to chat!"
)

"""
Great! Now let's assemble our agent using LangGraph's prebuilt [create_react_agent](https://langchain-ai.github.io/langgraph/reference/prebuilt/#create_react_agent), which allows you to create a [tool-calling agent](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#tool-calling-agent):
"""


agent = create_react_agent(model, tools, state_modifier=prompt)

"""
## Running the agent

Now that we've set up our agent, let's try interacting with it! It can handle both trivial queries that require no lookup:
"""


agent.invoke({"messages": [HumanMessage(content="I'm Nemo!")]})

"""
Or, it can use of the passed search tool to get up to date information if needed:
"""

agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="What is the current conservation status of the Great Barrier Reef?"
            )
        ],
    }
)

"""
## Conversational responses

Because our prompt contains a placeholder for chat history messages, our agent can also take previous interactions into account and respond conversationally like a standard chatbot:
"""


agent.invoke(
    {
        "messages": [
            HumanMessage(content="I'm Nemo!"),
            AIMessage(content="Hello Nemo! How can I assist you today?"),
            HumanMessage(content="What is my name?"),
        ],
    }
)

"""
If preferred, you can also add memory to the LangGraph agent to manage the history of messages. Let's redeclare it this way:
"""


memory = MemorySaver()
agent = create_react_agent(model, tools, state_modifier=prompt, checkpointer=memory)

agent.invoke(
    {"messages": [HumanMessage("I'm Nemo!")]},
    config={"configurable": {"thread_id": "1"}},
)

"""
And then if we rerun our wrapped agent executor:
"""

agent.invoke(
    {"messages": [HumanMessage("What is my name?")]},
    config={"configurable": {"thread_id": "1"}},
)

"""
This [LangSmith trace](https://smith.langchain.com/public/9e6b000d-08aa-4c5a-ac83-2fdf549523cb/r) shows what's going on under the hood.

## Further reading

For more on how to build agents, check these [LangGraph](https://langchain-ai.github.io/langgraph/) guides:

* [agents conceptual guide](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/)
* [agents tutorials](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/)
* [create_react_agent](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/)

For more on tool usage, you can also check out [this use case section](/docs/how_to#tools).
"""

logger.info("\n\n[DONE]", bright=True)