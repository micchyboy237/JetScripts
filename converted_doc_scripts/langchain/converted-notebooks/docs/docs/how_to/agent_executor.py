from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ChatModelTabs from "@theme/ChatModelTabs"
import CodeBlock from "@theme/CodeBlock"
import TabItem from '@theme/TabItem'
import Tabs from '@theme/Tabs'
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
sidebar_position: 4
---

# Build an Agent with AgentExecutor (Legacy)

:::important
This section will cover building with the legacy LangChain AgentExecutor. These are fine for getting started, but past a certain point, you will likely want flexibility and control that they do not offer. For working with more advanced agents, we'd recommend checking out [LangGraph Agents](/docs/concepts/architecture/#langgraph) or the [migration guide](/docs/how_to/migrate_agent/)
:::

By themselves, language models can't take actions - they just output text.
A big use case for LangChain is creating **[agents](/docs/concepts/agents/)**.
Agents are systems that use an LLM as a reasoning engine to determine which actions to take and what the inputs to those actions should be.
The results of those actions can then be fed back into the agent and it determines whether more actions are needed, or whether it is okay to finish.

In this tutorial, we will build an agent that can interact with multiple different tools: one being a local database, the other being a search engine. You will be able to ask this agent questions, watch it call tools, and have conversations with it.

## Concepts

Concepts we will cover are:
- Using [language models](/docs/concepts/chat_models), in particular their tool calling ability
- Creating a [Retriever](/docs/concepts/retrievers) to expose specific information to our agent
- Using a Search [Tool](/docs/concepts/tools) to look up things online
- [`Chat History`](/docs/concepts/chat_history), which allows a chatbot to "remember" past interactions and take them into account when responding to follow-up questions. 
- Debugging and tracing your application using [LangSmith](https://docs.smith.langchain.com/)

## Setup

### Jupyter Notebook

This guide (and most of the other guides in the documentation) uses [Jupyter notebooks](https://jupyter.org/) and assumes the reader is as well. Jupyter notebooks are perfect for learning how to work with LLM systems because oftentimes things can go wrong (unexpected output, API down, etc) and going through guides in an interactive environment is a great way to better understand them.

This and other tutorials are perhaps most conveniently run in a Jupyter notebook. See [here](https://jupyter.org/install) for instructions on how to install.

### Installation

To install LangChain run:


<Tabs>
  <TabItem value="pip" label="Pip" default>
    <CodeBlock language="bash">pip install langchain</CodeBlock>
  </TabItem>
  <TabItem value="conda" label="Conda">
    <CodeBlock language="bash">conda install langchain -c conda-forge</CodeBlock>
  </TabItem>
</Tabs>



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

## Define tools

We first need to create the tools we want to use. We will use two tools: [Tavily](/docs/integrations/tools/tavily_search) (to search online) and then a retriever over a local index we will create

### [Tavily](/docs/integrations/tools/tavily_search)

We have a built-in tool in LangChain to easily use Tavily search engine as tool.
Note that this requires an API key - they have a free tier, but if you don't have one or don't want to create one, you can always ignore this step.

Once you create your API key, you will need to export that as:

```bash
export TAVILY_API_KEY="..."
```
"""
logger.info("# Build an Agent with AgentExecutor (Legacy)")


search = TavilySearchResults(max_results=2)

search.invoke("what is the weather in SF")

"""
### Retriever

We will also create a retriever over some data of our own. For a deeper explanation of each step here, see [this tutorial](/docs/tutorials/rag).
"""
logger.info("### Retriever")


loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(
    documents, OllamaEmbeddings(model="nomic-embed-text"))
retriever = vector.as_retriever()

retriever.invoke("how to upload a dataset")[0]

"""
Now that we have populated our index that we will do doing retrieval over, we can easily turn it into a tool (the format needed for an agent to properly use it)
"""
logger.info("Now that we have populated our index that we will do doing retrieval over, we can easily turn it into a tool (the format needed for an agent to properly use it)")


retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)

"""
### Tools

Now that we have created both, we can create a list of tools that we will use downstream.
"""
logger.info("### Tools")

tools = [search, retriever_tool]

"""
## Using Language Models

Next, let's learn how to use a language model by to call tools. LangChain supports many different language models that you can use interchangably - select the one you want to use below!


<ChatModelTabs overrideParams={{ollama: {model: "gpt-4"}}} />
"""
logger.info("## Using Language Models")


model = ChatOllama(model="llama3.2")

"""
You can call the language model by passing in a list of messages. By default, the response is a `content` string.
"""
logger.info("You can call the language model by passing in a list of messages. By default, the response is a `content` string.")


response = model.invoke([HumanMessage(content="hi!")])
response.content

"""
We can now see what it is like to enable this model to do tool calling. In order to enable that we use `.bind_tools` to give the language model knowledge of these tools
"""
logger.info("We can now see what it is like to enable this model to do tool calling. In order to enable that we use `.bind_tools` to give the language model knowledge of these tools")

model_with_tools = model.bind_tools(tools)

"""
We can now call the model. Let's first call it with a normal message, and see how it responds. We can look at both the `content` field as well as the `tool_calls` field.
"""
logger.info("We can now call the model. Let's first call it with a normal message, and see how it responds. We can look at both the `content` field as well as the `tool_calls` field.")

response = model_with_tools.invoke([HumanMessage(content="Hi!")])

logger.debug(f"ContentString: {response.content}")
logger.debug(f"ToolCalls: {response.tool_calls}")

"""
Now, let's try calling it with some input that would expect a tool to be called.
"""
logger.info(
    "Now, let's try calling it with some input that would expect a tool to be called.")

response = model_with_tools.invoke(
    [HumanMessage(content="What's the weather in SF?")])

logger.debug(f"ContentString: {response.content}")
logger.debug(f"ToolCalls: {response.tool_calls}")

"""
We can see that there's now no content, but there is a tool call! It wants us to call the Tavily Search tool.

This isn't calling that tool yet - it's just telling us to. In order to actually calll it, we'll want to create our agent.

## Create the agent

Now that we have defined the tools and the LLM, we can create the agent. We will be using a tool calling agent - for more information on this type of agent, as well as other options, see [this guide](/docs/concepts/agents/).

We can first choose the prompt we want to use to guide the agent.

If you want to see the contents of this prompt and have access to LangSmith, you can go to:

[https://smith.langchain.com/hub/hwchase17/ollama-functions-agent](https://smith.langchain.com/hub/hwchase17/ollama-functions-agent)
"""
logger.info("## Create the agent")


prompt = hub.pull("hwchase17/ollama-functions-agent")
prompt.messages

"""
Now, we can initialize the agent with the LLM, the prompt, and the tools. The agent is responsible for taking in input and deciding what actions to take. Crucially, the Agent does not execute those actions - that is done by the AgentExecutor (next step). For more information about how to think about these components, see our [conceptual guide](/docs/concepts/agents).

Note that we are passing in the `model`, not `model_with_tools`. That is because `create_tool_calling_agent` will call `.bind_tools` for us under the hood.
"""
logger.info(
    "Now, we can initialize the agent with the LLM, the prompt, and the tools. The agent is responsible for taking in input and deciding what actions to take. Crucially, the Agent does not execute those actions - that is done by the AgentExecutor (next step). For more information about how to think about these components, see our [conceptual guide](/docs/concepts/agents).")


agent = create_tool_calling_agent(model, tools, prompt)

"""
Finally, we combine the agent (the brains) with the tools inside the AgentExecutor (which will repeatedly call the agent and execute tools).
"""
logger.info("Finally, we combine the agent (the brains) with the tools inside the AgentExecutor (which will repeatedly call the agent and execute tools).")


agent_executor = AgentExecutor(agent=agent, tools=tools)

"""
## Run the agent

We can now run the agent on a few queries! Note that for now, these are all **stateless** queries (it won't remember previous interactions).

First up, let's how it responds when there's no need to call a tool:
"""
logger.info("## Run the agent")

agent_executor.invoke({"input": "hi!"})

"""
In order to see exactly what is happening under the hood (and to make sure it's not calling a tool) we can take a look at the [LangSmith trace](https://smith.langchain.com/public/8441812b-94ce-4832-93ec-e1114214553a/r)

Let's now try it out on an example where it should be invoking the retriever
"""
logger.info(
    "In order to see exactly what is happening under the hood (and to make sure it's not calling a tool) we can take a look at the [LangSmith trace](https://smith.langchain.com/public/8441812b-94ce-4832-93ec-e1114214553a/r)")

agent_executor.invoke({"input": "how can langsmith help with testing?"})

"""
Let's take a look at the [LangSmith trace](https://smith.langchain.com/public/762153f6-14d4-4c98-8659-82650f860c62/r) to make sure it's actually calling that.

Now let's try one where it needs to call the search tool:
"""
logger.info(
    "Let's take a look at the [LangSmith trace](https://smith.langchain.com/public/762153f6-14d4-4c98-8659-82650f860c62/r) to make sure it's actually calling that.")

agent_executor.invoke({"input": "whats the weather in sf?"})

"""
We can check out the [LangSmith trace](https://smith.langchain.com/public/36df5b1a-9a0b-4185-bae2-964e1d53c665/r) to make sure it's calling the search tool effectively.

## Adding in memory

As mentioned earlier, this agent is stateless. This means it does not remember previous interactions. To give it memory we need to pass in previous `chat_history`. Note: it needs to be called `chat_history` because of the prompt we are using. If we use a different prompt, we could change the variable name
"""
logger.info("## Adding in memory")

agent_executor.invoke({"input": "hi! my name is bob", "chat_history": []})


agent_executor.invoke(
    {
        "chat_history": [
            HumanMessage(content="hi! my name is bob"),
            AIMessage(content="Hello Bob! How can I assist you today?"),
        ],
        "input": "what's my name?",
    }
)

"""
If we want to keep track of these messages automatically, we can wrap this in a RunnableWithMessageHistory. For more information on how to use this, see [this guide](/docs/how_to/message_history).
"""
logger.info(
    "If we want to keep track of these messages automatically, we can wrap this in a RunnableWithMessageHistory. For more information on how to use this, see [this guide](/docs/how_to/message_history).")


store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


"""
Because we have multiple inputs, we need to specify two things:

- `input_messages_key`: The input key to use to add to the conversation history.
- `history_messages_key`: The key to add the loaded messages into.
"""
logger.info("Because we have multiple inputs, we need to specify two things:")

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

agent_with_chat_history.invoke(
    {"input": "hi! I'm bob"},
    config={"configurable": {"session_id": "<foo>"}},
)

agent_with_chat_history.invoke(
    {"input": "what's my name?"},
    config={"configurable": {"session_id": "<foo>"}},
)

"""
Example LangSmith trace: https://smith.langchain.com/public/98c8d162-60ae-4493-aa9f-992d87bd0429/r

## Conclusion

That's a wrap! In this quick start we covered how to create a simple agent. Agents are a complex topic, and there's lot to learn! 

:::important
This section covered building with LangChain Agents. They are fine for getting started, but past a certain point you will likely want flexibility and control which they do not offer. To develop more advanced agents, we recommend checking out [LangGraph](/docs/concepts/architecture/#langgraph)
:::

If you want to continue using LangChain agents, some good advanced guides are:

- [How to use LangGraph's built-in versions of `AgentExecutor`](/docs/how_to/migrate_agent)
- [How to create a custom agent](https://python.langchain.com/v0.1/docs/modules/agents/how_to/custom_agent/)
- [How to stream responses from an agent](https://python.langchain.com/v0.1/docs/modules/agents/how_to/streaming/)
- [How to return structured output from an agent](https://python.langchain.com/v0.1/docs/modules/agents/how_to/agent_structured/)
"""
logger.info("## Conclusion")


logger.info("\n\n[DONE]", bright=True)
