from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import ChatOllama  # Example LLM
from jet.logger import logger
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
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
# Langfuse 🪢

> **What is Langfuse?** [Langfuse](https://langfuse.com) is an open source LLM engineering platform that helps teams trace API calls, monitor performance, and debug issues in their AI applications.

## Tracing LangChain

[Langfuse Tracing](https://langfuse.com/docs/tracing) integrates with Langchain using Langchain Callbacks ([Python](https://python.langchain.com/docs/how_to/#callbacks), [JS](https://js.langchain.com/docs/how_to/#callbacks)). Thereby, the Langfuse SDK automatically creates a nested trace for every run of your Langchain applications. This allows you to log, analyze and debug your LangChain application.

You can configure the integration via (1) constructor arguments or (2) environment variables. Get your Langfuse credentials by signing up at [cloud.langfuse.com](https://cloud.langfuse.com) or [self-hosting Langfuse](https://langfuse.com/self-hosting).

### Constructor arguments
"""
logger.info("# Langfuse 🪢")

pip install langfuse

"""

"""


Langfuse(
    public_key="your-public-key",
    secret_key="your-secret-key",
    host="https://cloud.langfuse.com"  # Optional: defaults to https://cloud.langfuse.com
)

langfuse = get_client()

langfuse_handler = CallbackHandler()

llm = ChatOllama(model="llama3.2")
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | llm

response = chain.invoke({"topic": "cats"}, config={"callbacks": [langfuse_handler]})
logger.debug(response.content)

langfuse.flush()

"""
### Environment variables
"""
logger.info("### Environment variables")

LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_HOST="https://cloud.langfuse.com"

"""

"""

langfuse_handler = CallbackHandler()


chain.invoke({"input": "<user_input>"}, config={"callbacks": [langfuse_handler]})

"""
To see how to use this integration together with other Langfuse features, check out [this end-to-end example](https://langfuse.com/docs/integrations/langchain/example-python).

## Tracing LangGraph

This part demonstrates how [Langfuse](https://langfuse.com/docs) helps to debug, analyze, and iterate on your LangGraph application using the [LangChain integration](https://langfuse.com/docs/integrations/langchain/tracing).

### Initialize Langfuse

**Note:** You need to run at least Python 3.11 ([GitHub Issue](https://github.com/langfuse/langfuse/issues/1926)).

Initialize the Langfuse client with your [API keys](https://langfuse.com/faq/all/where-are-langfuse-api-keys) from the project settings in the Langfuse UI and add them to your environment.
"""
logger.info("## Tracing LangGraph")

%pip install langfuse
%pip install langchain langgraph jet.adapters.langchain.chat_ollama langchain_community

"""

"""


os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-***"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-***"
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com" # for EU data region

# os.environ["OPENAI_API_KEY"] = "***"

"""
### Simple chat app with LangGraph

**What we will do in this section:**

*   Build a support chatbot in LangGraph that can answer common questions
*   Tracing the chatbot's input and output using Langfuse

We will start with a basic chatbot and build a more advanced multi agent setup in the next section, introducing key LangGraph concepts along the way.

#### Create Agent

Start by creating a `StateGraph`. A `StateGraph` object defines our chatbot's structure as a state machine. We will add nodes to represent the LLM and functions the chatbot can call, and edges to specify how the bot transitions between these functions.
"""
logger.info("### Simple chat app with LangGraph")




class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

llm = ChatOllama(model="llama3.2")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

graph_builder.set_entry_point("chatbot")

graph_builder.set_finish_point("chatbot")

graph = graph_builder.compile()

"""
#### Add Langfuse as callback to the invocation

Now, we will add then [Langfuse callback handler for LangChain](https://langfuse.com/docs/integrations/langchain/tracing) to trace the steps of our application: `config={"callbacks": [langfuse_handler]}`
"""
logger.info("#### Add Langfuse as callback to the invocation")


langfuse_handler = CallbackHandler()

for s in graph.stream({"messages": [HumanMessage(content = "What is Langfuse?")]},
                      config={"callbacks": [langfuse_handler]}):
    logger.debug(s)

"""


{'chatbot': {'messages': [AIMessage(content='Langfuse is a tool designed to help developers monitor and observe the performance of their Large Language Model (LLM) applications. It provides detailed insights into how these applications are functioning, allowing for better debugging, optimization, and overall management. Langfuse offers features such as tracking key metrics, visualizing data, and identifying potential issues in real-time, making it easier for developers to maintain and improve their LLM-based solutions.', response_metadata={'token_usage': {'completion_tokens': 86, 'prompt_tokens': 13, 'total_tokens': 99}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_400f27fa1f', 'finish_reason': 'stop', 'logprobs': None}, id='run-9a0c97cb-ccfe-463e-902c-5a5900b796b4-0', usage_metadata={'input_tokens': 13, 'output_tokens': 86, 'total_tokens': 99})]}}

#### View traces in Langfuse

Example trace in Langfuse: https://cloud.langfuse.com/project/cloramnkj0002jz088vzn1ja4/traces/d109e148-d188-4d6e-823f-aac0864afbab

![Trace view of chat app in Langfuse](https://langfuse.com/images/cookbook/integration-langgraph/integration_langgraph_chatapp_trace.png)

- Check out the [full notebook](https://langfuse.com/docs/integrations/langchain/example-python-langgraph) to see more examples.
- To learn how to evaluate the performance of your LangGraph application, check out the [LangGraph evaluation guide](https://langfuse.com/docs/integrations/langchain/example-langgraph-agents).
"""
logger.info("#### View traces in Langfuse")

logger.info("\n\n[DONE]", bright=True)