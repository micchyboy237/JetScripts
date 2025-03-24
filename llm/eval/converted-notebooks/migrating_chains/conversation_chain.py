from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
import uuid
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
# os.environ["OPENAI_API_KEY"] = getpass()

template = """
You are a pirate. Answer the following questions as best you can.
Chat history: {history}
Question: {input}
"""

prompt = ChatPromptTemplate.from_template(template)


model = ChatOllama(model="llama3.1")

"""
# Migrating from ConversationalChain

[`ConversationChain`](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.conversation.base.ConversationChain.html) incorporated a memory of previous messages to sustain a stateful conversation.

Some advantages of switching to the Langgraph implementation are:

- Innate support for threads/separate sessions. To make this work with `ConversationChain`, you'd need to instantiate a separate memory class outside the chain.
- More explicit parameters. `ConversationChain` contains a hidden default prompt, which can cause confusion.
- Streaming support. `ConversationChain` only supports streaming via callbacks.

Langgraph's [checkpointing](https://langchain-ai.github.io/langgraph/how-tos/persistence/) system supports multiple threads or sessions, which can be specified via the `"thread_id"` key in its configuration parameters.
"""

# %pip install --upgrade --quiet langchain langchain-openai


"""
## Legacy

<details open>
"""

memory = ConversationBufferMemory()

chain = ConversationChain(
    llm=ChatOllama(model="llama3.1"),
    memory=memory,
    prompt=prompt,
)

chain({"input": "I'm Bob, how are you?"})

chain({"input": "What is my name?"})

"""
</details>

## Langgraph

<details open>
"""

workflow = StateGraph(state_schema=MessagesState)


def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}

query = "I'm Bob, how are you?"

input_messages = [
    {
        "role": "system",
        "content": "You are a pirate. Answer the following questions as best you can.",
    },
    {"role": "user", "content": query},
]
for event in app.stream({"messages": input_messages}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

query = "What is my name?"

input_messages = [{"role": "user", "content": query}]
for event in app.stream({"messages": input_messages}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

"""
</details>

## Next steps

See [this tutorial](/docs/tutorials/chatbot) for a more end-to-end guide on building with [`RunnableWithMessageHistory`](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html).

Check out the [LCEL conceptual docs](/docs/concepts/lcel) for more background information.
"""

logger.info("\n\n[DONE]", bright=True)
