from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import os
import shutil
import uuid


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
# Migrating from ConversationalChain

[`ConversationChain`](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.conversation.base.ConversationChain.html) incorporated a memory of previous messages to sustain a stateful conversation.

Some advantages of switching to the Langgraph implementation are:

- Innate support for threads/separate sessions. To make this work with `ConversationChain`, you'd need to instantiate a separate memory class outside the chain.
- More explicit parameters. `ConversationChain` contains a hidden default prompt, which can cause confusion.
- Streaming support. `ConversationChain` only supports streaming via callbacks.

Langgraph's [checkpointing](https://langchain-ai.github.io/langgraph/how-tos/persistence/) system supports multiple threads or sessions, which can be specified via the `"thread_id"` key in its configuration parameters.
"""
logger.info("# Migrating from ConversationalChain")

# %pip install --upgrade --quiet langchain langchain-ollama

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass()

"""
## Legacy

<details open>
"""
logger.info("## Legacy")


template = """
You are a pirate. Answer the following questions as best you can.
Chat history: {history}
Question: {input}
"""

prompt = ChatPromptTemplate.from_template(template)

memory = ConversationBufferMemory()

chain = ConversationChain(
    llm=ChatOllama(model="llama3.2"),
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
logger.info("## Langgraph")



model = ChatOllama(model="llama3.2")

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
    event["messages"][-1].pretty_logger.debug()

query = "What is my name?"

input_messages = [{"role": "user", "content": query}]
for event in app.stream({"messages": input_messages}, config, stream_mode="values"):
    event["messages"][-1].pretty_logger.debug()

"""
</details>

## Next steps

See [this tutorial](/docs/tutorials/chatbot) for a more end-to-end guide on building with [`RunnableWithMessageHistory`](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html).

Check out the [LCEL conceptual docs](/docs/concepts/lcel) for more background information.
"""
logger.info("## Next steps")

logger.info("\n\n[DONE]", bright=True)