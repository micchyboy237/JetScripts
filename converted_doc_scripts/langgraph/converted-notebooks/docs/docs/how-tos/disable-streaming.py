from IPython.display import Image, display
from jet.llm.ollama.base_langchain import ChatOllama
from jet.logger import CustomLogger
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# How to disable streaming for models that don't support it

<div class="admonition tip">
    <p class="admonition-title">Prerequisites</p>
    <p>
        This guide assumes familiarity with the following:
        <ul>
            <li>
                <a href="https://python.langchain.com/docs/concepts/#streaming">
                    streaming
                </a>                
            </li>
            <li>
                <a href="https://python.langchain.com/docs/concepts/#chat-models/">
                    Chat Models
                </a>
            </li>
        </ul>
    </p>
</div> 

Some chat models, including the new O1 models from Ollama(depending on when you're reading this), do not support streaming. This can lead to issues when using the [astream_events API](https://python.langchain.com/docs/concepts/#astream_events), as it calls models in streaming mode, expecting streaming to function properly.

In this guide, we’ll show you how to disable streaming for models that don’t support it, ensuring they they're never called in streaming mode, even when invoked through the astream_events API.
"""
logger.info("# How to disable streaming for models that don't support it")


llm = ChatOllama(model="llama3.2")

graph_builder = StateGraph(MessagesState)


def chatbot(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()


display(Image(graph.get_graph().draw_mermaid_png()))

"""
## Without disabling streaming

Now that we've defined our graph, let's try to call `astream_events` without disabling streaming. This should throw an error because the `o1` model does not support streaming natively:
"""
logger.info("## Without disabling streaming")

input = {"messages": {"role": "user", "content": "how many r's are in strawberry?"}}
try:
    for event in graph.stream_events(input, version="v2"):
        if event["event"] == "on_chat_model_end":
            logger.debug(event["data"]["output"].content, end="", flush=True)
except:
    logger.debug("Streaming not supported!")

"""
An error occurred as we expected, luckily there is an easy fix!

## Disabling streaming

Now without making any changes to our graph, let's set the [disable_streaming](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.chat_models.BaseChatModel.html#langchain_core.language_models.chat_models.BaseChatModel.disable_streaming) parameter on our model to be `True` which will solve the problem:
"""
logger.info("## Disabling streaming")

llm = ChatOllama(model="llama3.2")

graph_builder = StateGraph(MessagesState)


def chatbot(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

"""
And now, rerunning with the same input, we should see no errors:
"""
logger.info("And now, rerunning with the same input, we should see no errors:")

input = {"messages": {"role": "user", "content": "how many r's are in strawberry?"}}
for event in graph.stream_events(input, version="v2"):
    if event["event"] == "on_chat_model_end":
        logger.debug(event["data"]["output"].content, end="", flush=True)

logger.info("\n\n[DONE]", bright=True)