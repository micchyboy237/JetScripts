from jet.llm.ollama.base_langchain import ChatMLX
from jet.logger import CustomLogger
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from mem0 import MemoryClient
from typing import Annotated, TypedDict, List
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: LangGraph
---

Build a personalized Customer Support AI Agent using LangGraph for conversation flow and Mem0 for memory retention. This integration enables context-aware and efficient support experiences.

## Overview

In this guide, we'll create a Customer Support AI Agent that:
1. Uses LangGraph to manage conversation flow
2. Leverages Mem0 to store and retrieve relevant information from past interactions
3. Provides personalized responses based on user history

## Setup and Configuration

Install necessary libraries:
"""
logger.info("## Overview")

pip install langgraph langchain-openai mem0ai

"""
Import required modules and set up configurations:

<Note>Remember to get the Mem0 API key from [Mem0 Platform](https://app.mem0.ai).</Note>
"""
logger.info("Import required modules and set up configurations:")


# OPENAI_API_KEY = 'sk-xxx'  # Replace with your actual MLX API key
MEM0_API_KEY = 'your-mem0-key'  # Replace with your actual Mem0 API key

# llm = ChatMLX(model="llama-3.2-3b-instruct", log_dir=f"{OUTPUT_DIR}/chats", api_key=OPENAI_API_KEY)
mem0 = MemoryClient(api_key=MEM0_API_KEY)

"""
## Define State and Graph

Set up the conversation state and LangGraph structure:
"""
logger.info("## Define State and Graph")

class State(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    mem0_user_id: str

graph = StateGraph(State)

"""
## Create Chatbot Function

Define the core logic for the Customer Support AI Agent:
"""
logger.info("## Create Chatbot Function")

def chatbot(state: State):
    messages = state["messages"]
    user_id = state["mem0_user_id"]

    memories = mem0.search(messages[-1].content, user_id=user_id)

    context = "Relevant information from previous conversations:\n"
    for memory in memories.get('results', []):
        context += f"- {memory['memory']}\n"

    system_message = SystemMessage(content=f"""You are a helpful customer support assistant. Use the provided context to personalize your responses and remember user preferences and past interactions.
{context}""")

    full_messages = [system_message] + messages
    response = llm.invoke(full_messages)

    mem0.add(f"User: {messages[-1].content}\nAssistant: {response.content}", user_id=user_id)
    return {"messages": [response]}

"""
## Set Up Graph Structure

Configure the LangGraph with appropriate nodes and edges:
"""
logger.info("## Set Up Graph Structure")

graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", "chatbot")

compiled_graph = graph.compile()

"""
## Create Conversation Runner

Implement a function to manage the conversation flow:
"""
logger.info("## Create Conversation Runner")

def run_conversation(user_input: str, mem0_user_id: str):
    config = {"configurable": {"thread_id": mem0_user_id}}
    state = {"messages": [HumanMessage(content=user_input)], "mem0_user_id": mem0_user_id}

    for event in compiled_graph.stream(state, config):
        for value in event.values():
            if value.get("messages"):
                logger.debug("Customer Support:", value["messages"][-1].content)
                return

"""
## Main Interaction Loop

Set up the main program loop for user interaction:
"""
logger.info("## Main Interaction Loop")

if __name__ == "__main__":
    logger.debug("Welcome to Customer Support! How can I assist you today?")
    mem0_user_id = "customer_123"  # You can generate or retrieve this based on your user management system
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            logger.debug("Customer Support: Thank you for contacting us. Have a great day!")
            break
        run_conversation(user_input, mem0_user_id)

"""
## Key Features

1. **Memory Integration**: Uses Mem0 to store and retrieve relevant information from past interactions.
2. **Personalization**: Provides context-aware responses based on user history.
3. **Flexible Architecture**: LangGraph structure allows for easy expansion of the conversation flow.
4. **Continuous Learning**: Each interaction is stored, improving future responses.

## Conclusion

By integrating LangGraph with Mem0, you can build a personalized Customer Support AI Agent that can maintain context across interactions and provide personalized assistance.

## Help

- For more details on LangGraph, visit the [LangChain documentation](https://python.langchain.com/docs/langgraph).
- [Mem0 Platform](https://app.mem0.ai/).
- If you need further assistance, please feel free to reach out to us through following methods:

<Snippet file="get-help.mdx" />
"""
logger.info("## Key Features")

logger.info("\n\n[DONE]", bright=True)