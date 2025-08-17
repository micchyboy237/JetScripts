from jet.llm.ollama.base_langchain import ChatMLX
from jet.logger import CustomLogger
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from mem0 import MemoryClient
from typing import List, Dict
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
title: Langchain
---

Build a personalized Travel Agent AI using LangChain for conversation flow and Mem0 for memory retention. This integration enables context-aware and efficient travel planning experiences.

## Overview

In this guide, we'll create a Travel Agent AI that:
1. Uses LangChain to manage conversation flow
2. Leverages Mem0 to store and retrieve relevant information from past interactions
3. Provides personalized travel recommendations based on user history

## Setup and Configuration

Install necessary libraries:
"""
logger.info("## Overview")

pip install langchain jet.llm.ollama.base_langchain mem0ai

"""
Import required modules and set up configurations:

<Note>Remember to get the Mem0 API key from [Mem0 Platform](https://app.mem0.ai).</Note>
"""
logger.info("Import required modules and set up configurations:")


# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["MEM0_API_KEY"] = "your-mem0-api-key"

llm = ChatMLX(model="llama-3.2-3b-instruct")
mem0 = MemoryClient()

"""
## Create Prompt Template

Set up the conversation prompt template:
"""
logger.info("## Create Prompt Template")

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a helpful travel agent AI. Use the provided context to personalize your responses and remember user preferences and past interactions.
    Provide travel recommendations, itinerary suggestions, and answer questions about destinations.
    If you don't have specific information, you can make general suggestions based on common travel knowledge."""),
    MessagesPlaceholder(variable_name="context"),
    HumanMessage(content="{input}")
])

"""
## Define Helper Functions

Create functions to handle context retrieval, response generation, and addition to Mem0:
"""
logger.info("## Define Helper Functions")

def retrieve_context(query: str, user_id: str) -> List[Dict]:
    """Retrieve relevant context from Mem0"""
    memories = mem0.search(query, user_id=user_id)
    serialized_memories = ' '.join([mem["memory"] for mem in memories.get('results', [])])
    context = [
        {
            "role": "system",
            "content": f"Relevant information: {serialized_memories}"
        },
        {
            "role": "user",
            "content": query
        }
    ]
    return context

def generate_response(input: str, context: List[Dict]) -> str:
    """Generate a response using the language model"""
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "input": input
    })
    return response.content

def save_interaction(user_id: str, user_input: str, assistant_response: str):
    """Save the interaction to Mem0"""
    interaction = [
        {
          "role": "user",
          "content": user_input
        },
        {
            "role": "assistant",
            "content": assistant_response
        }
    ]
    mem0.add(interaction, user_id=user_id)

"""
## Create Chat Turn Function

Implement the main function to manage a single turn of conversation:
"""
logger.info("## Create Chat Turn Function")

def chat_turn(user_input: str, user_id: str) -> str:
    context = retrieve_context(user_input, user_id)

    response = generate_response(user_input, context)

    save_interaction(user_id, user_input, response)

    return response

"""
## Main Interaction Loop

Set up the main program loop for user interaction:
"""
logger.info("## Main Interaction Loop")

if __name__ == "__main__":
    logger.debug("Welcome to your personal Travel Agent Planner! How can I assist you with your travel plans today?")
    user_id = "john"

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            logger.debug("Travel Agent: Thank you for using our travel planning service. Have a great trip!")
            break

        response = chat_turn(user_input, user_id)
        logger.debug(f"Travel Agent: {response}")

"""
## Key Features

1. **Memory Integration**: Uses Mem0 to store and retrieve relevant information from past interactions.
2. **Personalization**: Provides context-aware responses based on user history and preferences.
3. **Flexible Architecture**: LangChain structure allows for easy expansion of the conversation flow.
4. **Continuous Learning**: Each interaction is stored, improving future responses.

## Conclusion

By integrating LangChain with Mem0, you can build a personalized Travel Agent AI that can maintain context across interactions and provide tailored travel recommendations and assistance.

## Help

- For more details on LangChain, visit the [LangChain documentation](https://python.langchain.com/).
- [Mem0 Platform](https://app.mem0.ai/).
- If you need further assistance, please feel free to reach out to us through the following methods:

<Snippet file="get-help.mdx" />
"""
logger.info("## Key Features")

logger.info("\n\n[DONE]", bright=True)