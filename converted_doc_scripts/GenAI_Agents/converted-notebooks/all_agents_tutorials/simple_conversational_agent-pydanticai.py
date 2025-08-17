from dotenv import load_dotenv
from itertools import chain
from jet.logger import CustomLogger
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Building a Conversational Agent with Context Awareness with PydanticAI

**This tutorial is based off of the LangChain tutorial: `Building a Conversational Agent with Context Awareness`. It demonstrates the same concept using PydanticAI as the agent framework.**

## PydanticAI

[PydanticAI](https://ai.pydantic.dev/) is a new Python agent framework designed to make it less painful to build production grade applications with Generative AI. Developed by the team behind **Pydantic**, it brings the same robust validation and type-safety principles that have made Pydantic a cornerstone for many LLM libraries, including MLX SDK, Anthropic SDK, LangChain, LlamaIndex, and more.

With PydanticAI, control flow and agent composition are handled using **vanilla Python**, allowing you to apply the same development best practices you’d use in any other (non-AI) project.

Key features include:

- **[Validation](https://ai.pydantic.dev/results/#structured-result-validation)** and **[type safety](https://ai.pydantic.dev/agents/#static-type-checking)** powered by Pydantic.
- A **[dependency injection system](https://ai.pydantic.dev/dependencies/)** for defining tools, with demonstrations in upcoming notebooks.
- **[Logfire](https://ai.pydantic.dev/logfire/)**, a debugging and monitoring tool for enhanced observability.
- And much more!

## Overview

This tutorial outlines the process of creating a conversational agent that maintains context across multiple interactions. We'll use a modern AI framework to build an agent capable of engaging in more natural and coherent conversations.

## Motivation
Many simple chatbots lack the ability to maintain context, leading to disjointed and frustrating user experiences. This tutorial aims to solve that problem by implementing a conversational agent that can remember and refer to previous parts of the conversation, enhancing the overall interaction quality.

## Key Components
1. **Language Model**: The core AI component that generates responses.
2. **Prompt Template**: Defines the structure of our conversations.
3. **History Manager**: Manages conversation history and context.
4. **Message Store**: Stores the messages for each conversation session.

## Method Details

### Setting Up the Environment
Begin by setting up the necessary AI framework and ensuring access to a suitable language model. This forms the foundation of our conversational agent.

### Creating the Chat History Store
Implement a system to manage multiple conversation sessions. Each session should be uniquely identifiable and associated with its own message history.

### Defining the Conversation Structure
Create a template that includes:
- A system message defining the AI's role
- A placeholder for conversation history
- The user's input

This structure guides the AI's responses and maintains consistency throughout the conversation.

### Building the Conversational Agent
Combine the prompt template with the language model to create a basic conversational agent. Wrap the agent with a history management component that automatically handles the insertion and retrieval of conversation history.

### Interacting with the Agent
To use the agent, invoke it with a user input and a session identifier. The history manager takes care of retrieving the appropriate conversation history, inserting it into the prompt, and storing new messages after each interaction.

## Conclusion
This approach to creating a conversational agent offers several advantages:
- **Context Awareness**: The agent can refer to previous parts of the conversation, leading to more natural interactions.
- **Simplicity**: The modular design keeps the implementation straightforward.
- **Flexibility**: It's easy to modify the conversation structure or switch to a different language model.
- **Scalability**: The session-based approach allows for managing multiple independent conversations.

With this foundation, you can further enhance the agent by:
- Implementing more sophisticated prompt engineering
- Integrating it with external knowledge bases
- Adding specialized capabilities for specific domains
- Incorporating error handling and conversation repair strategies

By focusing on context management, this conversational agent design significantly improves upon basic chatbot functionality, paving the way for more engaging and helpful AI assistants.

# Conversational Agent Tutorial

This notebook demonstrates how to create a simple conversational agent using PydanticAI.

### Import required libraries
"""
logger.info("# Building a Conversational Agent with Context Awareness with PydanticAI")






# import nest_asyncio
# nest_asyncio.apply()

load_dotenv()
# os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LOGFIRE_IGNORE_NO_CONFIG'] = '1'

agent = Agent(
    model='ollama:llama-3.2-3b-instruct',
    system_prompt='You are a helpful AI assistant.',
)

"""
###  Create a simple in-memory store for chat histories
"""
logger.info("###  Create a simple in-memory store for chat histories")

store: dict[str, list[bytes]] = {}

def create_session_if_not_exists(session_id: str) -> None:
    """Makes sure that `session_id` exists in the chat storage."""
    if session_id not in store:
        store[session_id]: list[ModelMessage] = []

def get_chat_history(session_id: str) -> list[ModelMessage]:
    """Returns the existing chat history."""

    create_session_if_not_exists(session_id)

    return list(chain.from_iterable(
        ModelMessagesTypeAdapter.validate_json(msg_group)
        for msg_group in store[session_id]
    ))

def store_messages_in_history(session_id: str, run_result: AgentRunResult[ModelMessage]) -> None:
    """Stores all new messages from the recent `run` with the model, into the local store.

    Receives a session ID and the results that the model returned, fetches all the new
    messages in `bytes` format and stores them in our local storage.
    """
    create_session_if_not_exists(session_id)

    store[session_id].append(run_result.new_messages_json())

"""
### Wrap the ask with message history
"""
logger.info("### Wrap the ask with message history")

def ask_with_history(user_message: str, user_session_id: str) -> AgentRunResult[ModelMessage]:
    """Asks the chatbot the user's question and stores the new messages in the chat history."""

    chat_history = get_chat_history(user_session_id)

    chat_response: AgentRunResult[ModelMessage] = agent.run_sync(user_message, message_history=chat_history)

    store_messages_in_history(user_session_id, chat_response)

    return chat_response

"""
### Example usage
"""
logger.info("### Example usage")

session_id = 'user_123'

result1 = ask_with_history('Hello! How are you?', session_id)
logger.debug('AI:', result1.data)

result2 = ask_with_history('What was my previous message?', session_id)
logger.debug('AI:', result2.data)

"""
### Print the conversation history
"""
logger.info("### Print the conversation history")

logger.debug('\nConversation History:')
tmp = get_chat_history(session_id)
for message in get_chat_history(session_id):
    logger.debug(f'{message.parts[-1].part_kind}: {message.parts[-1].content}')

logger.info("\n\n[DONE]", bright=True)