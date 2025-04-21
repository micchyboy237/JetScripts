from jet.logger import CustomLogger
from jet.llm.ollama.base import initialize_ollama_settings
from jet.llm.ollama.base_langchain import ChatOllama
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv

    
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

initialize_ollama_settings()

"""
# Building a Conversational Agent with Context Awareness

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

### Building the Conversational Chain
Combine the prompt template with the language model to create a basic conversational chain. Wrap this chain with a history management component that automatically handles the insertion and retrieval of conversation history.

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

This notebook demonstrates how to create a simple conversational agent using LangChain.

### Import required libraries
"""



# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

"""
### Load environment variables and initialize the language model
"""

load_dotenv()
llm = ChatOllama(model="llama3.1")

"""
###  Create a simple in-memory store for chat histories
"""

store = {}

def get_chat_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

"""
### Create the prompt template
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

"""
### Combine the prompt and model into a runnable chain
"""

chain = prompt | llm

"""
### Wrap the chain with message history
"""

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="input",
    history_messages_key="history"
)

"""
### Example usage
"""

session_id = "user_123"


response1 = chain_with_history.invoke(
    {"input": "Hello! How are you?"},
    config={"configurable": {"session_id": session_id}}
)
logger.debug("AI:", response1.content)

response2 = chain_with_history.invoke(
    {"input": "What was my previous message?"},
    config={"configurable": {"session_id": session_id}}
)
logger.debug("AI:", response2.content)

"""
### Print the conversation history
"""

logger.debug("\nConversation History:")
for message in store[session_id].messages:
    logger.debug(f"{message.type}: {message.content}")

logger.info("\n\n[DONE]", bright=True)