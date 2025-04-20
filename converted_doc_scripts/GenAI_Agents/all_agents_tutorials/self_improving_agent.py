from jet.logger import CustomLogger
from jet.llm.ollama.base import initialize_ollama_settings
from jet.llm.ollama.base_langchain import ChatOllama
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os


logger = CustomLogger(f"{os.path.splitext(os.path.basename(__file__))[0]}.log", overwrite=True)

initialize_ollama_settings()

"""
# Self-Improving Agent Tutorial

## Overview
This tutorial demonstrates the implementation of a Self-Improving Agent using LangChain, a framework for developing applications powered by language models. The agent is designed to engage in conversations, learn from its interactions, and continuously improve its performance over time.

## Motivation
As AI systems become more integrated into our daily lives, there's a growing need for agents that can adapt and improve based on their interactions. This self-improving agent serves as a practical example of how we can create AI systems that don't just rely on their initial training, but continue to evolve and enhance their capabilities through ongoing interactions.

## Key Components

1. **Language Model**: The core of the agent, responsible for generating responses and processing information.
2. **Chat History Management**: Keeps track of conversations for context and learning.
3. **Response Generation**: Produces relevant replies to user inputs.
4. **Reflection Mechanism**: Analyzes past interactions to identify areas for improvement.
5. **Learning System**: Incorporates insights from reflection to enhance future performance.

## Method Details

### Initialization
The agent is initialized with a language model, a conversation store, and a system for managing prompts and chains. This setup allows the agent to maintain context across multiple interactions and sessions.

### Response Generation
When the agent receives input, it considers the current conversation history and any recent insights gained from learning. This context-aware approach allows for more coherent and improving responses over time.

### Reflection Process
After a series of interactions, the agent reflects on its performance. It analyzes the conversation history to identify patterns, potential improvements, and areas where it could have provided better responses.

### Learning Mechanism
Based on the reflections, the agent generates learning points. These are concise summaries of how it can improve, which are then incorporated into its knowledge base and decision-making process for future interactions.

### Continuous Improvement Loop
The cycle of interaction, reflection, and learning creates a feedback loop that allows the agent to continuously refine its responses and adapt to different conversation styles and topics.

## Conclusion
This Self-Improving Agent demonstrates a practical implementation of an AI system that can learn and adapt from its interactions. By combining the power of large language models with mechanisms for reflection and learning, we create an agent that not only provides responses but also improves its capabilities over time.

This approach opens up exciting possibilities for creating more dynamic and adaptable AI assistants, chatbots, and other conversational AI applications. As we continue to refine these techniques, we move closer to AI systems that can truly learn and grow from their experiences, much like humans do.

## Imports and Setup

First, we'll import the necessary libraries and load our environment variables.
"""

load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

"""
## Helper Functions

We'll define helper functions for each capability of our agent.

### Chat History Management
"""

def get_chat_history(store, session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

"""
### Response Generation
"""

def generate_response(chain_with_history, human_input: str, session_id: str, insights: str):
    response = chain_with_history.invoke(
        {"input": human_input, "insights": insights},
        config={"configurable": {"session_id": session_id}}
    )
    return response.content

"""
### Reflection
"""

def reflect(llm, store, session_id: str):
    reflection_prompt = ChatPromptTemplate.from_messages([
        ("system", "Based on the following conversation history, provide insights on how to improve responses:"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Generate insights for improvement:")
    ])
    reflection_chain = reflection_prompt | llm
    history = get_chat_history(store, session_id)
    reflection_response = reflection_chain.invoke({"history": history.messages})
    return reflection_response.content

"""
### Learning
"""

def learn(llm, store, session_id: str, insights: str):
    learning_prompt = ChatPromptTemplate.from_messages([
        ("system", "Based on these insights, update the agent's knowledge and behavior:"),
        ("human", "{insights}"),
        ("human", "Summarize the key points to remember:")
    ])
    learning_chain = learning_prompt | llm
    learned_points = learning_chain.invoke({"insights": insights}).content
    get_chat_history(store, session_id).add_ai_message(f"[SYSTEM] Agent learned: {learned_points}")
    return learned_points

"""
## Self-Improving Agent Class

Now we'll define our `SelfImprovingAgent` class that uses these functions.
"""

class SelfImprovingAgent:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.1")
        self.store = {}
        self.insights = ""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a self-improving AI assistant. Learn from your interactions and improve your performance over time."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
            ("system", "Recent insights for improvement: {insights}")
        ])
        
        self.chain = self.prompt | self.llm
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            lambda session_id: get_chat_history(self.store, session_id),
            input_messages_key="input",
            history_messages_key="history"
        )

    def respond(self, human_input: str, session_id: str):
        return generate_response(self.chain_with_history, human_input, session_id, self.insights)

    def reflect(self, session_id: str):
        self.insights = reflect(self.llm, self.store, session_id)
        return self.insights

    def learn(self, session_id: str):
        self.reflect(session_id)
        return learn(self.llm, self.store, session_id, self.insights)

"""
## Example Usage

Let's create an instance of our agent and interact with it to demonstrate its self-improving capabilities.
"""

agent = SelfImprovingAgent()
session_id = "user_123"

logger.debug("AI:", agent.respond("What's the capital of France?", session_id))

logger.debug("AI:", agent.respond("Can you tell me more about its history?", session_id))

logger.debug("\nReflecting and learning...")
learned = agent.learn(session_id)
logger.debug("Learned:", learned)

logger.debug("\nAI:", agent.respond("What's a famous landmark in this city?", session_id))

logger.debug("AI:", agent.respond("What's another interesting fact about this city?", session_id))

logger.info("\n\n[DONE]", bright=True)