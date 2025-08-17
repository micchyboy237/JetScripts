from autogen import ConversableAgent
from jet.logger import CustomLogger
from mem0 import MemoryClient
from openai import MLX
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
Build conversational AI agents with memory capabilities. This integration combines AutoGen for creating AI agents with Mem0 for memory management, enabling context-aware and personalized interactions.

## Overview

In this guide, we'll explore an example of creating a conversational AI system with memory:
- A customer service bot that can recall previous interactions and provide personalized responses.

## Setup and Configuration

Install necessary libraries:
"""
logger.info("## Overview")

pip install pyautogen mem0ai openai

"""
First, we'll import the necessary libraries and set up our configurations.

<Note>Remember to get the Mem0 API key from [Mem0 Platform](https://app.mem0.ai).</Note>
"""
logger.info("First, we'll import the necessary libraries and set up our configurations.")


# OPENAI_API_KEY = 'sk-xxx'  # Replace with your actual MLX API key
MEM0_API_KEY = 'your-mem0-key'  # Replace with your actual Mem0 API key from https://app.mem0.ai
USER_ID = "customer_service_bot"

# os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['MEM0_API_KEY'] = MEM0_API_KEY

memory_client = MemoryClient()
agent = ConversableAgent(
    "chatbot",
#     llm_config={"config_list": [{"model": "gpt-4", "api_key": OPENAI_API_KEY}]},
    code_execution_config=False,
    human_input_mode="NEVER",
)

"""
## Storing Conversations in Memory

Add conversation history to Mem0 for future reference:
"""
logger.info("## Storing Conversations in Memory")

conversation = [
    {"role": "assistant", "content": "Hi, I'm Best Buy's chatbot! How can I help you?"},
    {"role": "user", "content": "I'm seeing horizontal lines on my TV."},
    {"role": "assistant", "content": "I'm sorry to hear that. Can you provide your TV model?"},
    {"role": "user", "content": "It's a Sony - 77\" Class BRAVIA XR A80K OLED 4K UHD Smart Google TV"},
    {"role": "assistant", "content": "Thank you for the information. Let's troubleshoot this issue..."}
]

memory_client.add(messages=conversation, user_id=USER_ID)
logger.debug("Conversation added to memory.")

"""
## Retrieving and Using Memory

Create a function to get context-aware responses based on user's question and previous interactions:
"""
logger.info("## Retrieving and Using Memory")

def get_context_aware_response(question):
    relevant_memories = memory_client.search(question, user_id=USER_ID)
    context = "\n".join([m["memory"] for m in relevant_memories.get('results', [])])

    prompt = f"""Answer the user question considering the previous interactions:
    Previous interactions:
    {context}

    Question: {question}
    """

    reply = agent.generate_reply(messages=[{"content": prompt, "role": "user"}])
    return reply

question = "What was the issue with my TV?"
answer = get_context_aware_response(question)
logger.debug("Context-aware answer:", answer)

"""
## Multi-Agent Conversation

For more complex scenarios, you can create multiple agents:
"""
logger.info("## Multi-Agent Conversation")

manager = ConversableAgent(
    "manager",
    system_message="You are a manager who helps in resolving complex customer issues.",
#     llm_config={"config_list": [{"model": "gpt-4", "api_key": OPENAI_API_KEY}]},
    human_input_mode="NEVER"
)

def escalate_to_manager(question):
    relevant_memories = memory_client.search(question, user_id=USER_ID)
    context = "\n".join([m["memory"] for m in relevant_memories.get('results', [])])

    prompt = f"""
    Context from previous interactions:
    {context}

    Customer question: {question}

    As a manager, how would you address this issue?
    """

    manager_response = manager.generate_reply(messages=[{"content": prompt, "role": "user"}])
    return manager_response

complex_question = "I'm not satisfied with the troubleshooting steps. What else can be done?"
manager_answer = escalate_to_manager(complex_question)
logger.debug("Manager's response:", manager_answer)

"""
## Conclusion

By integrating AutoGen with Mem0, you've created a conversational AI system with memory capabilities. This example demonstrates a customer service bot that can recall previous interactions and provide context-aware responses, with the ability to escalate complex issues to a manager agent.

This integration enables the creation of more intelligent and personalized AI agents for various applications, such as customer support, virtual assistants, and interactive chatbots.

## Help

In case of any questions, please feel free to reach out to us using one of the following methods:

<Snippet file="get-help.mdx" />
"""
logger.info("## Conclusion")

logger.info("\n\n[DONE]", bright=True)