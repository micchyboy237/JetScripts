from dotenv import load_dotenv
from jet.llm.ollama.base_langchain import ChatOllama
from jet.logger import CustomLogger
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Basic Prompt Structures Tutorial

## Overview

This tutorial focuses on two fundamental types of prompt structures:
1. Single-turn prompts
2. Multi-turn prompts (conversations)

We'll use Ollama's GPT model and LangChain to demonstrate these concepts.

## Motivation

Understanding different prompt structures is crucial for effective communication with AI models. Single-turn prompts are useful for quick, straightforward queries, while multi-turn prompts enable more complex, context-aware interactions. Mastering these structures allows for more versatile and effective use of AI in various applications.

## Key Components

1. **Single-turn Prompts**: One-shot interactions with the language model.
2. **Multi-turn Prompts**: Series of interactions that maintain context.
3. **Prompt Templates**: Reusable structures for consistent prompting.
4. **Conversation Chains**: Maintaining context across multiple interactions.

## Method Details

We'll use a combination of Ollama's API and LangChain library to demonstrate these prompt structures. The tutorial will include practical examples and comparisons of different prompt types.

## Setup

First, let's import the necessary libraries and set up our environment.
"""
logger.info("# Basic Prompt Structures Tutorial")


load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY') # Ollama API key
llm = ChatOllama(model="llama3.2")

"""
## 1. Single-turn Prompts

Single-turn prompts are one-shot interactions with the language model. They consist of a single input (prompt) and generate a single output (response).
"""
logger.info("## 1. Single-turn Prompts")

single_turn_prompt = "What are the three primary colors?"
logger.debug(llm.invoke(single_turn_prompt).content)

"""
Now, let's use a PromptTemplate to create a more structured single-turn prompt:
"""
logger.info(
    "Now, let's use a PromptTemplate to create a more structured single-turn prompt:")

structured_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Provide a brief explanation of {topic} and list its three main components."
)

chain = structured_prompt | llm
logger.debug(chain.invoke({"topic": "color theory"}).content)

"""
## 2. Multi-turn Prompts (Conversations)

Multi-turn prompts involve a series of interactions with the language model, allowing for more complex and context-aware conversations.
"""
logger.info("## 2. Multi-turn Prompts (Conversations)")

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

logger.debug(conversation.predict(
    input="Hi, I'm learning about space. Can you tell me about planets?"))
logger.debug(conversation.predict(
    input="What's the largest planet in our solar system?"))
logger.debug(conversation.predict(input="How does its size compare to Earth?"))

"""
Let's compare how single-turn and multi-turn prompts handle a series of related questions:
"""
logger.info(
    "Let's compare how single-turn and multi-turn prompts handle a series of related questions:")

prompts = [
    "What is the capital of France?",
    "What is its population?",
    "What is the city's most famous landmark?"
]

logger.debug("Single-turn responses:")
for prompt in prompts:
    logger.debug(f"Q: {prompt}")
    logger.debug(f"A: {llm.invoke(prompt).content}\n")

logger.debug("Multi-turn responses:")
conversation = ConversationChain(llm=llm, memory=ConversationBufferMemory())
for prompt in prompts:
    logger.debug(f"Q: {prompt}")
    logger.debug(f"A: {conversation.predict(input=prompt)}\n")

"""
## Conclusion

This tutorial has introduced you to the basics of single-turn and multi-turn prompt structures. We've seen how:

1. Single-turn prompts are useful for quick, isolated queries.
2. Multi-turn prompts maintain context across a conversation, allowing for more complex interactions.
3. PromptTemplates can be used to create structured, reusable prompts.
4. Conversation chains in LangChain help manage context in multi-turn interactions.

Understanding these different prompt structures allows you to choose the most appropriate approach for various tasks and create more effective interactions with AI language models.
"""
logger.info("## Conclusion")

logger.info("\n\n[DONE]", bright=True)
