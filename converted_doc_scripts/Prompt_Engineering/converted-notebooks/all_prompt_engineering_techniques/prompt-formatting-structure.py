from dotenv import load_dotenv
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import CustomLogger
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
# Prompt Formatting and Structure Tutorial

## Overview

This tutorial explores various prompt formats and structural elements in prompt engineering, demonstrating their impact on AI model responses. We'll use Ollama's GPT model and the LangChain library to experiment with different prompt structures and analyze their effectiveness.

## Motivation

Understanding how to format and structure prompts is crucial for effective communication with AI models. Well-structured prompts can significantly improve the quality, relevance, and consistency of AI-generated responses. This tutorial aims to provide practical insights into crafting prompts that elicit desired outcomes across various use cases.

## Key Components

1. Different prompt formats (Q&A, dialogue, instructions)
2. Structural elements (headings, bullet points, numbered lists)
3. Comparison of prompt effectiveness
4. Best practices for prompt formatting

## Method Details

We'll use the Ollama API through LangChain to interact with the GPT model. The tutorial will demonstrate:

1. Setting up the environment with necessary libraries
2. Creating various prompt formats (Q&A, dialogue, instructions)
3. Incorporating structural elements like headings and lists
4. Comparing responses from different prompt structures

Throughout the tutorial, we'll use a consistent theme (e.g., explaining a scientific concept) to showcase how different prompt formats and structures can yield varied results.

## Conclusion

By the end of this tutorial, you'll have a solid understanding of how prompt formatting and structure influence AI responses. You'll be equipped with practical techniques to craft more effective prompts, enhancing your ability to communicate with and leverage AI models for various applications.

## Setup

First, let's import the necessary libraries and set up our environment.
"""
logger.info("# Prompt Formatting and Structure Tutorial")


load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

llm = ChatOllama(model="llama3.2")


def get_response(prompt):
    """Helper function to get model response and print it."""
    response = llm.invoke(prompt).content
    logger.debug(response)
    logger.debug("-" * 50)
    return response


"""
## Exploring Different Prompt Formats

Let's explore various prompt formats using the topic of photosynthesis as our consistent theme.

### 1. Question and Answer (Q&A) Format
"""
logger.info("## Exploring Different Prompt Formats")

qa_prompt = """Q: What is photosynthesis?
A:"""

get_response(qa_prompt)

"""
### 2. Dialogue Format
"""
logger.info("### 2. Dialogue Format")

dialogue_prompt = """Student: Can you explain photosynthesis to me?
Teacher: Certainly! Photosynthesis is...
Student: What does a plant need for photosynthesis?
Teacher:"""

get_response(dialogue_prompt)

"""
### 3. Instruction Format
"""
logger.info("### 3. Instruction Format")

instruction_prompt = """Provide a brief explanation of photosynthesis, including its main components and importance."""

get_response(instruction_prompt)

"""
## Impact of Structural Elements

Now, let's examine how structural elements like headings and lists affect the AI's response.

### 1. Using Headings
"""
logger.info("## Impact of Structural Elements")

headings_prompt = """Explain photosynthesis using the following structure:



"""

get_response(headings_prompt)

"""
### 2. Using Bullet Points
"""
logger.info("### 2. Using Bullet Points")

bullet_points_prompt = """List the key components needed for photosynthesis:

•
•
•
"""

get_response(bullet_points_prompt)

"""
### 3. Using Numbered Lists
"""
logger.info("### 3. Using Numbered Lists")

numbered_list_prompt = """Describe the steps of photosynthesis in order:

1.
2.
3.
4.
"""

get_response(numbered_list_prompt)

"""
## Comparing Prompt Effectiveness

Let's compare the effectiveness of different prompt structures for a specific task.
"""
logger.info("## Comparing Prompt Effectiveness")

comparison_prompts = [
    "Explain the importance of photosynthesis for life on Earth.",
    """Explain the importance of photosynthesis for life on Earth. Structure your answer as follows:
    1. Oxygen production
    2. Food chain support
    3. Carbon dioxide absorption""",
    """Q: Why is photosynthesis important for life on Earth?
    A: Photosynthesis is crucial for life on Earth because:
    1.
    2.
    3."""
]

for i, prompt in enumerate(comparison_prompts, 1):
    logger.debug(f"Prompt {i}:")
    get_response(prompt)

logger.info("\n\n[DONE]", bright=True)
