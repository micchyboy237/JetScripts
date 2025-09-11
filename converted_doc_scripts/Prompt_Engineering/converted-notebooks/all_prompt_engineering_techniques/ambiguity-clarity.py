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
# Handling Ambiguity and Improving Clarity in Prompt Engineering

## Overview

This tutorial focuses on two critical aspects of prompt engineering: identifying and resolving ambiguous prompts, and techniques for writing clearer prompts. These skills are essential for effective communication with AI models and obtaining more accurate and relevant responses.

## Motivation

Ambiguity in prompts can lead to inconsistent or irrelevant AI responses, while lack of clarity can result in misunderstandings and inaccurate outputs. By mastering these aspects of prompt engineering, you can significantly improve the quality and reliability of AI-generated content across various applications.

## Key Components

1. Identifying ambiguous prompts
2. Strategies for resolving ambiguity
3. Techniques for writing clearer prompts
4. Practical examples and exercises

## Method Details

We'll use Ollama's GPT model and the LangChain library to demonstrate various techniques for handling ambiguity and improving clarity in prompts. The tutorial will cover:

1. Setting up the environment and necessary libraries
2. Analyzing ambiguous prompts and their potential interpretations
3. Implementing strategies to resolve ambiguity, such as providing context and specifying parameters
4. Exploring techniques for writing clearer prompts, including using specific language and structured formats
5. Practical exercises to apply these concepts in real-world scenarios

## Conclusion

By the end of this tutorial, you'll have a solid understanding of how to identify and resolve ambiguity in prompts, as well as techniques for crafting clearer prompts. These skills will enable you to communicate more effectively with AI models, resulting in more accurate and relevant outputs across various applications.

## Setup

First, let's import the necessary libraries and set up our environment.
"""
logger.info("# Handling Ambiguity and Improving Clarity in Prompt Engineering")


load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

llm = ChatOllama(model="llama3.2")

"""
## Identifying Ambiguous Prompts

Let's start by examining some ambiguous prompts and analyzing their potential interpretations.
"""
logger.info("## Identifying Ambiguous Prompts")

ambiguous_prompts = [
    "Tell me about the bank.",
    "What's the best way to get to school?",
    "Can you explain the theory?"
]

for prompt in ambiguous_prompts:
    analysis_prompt = f"Analyze the following prompt for ambiguity: '{prompt}'. Explain why it's ambiguous and list possible interpretations."
    logger.debug(f"Prompt: {prompt}")
    logger.debug(llm.invoke(analysis_prompt).content)
    logger.debug("-" * 50)

"""
## Resolving Ambiguity

Now, let's explore strategies for resolving ambiguity in prompts.
"""
logger.info("## Resolving Ambiguity")


def resolve_ambiguity(prompt, context):
    """
    Resolve ambiguity in a prompt by providing additional context.

    Args:
    prompt (str): The original ambiguous prompt
    context (str): Additional context to resolve ambiguity

    Returns:
    str: The AI's response to the clarified prompt
    """
    clarified_prompt = f"{context}\n\nBased on this context, {prompt}"
    return llm.invoke(clarified_prompt).content


ambiguous_prompt = "Tell me about the bank."
contexts = [
    "You are a financial advisor discussing savings accounts.",
    "You are a geographer describing river formations."
]

for context in contexts:
    logger.debug(f"Context: {context}")
    logger.debug(
        f"Clarified response: {resolve_ambiguity(ambiguous_prompt, context)}")
    logger.debug("-" * 50)

"""
## Techniques for Writing Clearer Prompts

Let's explore some techniques for writing clearer prompts to improve AI responses.
"""
logger.info("## Techniques for Writing Clearer Prompts")


def compare_prompt_clarity(original_prompt, improved_prompt):
    """
    Compare the responses to an original prompt and an improved, clearer version.

    Args:
    original_prompt (str): The original, potentially unclear prompt
    improved_prompt (str): An improved, clearer version of the prompt

    Returns:
    tuple: Responses to the original and improved prompts
    """
    original_response = llm.invoke(original_prompt).content
    improved_response = llm.invoke(improved_prompt).content
    return original_response, improved_response


original_prompt = "How do I make it?"
improved_prompt = "Provide a step-by-step guide for making a classic margherita pizza, including ingredients and cooking instructions."

original_response, improved_response = compare_prompt_clarity(
    original_prompt, improved_prompt)

logger.debug("Original Prompt Response:")
logger.debug(original_response)
logger.debug("\nImproved Prompt Response:")
logger.debug(improved_response)

"""
## Structured Prompts for Clarity

Using structured prompts can significantly improve clarity and consistency in AI responses.
"""
logger.info("## Structured Prompts for Clarity")

structured_prompt = PromptTemplate(
    input_variables=["topic", "aspects", "tone"],
    template="""Provide an analysis of {topic} considering the following aspects:
    1. {{aspects[0]}}
    2. {{aspects[1]}}
    3. {{aspects[2]}}

    Present the analysis in a {tone} tone.
    """
)

input_variables = {
    "topic": "the impact of social media on society",
    "aspects": ["communication patterns", "mental health", "information spread"],
    "tone": "balanced and objective"
}

chain = structured_prompt | llm
response = chain.invoke(input_variables).content
logger.debug(response)

"""
## Practical Exercise: Improving Prompt Clarity

Now, let's practice improving the clarity of prompts.
"""
logger.info("## Practical Exercise: Improving Prompt Clarity")

unclear_prompts = [
    "What's the difference?",
    "How does it work?",
    "Why is it important?"
]


def improve_prompt_clarity(unclear_prompt):
    """
    Improve the clarity of a given prompt.

    Args:
    unclear_prompt (str): The original unclear prompt

    Returns:
    str: An improved, clearer version of the prompt
    """
    improvement_prompt = f"The following prompt is unclear: '{unclear_prompt}'. Please provide a clearer, more specific version of this prompt. output just the improved prompt and nothing else."
    return llm.invoke(improvement_prompt).content


for prompt in unclear_prompts:
    improved_prompt = improve_prompt_clarity(prompt)
    logger.debug(f"Original: {prompt}")
    logger.debug(f"Improved: {improved_prompt}")
    logger.debug("-" * 50)

logger.info("\n\n[DONE]", bright=True)
