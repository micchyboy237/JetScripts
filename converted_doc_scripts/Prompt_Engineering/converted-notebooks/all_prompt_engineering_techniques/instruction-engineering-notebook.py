from dotenv import load_dotenv
from jet.llm.ollama.base_langchain import ChatOllama
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
# Instruction Engineering Tutorial

## Overview

This tutorial focuses on Instruction Engineering, a crucial aspect of prompt engineering that deals with crafting clear and effective instructions for language models. We'll explore techniques for creating well-structured prompts and balancing specificity with generality to achieve optimal results.

## Motivation

As language models become more advanced, the quality of instructions we provide becomes increasingly important. Well-crafted instructions can significantly improve the model's output, leading to more accurate, relevant, and useful responses. This tutorial aims to equip learners with the skills to create effective instructions that maximize the potential of AI language models.

## Key Components

1. Crafting Clear Instructions: Techniques for writing unambiguous and easily understandable prompts.
2. Effective Instruction Structures: Exploring different ways to format and organize instructions.
3. Balancing Specificity and Generality: Finding the right level of detail in instructions.
4. Iterative Refinement: Techniques for improving instructions based on model outputs.

## Method Details

We'll use the Ollama API and LangChain library to demonstrate instruction engineering techniques. The tutorial will cover:

1. Setting up the environment and necessary libraries.
2. Creating basic instructions and analyzing their effectiveness.
3. Refining instructions for clarity and specificity.
4. Experimenting with different instruction structures.
5. Balancing specific and general instructions for versatile outputs.
6. Iterative improvement of instructions based on model responses.

Throughout the tutorial, we'll use practical examples to illustrate these concepts and provide hands-on experience in crafting effective instructions.

## Conclusion

By the end of this tutorial, learners will have gained practical skills in instruction engineering, including how to craft clear and effective instructions, balance specificity and generality, and iteratively refine prompts for optimal results. These skills are essential for anyone working with AI language models and can significantly enhance the quality and usefulness of AI-generated content across various applications.

## Setup

First, let's import the necessary libraries and set up our environment.
"""
logger.info("# Instruction Engineering Tutorial")


load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

llm = ChatOllama(model="llama3.2")

def get_completion(prompt):
    """Helper function to get model completion."""
    return llm.invoke(prompt).content

"""
## Crafting Clear Instructions

Let's start by examining the importance of clarity in instructions. We'll compare vague and clear instructions to see the difference in model outputs.
"""
logger.info("## Crafting Clear Instructions")

vague_instruction = "Tell me about climate change concisely."
clear_instruction = "Provide a concise summary of the primary causes and effects of climate change, focusing on scientific consensus from the past five years concisely."

logger.debug("Vague Instruction Output:")
logger.debug(get_completion(vague_instruction))

logger.debug("\nClear Instruction Output:")
logger.debug(get_completion(clear_instruction))

"""
## Effective Instruction Structures

Now, let's explore different structures for instructions to see how they affect the model's output.
"""
logger.info("## Effective Instruction Structures")

bullet_structure = """
Explain the process of photosynthesis concisely:
- Define photosynthesis
- List the main components involved
- Describe the steps in order
- Mention its importance for life on Earth
"""

narrative_structure = """
Imagine you're a botanist explaining photosynthesis to a curious student.
Start with a simple definition, then walk through the process step-by-step,
highlighting the key components involved. Conclude by emphasizing why
photosynthesis is crucial for life on Earth. Write it concisely.
"""

logger.debug("Bullet Structure Output:")
logger.debug(get_completion(bullet_structure))

logger.debug("\nNarrative Structure Output:")
logger.debug(get_completion(narrative_structure))

"""
## Balancing Specificity and Generality

Let's experiment with instructions that vary in their level of specificity to understand how this affects the model's responses.
"""
logger.info("## Balancing Specificity and Generality")

specific_instruction = """
Describe the plot of the 1985 film 'Back to the Future', focusing on:
1. The main character's name and his friendship with Dr. Brown
2. The time machine and how it works
3. The specific year the main character travels to and why it's significant
4. The main conflict involving his parents' past
5. How the protagonist resolves the issues and returns to his time
Limit your response to 150 words.
"""

general_instruction = """
Describe the plot of a popular time travel movie from the 1980s. Include:
1. The main characters and their relationships
2. The method of time travel
3. The time period visited and its significance
4. The main conflict or challenge faced
5. How the story is resolved
Keep your response around 150 words.
"""

logger.debug("Specific Instruction Output:")
logger.debug(get_completion(specific_instruction))

logger.debug("\nGeneral Instruction Output:")
logger.debug(get_completion(general_instruction))

"""
## Iterative Refinement

Now, let's demonstrate how to iteratively refine instructions based on the model's output.
"""
logger.info("## Iterative Refinement")

initial_instruction = "Explain how to make a peanut butter and jelly sandwich."

logger.debug("Initial Instruction Output:")
initial_output = get_completion(initial_instruction)
logger.debug(initial_output)

refined_instruction = """
Explain how to make a peanut butter and jelly sandwich, with the following improvements:
1. Specify the type of bread, peanut butter, and jelly to use
2. Include a step about washing hands before starting
3. Mention how to deal with potential allergies
4. Add a tip for storing the sandwich if not eaten immediately
Present the instructions in a numbered list format.
"""

logger.debug("\nRefined Instruction Output:")
refined_output = get_completion(refined_instruction)
logger.debug(refined_output)

"""
## Practical Application

Let's apply what we've learned to create a well-structured, balanced instruction for a more complex task.
"""
logger.info("## Practical Application")

final_instruction = """
Task: Create a brief lesson plan for teaching basic personal finance to high school students.

Instructions:
1. Start with a concise introduction explaining the importance of personal finance.
2. List 3-5 key topics to cover (e.g., budgeting, saving, understanding credit).
3. For each topic:
   a) Provide a brief explanation suitable for teenagers.
   b) Suggest one practical activity or exercise to reinforce the concept.
4. Conclude with a summary and a suggestion for further learning resources.

Format your response as a structured outline. Aim for clarity and engagement,
balancing specific examples with general principles that can apply to various
financial situations. Keep the entire lesson plan to approximately 300 words.
"""

logger.debug("Final Instruction Output:")
logger.debug(get_completion(final_instruction))

logger.info("\n\n[DONE]", bright=True)