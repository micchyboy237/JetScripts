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
# Chain of Thought (CoT) Prompting Tutorial

## Overview

This tutorial introduces Chain of Thought (CoT) prompting, a powerful technique in prompt engineering that encourages AI models to break down complex problems into step-by-step reasoning processes. We'll explore how to implement CoT prompting using Ollama's GPT models and the LangChain library.

## Motivation

As AI language models become more advanced, there's an increasing need to guide them towards producing more transparent, logical, and verifiable outputs. CoT prompting addresses this need by encouraging models to show their work, much like how humans approach complex problem-solving tasks. This technique not only improves the accuracy of AI responses but also makes them more interpretable and trustworthy.

## Key Components

1. **Basic CoT Prompting**: Introduction to the concept and simple implementation.
2. **Advanced CoT Techniques**: Exploring more sophisticated CoT approaches.
3. **Comparative Analysis**: Examining the differences between standard and CoT prompting.
4. **Problem-Solving Applications**: Applying CoT to various complex tasks.

## Method Details

The tutorial will guide learners through the following methods:

1. **Setting up the environment**: We'll start by importing necessary libraries and setting up the Ollama API.

2. **Basic CoT Implementation**: We'll create simple CoT prompts and compare their outputs to standard prompts.

3. **Advanced CoT Techniques**: We'll explore more complex CoT strategies, including multi-step reasoning and self-consistency checks.

4. **Practical Applications**: We'll apply CoT prompting to various problem-solving scenarios, such as mathematical word problems and logical reasoning tasks.


## Conclusion

By the end of this tutorial, learners will have a solid understanding of Chain of Thought prompting and its applications. They will be equipped with practical skills to implement CoT techniques in various scenarios, improving the quality and interpretability of AI-generated responses. This knowledge will be valuable for anyone working with large language models, from developers and researchers to business analysts and decision-makers relying on AI-powered insights.

## Setup

Let's start by importing the necessary libraries and setting up our environment.
"""
logger.info("# Chain of Thought (CoT) Prompting Tutorial")


load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = ChatOllama(model="llama3.2")

"""
## Basic Chain of Thought Prompting

Let's start with a simple example to demonstrate the difference between a standard prompt and a Chain of Thought prompt.
"""
logger.info("## Basic Chain of Thought Prompting")

standard_prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question concisely: {question}."
)

cot_prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question step by step concisely: {question}"
)

standard_chain = standard_prompt | llm
cot_chain = cot_prompt | llm

question = "If a train travels 120 km in 2 hours, what is its average speed in km/h?"

standard_response = standard_chain.invoke(question).content
cot_response = cot_chain.invoke(question).content

logger.debug("Standard Response:")
logger.debug(standard_response)
logger.debug("\nChain of Thought Response:")
logger.debug(cot_response)

"""
## Advanced Chain of Thought Techniques

Now, let's explore a more advanced CoT technique that encourages multi-step reasoning.
"""
logger.info("## Advanced Chain of Thought Techniques")

advanced_cot_prompt = PromptTemplate(
    input_variables=["question"],
    template="""Solve the following problem step by step. For each step:
1. State what you're going to calculate
2. Write the formula you'll use (if applicable)
3. Perform the calculation
4. Explain the result

Question: {question}

Solution:"""
)

advanced_cot_chain = advanced_cot_prompt | llm

complex_question = "A car travels 150 km at 60 km/h, then another 100 km at 50 km/h. What is the average speed for the entire journey?"

advanced_cot_response = advanced_cot_chain.invoke(complex_question).content
logger.debug(advanced_cot_response)

"""
## Comparative Analysis

Let's compare the effectiveness of standard prompting vs. CoT prompting on a more challenging problem.
"""
logger.info("## Comparative Analysis")

challenging_question = """
A cylindrical water tank with a radius of 1.5 meters and a height of 4 meters is 2/3 full.
If water is being added at a rate of 10 liters per minute, how long will it take for the tank to overflow?
Give your answer in hours and minutes, rounded to the nearest minute.
(Use 3.14159 for π and 1000 liters = 1 cubic meter)"""

standard_response = standard_chain.invoke(challenging_question).content
cot_response = advanced_cot_chain.invoke(challenging_question).content

logger.debug("Standard Response:")
logger.debug(standard_response)
logger.debug("\nChain of Thought Response:")
logger.debug(cot_response)

"""
## Problem-Solving Applications

Now, let's apply CoT prompting to a more complex logical reasoning task.
"""
logger.info("## Problem-Solving Applications")

llm = ChatOllama(model="llama3.2")

logical_reasoning_prompt = PromptTemplate(
    input_variables=["scenario"],
    template="""Analyze the following logical puzzle thoroughly. Follow these steps in your analysis:

List the Facts:

Summarize all the given information and statements clearly.
Identify all the characters or elements involved.
Identify Possible Roles or Conditions:

Determine all possible roles, behaviors, or states applicable to the characters or elements (e.g., truth-teller, liar, alternator).
Note the Constraints:

Outline any rules, constraints, or relationships specified in the puzzle.
Generate Possible Scenarios:

Systematically consider all possible combinations of roles or conditions for the characters or elements.
Ensure that all permutations are accounted for.
Test Each Scenario:

For each possible scenario:
Assume the roles or conditions you've assigned.
Analyze each statement based on these assumptions.
Check for consistency or contradictions within the scenario.
Eliminate Inconsistent Scenarios:

Discard any scenarios that lead to contradictions or violate the constraints.
Keep track of the reasoning for eliminating each scenario.
Conclude the Solution:

Identify the scenario(s) that remain consistent after testing.
Summarize the findings.
Provide a Clear Answer:

State definitively the role or condition of each character or element.
Explain why this is the only possible solution based on your analysis.
Scenario:

{scenario}

Analysis:""")

logical_reasoning_chain = logical_reasoning_prompt | llm

logical_puzzle = """In a room, there are three people: Amy, Bob, and Charlie.
One of them always tells the truth, one always lies, and one alternates between truth and lies.
Amy says, 'Bob is a liar.'
Bob says, 'Charlie alternates between truth and lies.'
Charlie says, 'Amy and I are both liars.'
Determine the nature (truth-teller, liar, or alternator) of each person."""

logical_reasoning_response = logical_reasoning_chain.invoke(logical_puzzle).content
logger.debug(logical_reasoning_response)

logger.info("\n\n[DONE]", bright=True)