from collections import Counter
from dotenv import load_dotenv
from jet.llm.ollama.base_langchain import ChatOllama
from jet.logger import CustomLogger
from langchain.prompts import PromptTemplate
import os
import random
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Self-Consistency and Multiple Paths of Reasoning Tutorial

## Overview

This tutorial explores the concept of self-consistency and multiple paths of reasoning in prompt engineering. We'll focus on techniques for generating diverse reasoning paths and aggregating results to improve the quality and reliability of AI-generated answers.

## Motivation

Large language models can sometimes produce inconsistent or unreliable outputs. By leveraging multiple reasoning paths and aggregating results, we can enhance the robustness and accuracy of AI-generated responses. This approach is particularly useful for complex problem-solving tasks where a single path of reasoning might be insufficient or prone to errors.

## Key Components

1. Generating multiple reasoning paths
2. Aggregating results for better answers
3. Implementing self-consistency checks
4. Applying these techniques to various problem-solving scenarios

## Method Details

Our approach involves the following steps:

1. Setting up the environment with necessary libraries (Ollama and LangChain)
2. Designing prompts that encourage diverse reasoning paths
3. Generating multiple responses using these prompts
4. Implementing aggregation methods to combine and analyze the generated responses
5. Applying self-consistency checks to evaluate the reliability of the results
6. Demonstrating the effectiveness of this approach on various problem types

Throughout the tutorial, we'll use practical examples to illustrate how these techniques can be applied to enhance the quality and reliability of AI-generated answers.

By the end of this tutorial, you'll have a solid understanding of how to implement self-consistency and multiple paths of reasoning in your prompt engineering workflows, leading to more robust and reliable AI-generated responses.

## Conclusion

This tutorial will equipped you with powerful techniques for enhancing the reliability and consistency of AI-generated responses through self-consistency and multiple paths of reasoning. By implementing these methods, you can:

1. Generate diverse problem-solving approaches, reducing the risk of biased or narrow solutions.
2. Aggregate multiple reasoning paths to arrive at more robust and reliable answers.
3. Apply self-consistency checks to evaluate and improve the quality of AI-generated outputs.
4. Adapt these techniques to various problem types, from factual queries to complex reasoning tasks.

Mastering these skills will significantly improve your ability to leverage AI language models for more accurate and trustworthy results across a wide range of applications. As you continue to explore and refine these techniques, you'll be better equipped to handle complex problems and generate high-quality, consistent outputs in your AI-driven projects.

## Setup

First, let's import the necessary libraries and set up our environment.
"""
logger.info("# Self-Consistency and Multiple Paths of Reasoning Tutorial")


load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

llm = ChatOllama(model="llama3.2")

"""
## Generating Multiple Reasoning Paths

Let's create a function that generates multiple reasoning paths for a given problem.
"""
logger.info("## Generating Multiple Reasoning Paths")

def generate_multiple_paths(problem, num_paths=3):
    """
    Generate multiple reasoning paths for a given problem.

    Args:
    problem (str): The problem statement.
    num_paths (int): Number of reasoning paths to generate.

    Returns:
    list: A list of generated reasoning paths.
    """
    prompt_template = PromptTemplate(
        input_variables=["problem", "path_number"],
        template="""Solve the following problem using a unique approach. This is reasoning path {path_number}.
        Problem: {problem}
        Reasoning path {path_number}:"""
    )

    paths = []
    for i in range(num_paths):
        chain = prompt_template | llm
        response = chain.invoke({"problem": problem, "path_number": i+1}).content
        paths.append(response)

    return paths

"""
Now, let's test our function with a sample problem.
"""
logger.info("Now, let's test our function with a sample problem.")

problem = "A ball is thrown upwards with an initial velocity of 20 m/s. How high will it go?"
paths = generate_multiple_paths(problem)

for i, path in enumerate(paths, 1):
    logger.debug(f"Path {i}:\n{path}\n")

"""
## Aggregating Results

Now that we have multiple reasoning paths, let's create a function to aggregate the results and determine the most consistent answer.
"""
logger.info("## Aggregating Results")

def aggregate_results(paths):
    """
    Aggregate results from multiple reasoning paths.

    Args:
    paths (list): List of reasoning paths.

    Returns:
    str: The most consistent answer.
    """
    prompt_template = PromptTemplate(
        input_variables=["paths"],
        template="""Analyze the following reasoning paths and determine the most consistent answer. If there are discrepancies, explain why and provide the most likely correct answer.
        Reasoning paths:
        {paths}

        Most consistent answer:"""
    )

    chain = prompt_template | llm
    response = chain.invoke({"paths": "\n".join(paths)}).content
    return response

"""
Let's apply this aggregation function to our previous results.
"""
logger.info("Let's apply this aggregation function to our previous results.")

aggregated_result = aggregate_results(paths)
logger.debug("Aggregated Result:\n", aggregated_result)

"""
## Self-Consistency Check

To further improve our results, let's implement a self-consistency check that evaluates the reliability of our aggregated answer.
"""
logger.info("## Self-Consistency Check")

def self_consistency_check(problem, aggregated_result):
    """
    Perform a self-consistency check on the aggregated result.

    Args:
    problem (str): The original problem statement.
    aggregated_result (str): The aggregated result to check.

    Returns:
    str: An evaluation of the result's consistency and reliability.
    """
    prompt_template = PromptTemplate(
        input_variables=["problem", "result"],
        template="""Evaluate the consistency and reliability of the following result for the given problem.
        Problem: {problem}
        Result: {result}

        Evaluation (consider factors like logical consistency, adherence to known facts, and potential biases):"""
    )

    chain = prompt_template | llm
    response = chain.invoke({"problem": problem, "result": aggregated_result}).content
    return response

"""
Now, let's apply the self-consistency check to our aggregated result.
"""
logger.info("Now, let's apply the self-consistency check to our aggregated result.")

consistency_evaluation = self_consistency_check(problem, aggregated_result)
logger.debug("Self-Consistency Evaluation:\n", consistency_evaluation)

"""
## Applying to Different Problem Types

Let's demonstrate how this approach can be applied to different types of problems.
"""
logger.info("## Applying to Different Problem Types")

def solve_problem(problem):
    """
    Solve a problem using multiple reasoning paths, aggregation, and self-consistency check.

    Args:
    problem (str): The problem statement.

    Returns:
    tuple: (aggregated_result, consistency_evaluation)
    """
    paths = generate_multiple_paths(problem)
    aggregated_result = aggregate_results(paths)
    consistency_evaluation = self_consistency_check(problem, aggregated_result)
    return aggregated_result, consistency_evaluation

problems = [
    "What is the capital of France?",
    "Explain the concept of supply and demand in economics.",
    "If a train travels at 60 km/h, how long will it take to cover 180 km?"
]

for problem in problems:
    logger.debug(f"Problem: {problem}")
    result, evaluation = solve_problem(problem)
    logger.debug("Aggregated Result:\n", result)
    logger.debug("\nConsistency Evaluation:\n", evaluation)
    logger.debug("\n" + "-"*50 + "\n")

logger.info("\n\n[DONE]", bright=True)