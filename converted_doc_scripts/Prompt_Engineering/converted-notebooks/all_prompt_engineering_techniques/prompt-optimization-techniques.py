from dotenv import load_dotenv
from jet.llm.ollama.base_langchain import ChatOllama
from jet.logger import CustomLogger
from langchain.prompts import PromptTemplate
import numpy as np
import os
import re
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Prompt Optimization Techniques

## Overview

This tutorial explores advanced techniques for optimizing prompts when working with large language models. We focus on two key strategies: A/B testing prompts and iterative refinement. These methods are crucial for improving the effectiveness and efficiency of AI-driven applications.

## Motivation

As AI language models become more sophisticated, the quality of prompts used to interact with them becomes increasingly important. Optimized prompts can lead to more accurate, relevant, and useful responses, enhancing the overall performance of AI applications. This tutorial aims to equip learners with practical techniques to systematically improve their prompts.

## Key Components

1. **A/B Testing Prompts**: A method to compare the effectiveness of different prompt variations.
2. **Iterative Refinement**: A strategy for gradually improving prompts based on feedback and results.
3. **Performance Metrics**: Ways to measure and compare the quality of responses from different prompts.
4. **Practical Implementation**: Hands-on examples using Ollama's GPT model and LangChain.

## Method Details

1. **Setup**: We'll start by setting up our environment with the necessary libraries and API keys.

2. **A/B Testing**: 
   - Define multiple versions of a prompt
   - Generate responses for each version
   - Compare results using predefined metrics

3. **Iterative Refinement**:
   - Start with an initial prompt
   - Generate responses and evaluate
   - Identify areas for improvement
   - Refine the prompt based on insights
   - Repeat the process to continuously enhance the prompt

4. **Performance Evaluation**:
   - Define relevant metrics (e.g., relevance, specificity, coherence)
   - Implement scoring functions
   - Compare scores across different prompt versions

Throughout the tutorial, we'll use practical examples to demonstrate these techniques, providing learners with hands-on experience in prompt optimization.

## Conclusion

By the end of this tutorial, learners will have gained:
1. Practical skills in conducting A/B tests for prompt optimization
2. Understanding of iterative refinement processes for prompts
3. Ability to define and use metrics for evaluating prompt effectiveness
4. Hands-on experience with Ollama and LangChain libraries for prompt optimization

These skills will enable learners to create more effective AI applications by systematically improving their interaction with language models.

## Setup

First, let's import the necessary libraries and set up our environment.
"""
logger.info("# Prompt Optimization Techniques")



load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

llm = ChatOllama(model="llama3.2")

def generate_response(prompt):
    """Generate a response using the language model.

    Args:
        prompt (str): The input prompt.

    Returns:
        str: The generated response.
    """
    return llm.invoke(prompt).content

"""
## A/B Testing Prompts

Let's start with A/B testing by comparing different prompt variations for a specific task.
"""
logger.info("## A/B Testing Prompts")

prompt_a = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms."
)

prompt_b = PromptTemplate(
    input_variables=["topic"],
    template="Provide a beginner-friendly explanation of {topic}, including key concepts and an example."
)

def evaluate_response(response, criteria):
    """Evaluate the quality of a response based on given criteria.

    Args:
        response (str): The generated response.
        criteria (list): List of criteria to evaluate.

    Returns:
        float: The average score across all criteria.
    """
    scores = []
    for criterion in criteria:
        logger.debug(f"Evaluating response based on {criterion}...")
        prompt = f"On a scale of 1-10, rate the following response on {criterion}. Start your response with the numeric score:\n\n{response}"
        response = generate_response(prompt)
        score_match = re.search(r'\d+', response)
        if score_match:
            score = int(score_match.group())
            scores.append(min(score, 10))  # Ensure score is not greater than 10
        else:
            logger.debug(f"Warning: Could not extract numeric score for {criterion}. Using default score of 5.")
            scores.append(5)  # Default score if no number is found
    return np.mean(scores)

topic = "machine learning"
response_a = generate_response(prompt_a.format(topic=topic))
response_b = generate_response(prompt_b.format(topic=topic))

criteria = ["clarity", "informativeness", "engagement"]
score_a = evaluate_response(response_a, criteria)
score_b = evaluate_response(response_b, criteria)

logger.debug(f"Prompt A score: {score_a:.2f}")
logger.debug(f"Prompt B score: {score_b:.2f}")
logger.debug(f"Winning prompt: {'A' if score_a > score_b else 'B'}")

"""
## Iterative Refinement

Now, let's demonstrate the iterative refinement process for improving a prompt.
"""
logger.info("## Iterative Refinement")

def refine_prompt(initial_prompt, topic, iterations=3):
    """Refine a prompt through multiple iterations.

    Args:
        initial_prompt (PromptTemplate): The starting prompt template.
        topic (str): The topic to explain.
        iterations (int): Number of refinement iterations.

    Returns:
        PromptTemplate: The final refined prompt template.
    """
    current_prompt = initial_prompt
    for i in range(iterations):
        try:
            response = generate_response(current_prompt.format(topic=topic))
        except KeyError as e:
            logger.debug(f"Error in iteration {i+1}: Missing key {e}. Adjusting prompt...")
            current_prompt.template = current_prompt.template.replace(f"{{{e.args[0]}}}", "relevant example")
            response = generate_response(current_prompt.format(topic=topic))

        feedback_prompt = f"Analyze the following explanation of {topic} and suggest improvements to the prompt that generated it:\n\n{response}"
        feedback = generate_response(feedback_prompt)

        refine_prompt = f"Based on this feedback: '{feedback}', improve the following prompt template. Ensure to only use the variable {{topic}} in your template:\n\n{current_prompt.template}"
        refined_template = generate_response(refine_prompt)

        current_prompt = PromptTemplate(
            input_variables=["topic"],
            template=refined_template
        )

        logger.debug(f"Iteration {i+1} prompt: {current_prompt.template}")

    return current_prompt

topic = "machine learning"
response_a = generate_response(prompt_a.format(topic=topic))
response_b = generate_response(prompt_b.format(topic=topic))

criteria = ["clarity", "informativeness", "engagement"]
score_a = evaluate_response(response_a, criteria)
score_b = evaluate_response(response_b, criteria)

logger.debug(f"Prompt A score: {score_a:.2f}")
logger.debug(f"Prompt B score: {score_b:.2f}")
logger.debug(f"Winning prompt: {'A' if score_a > score_b else 'B'}")

initial_prompt = prompt_b if score_b > score_a else prompt_a
refined_prompt = refine_prompt(initial_prompt, "machine learning")

logger.debug("\nFinal refined prompt:")
logger.debug(refined_prompt.template)

"""
## Comparing Original and Refined Prompts

Let's compare the performance of the original and refined prompts.
"""
logger.info("## Comparing Original and Refined Prompts")

original_response = generate_response(initial_prompt.format(topic="machine learning"))
refined_response = generate_response(refined_prompt.format(topic="machine learning"))

original_score = evaluate_response(original_response, criteria)
refined_score = evaluate_response(refined_response, criteria)

logger.debug(f"Original prompt score: {original_score:.2f}")
logger.debug(f"Refined prompt score: {refined_score:.2f}")
logger.debug(f"Improvement: {(refined_score - original_score):.2f} points")

logger.info("\n\n[DONE]", bright=True)