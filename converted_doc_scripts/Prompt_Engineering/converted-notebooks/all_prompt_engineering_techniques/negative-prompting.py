from dotenv import load_dotenv
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import CustomLogger
from langchain.prompts import PromptTemplate
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
# Negative Prompting and Avoiding Undesired Outputs

## Overview
This tutorial explores the concept of negative prompting and techniques for avoiding undesired outputs when working with large language models. We'll focus on using Ollama's GPT models and the LangChain library to implement these strategies.

## Motivation
As AI language models become more powerful, it's crucial to guide their outputs effectively. Negative prompting allows us to specify what we don't want in the model's responses, helping to refine and control the generated content. This approach is particularly useful when dealing with sensitive topics, ensuring factual accuracy, or maintaining a specific tone or style in the output.

## Key Components
1. Using negative examples to guide the model
2. Specifying exclusions in prompts
3. Implementing constraints using LangChain
4. Evaluating and refining negative prompts

## Method Details
We'll start by setting up our environment with the necessary libraries. Then, we'll explore different techniques for negative prompting:

1. Basic negative examples: We'll demonstrate how to provide examples of undesired outputs to guide the model.
2. Explicit exclusions: We'll use prompts that specifically state what should not be included in the response.
3. Constraint implementation: Using LangChain, we'll create more complex prompts that enforce specific constraints on the output.
4. Evaluation and refinement: We'll discuss methods to assess the effectiveness of our negative prompts and iteratively improve them.

Throughout the tutorial, we'll use practical examples to illustrate these concepts and provide code snippets for implementation.

## Conclusion
By the end of this tutorial, you'll have a solid understanding of negative prompting techniques and how to apply them to avoid undesired outputs from language models. These skills will enable you to create more controlled, accurate, and appropriate AI-generated content for various applications.

## Setup

First, let's import the necessary libraries and set up our environment.
"""
logger.info("# Negative Prompting and Avoiding Undesired Outputs")


load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

llm = ChatOllama(model="llama3.2")


def get_response(prompt):
    """Helper function to get response from the language model."""
    return llm.invoke(prompt).content


"""
## 1. Using Negative Examples

Let's start with a simple example of using negative examples to guide the model's output.
"""
logger.info("## 1. Using Negative Examples")

negative_example_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""Provide a brief explanation of {topic}.
    Do NOT include any of the following in your explanation:
    - Technical jargon or complex terminology
    - Historical background or dates
    - Comparisons to other related topics
    Your explanation should be simple, direct, and focus only on the core concept."""
)

response = get_response(negative_example_prompt.format(topic="photosynthesis"))
logger.debug(response)

"""
## 2. Specifying Exclusions

Now, let's explore how to explicitly specify what should be excluded from the response.
"""
logger.info("## 2. Specifying Exclusions")

exclusion_prompt = PromptTemplate(
    input_variables=["topic", "exclude"],
    template="""Write a short paragraph about {topic}.
    Important: Do not mention or reference anything related to {exclude}."""
)

response = get_response(exclusion_prompt.format(
    topic="the benefits of exercise",
    exclude="weight loss or body image"
))
logger.debug(response)

"""
## 3. Implementing Constraints

Let's use LangChain to create more complex prompts that enforce specific constraints on the output.
"""
logger.info("## 3. Implementing Constraints")

constraint_prompt = PromptTemplate(
    input_variables=["topic", "style", "excluded_words"],
    template="""Write a {style} description of {topic}.
    Constraints:
    1. Do not use any of these words: {excluded_words}
    2. Keep the description under 100 words
    3. Do not use analogies or metaphors
    4. Focus only on factual information"""
)

response = get_response(constraint_prompt.format(
    topic="artificial intelligence",
    style="technical",
    excluded_words="robot, human-like, science fiction"
))
logger.debug(response)

"""
## 4. Evaluation and Refinement

To evaluate and refine our negative prompts, we can create a function that checks if the output adheres to our constraints.
"""
logger.info("## 4. Evaluation and Refinement")


def evaluate_output(output, constraints):
    """Evaluate if the output meets the given constraints."""
    results = {}
    for constraint, check_func in constraints.items():
        results[constraint] = check_func(output)
    return results


constraints = {
    "word_count": lambda x: len(x.split()) <= 100,
    "no_excluded_words": lambda x: all(word not in x.lower() for word in ["robot", "human-like", "science fiction"]),
    "no_analogies": lambda x: not re.search(r"\b(as|like)\b", x, re.IGNORECASE)

}

evaluation_results = evaluate_output(response, constraints)
logger.debug("Evaluation results:", evaluation_results)

if not all(evaluation_results.values()):
    refined_prompt = constraint_prompt.format(
        topic="artificial intelligence",
        style="technical and concise",  # Added 'concise' to address word count
        # Added 'like' and 'as' to avoid analogies
        excluded_words="robot, human-like, science fiction, like, as"
    )
    refined_response = get_response(refined_prompt)
    logger.debug("\nRefined response:\n", refined_response)

    refined_evaluation = evaluate_output(refined_response, constraints)
    logger.debug("\nRefined evaluation results:", refined_evaluation)

logger.info("\n\n[DONE]", bright=True)
