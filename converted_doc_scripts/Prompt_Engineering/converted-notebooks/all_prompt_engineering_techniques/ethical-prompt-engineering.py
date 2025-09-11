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
# Ethical Considerations in Prompt Engineering

## Overview

This tutorial explores the ethical dimensions of prompt engineering, focusing on two critical aspects: avoiding biases in prompts and creating inclusive and fair prompts. As AI language models become increasingly integrated into various applications, ensuring ethical use becomes paramount.

## Motivation

AI language models, trained on vast amounts of data, can inadvertently perpetuate or amplify existing biases. Prompt engineers play a crucial role in mitigating these biases and promoting fairness. This tutorial aims to equip learners with the knowledge and tools to create more ethical and inclusive prompts.

## Key Components

1. Understanding biases in AI
2. Techniques for identifying biases in prompts
3. Strategies for creating inclusive prompts
4. Methods for evaluating fairness in AI outputs
5. Practical examples and exercises

## Method Details

This tutorial employs a combination of theoretical explanations and practical demonstrations:

1. We begin by setting up the necessary environment, including the Ollama API and LangChain library.
2. We explore common types of biases in AI and how they can manifest in prompts.
3. Through examples, we demonstrate how to identify and mitigate biases in prompts.
4. We introduce techniques for creating inclusive prompts that consider diverse perspectives.
5. We implement methods to evaluate the fairness of AI-generated outputs.
6. Throughout the tutorial, we provide exercises for hands-on learning and application of ethical prompt engineering principles.

## Conclusion

By the end of this tutorial, learners will have gained:
1. An understanding of the ethical implications of prompt engineering
2. Skills to identify and mitigate biases in prompts
3. Techniques for creating inclusive and fair prompts
4. Methods to evaluate and improve the ethical quality of AI outputs
5. Practical experience in applying ethical considerations to real-world prompt engineering scenarios

This knowledge will empower prompt engineers to create more responsible and equitable AI applications, contributing to the development of AI systems that benefit all members of society.

## Setup

First, let's import the necessary libraries and set up our environment.
"""
logger.info("# Ethical Considerations in Prompt Engineering")


load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

llm = ChatOllama(model="llama3.2")


def get_model_response(prompt):
    """Helper function to get model response."""
    return llm.invoke(prompt).content


"""
## Understanding Biases in AI

Let's start by examining how biases can manifest in AI responses. We'll use a potentially biased prompt and analyze the output.
"""
logger.info("## Understanding Biases in AI")

biased_prompt = "Describe a typical programmer."
biased_response = get_model_response(biased_prompt)
logger.debug("Potentially biased response:")
logger.debug(biased_response)

"""
## Identifying and Mitigating Biases

Now, let's create a more inclusive prompt and compare the results.
"""
logger.info("## Identifying and Mitigating Biases")

inclusive_prompt = PromptTemplate(
    input_variables=["profession"],
    template="Describe the diverse range of individuals who work as {profession}, emphasizing the variety in their backgrounds, experiences, and characteristics."
)

inclusive_response = (inclusive_prompt | llm).invoke(
    {"profession": "computer programmers"}).content
logger.debug("More inclusive response:")
logger.debug(inclusive_response)

"""
## Creating Inclusive Prompts

Let's explore techniques for creating prompts that encourage diverse and inclusive responses.
"""
logger.info("## Creating Inclusive Prompts")


def create_inclusive_prompt(topic):
    """Creates an inclusive prompt template for a given topic."""
    return PromptTemplate(
        input_variables=["topic"],
        template="Provide a balanced and inclusive perspective on {topic}, considering diverse viewpoints, experiences, and cultural contexts."
    )


topics = ["leadership", "family structures", "beauty standards"]

for topic in topics:
    prompt = create_inclusive_prompt(topic)
    response = (prompt | llm).invoke({"topic": topic}).content
    logger.debug(f"Inclusive perspective on {topic}:")
    logger.debug(response)
    logger.debug("\n" + "-"*50 + "\n")

"""
## Evaluating Fairness in AI Outputs

Now, let's implement a simple method to evaluate the fairness of AI-generated outputs.
"""
logger.info("## Evaluating Fairness in AI Outputs")


def evaluate_fairness(text):
    """Evaluates the fairness of a given text."""
    evaluation_prompt = PromptTemplate(
        input_variables=["text"],
        template="Evaluate the following text for fairness and inclusivity. Identify any potential biases or exclusionary language. Provide a fairness score from 1 to 10, where 10 is most fair and inclusive:\n\nText: {text}\n\nEvaluation:"
    )
    return (evaluation_prompt | llm).invoke({"text": text}).content


sample_text = "In the corporate world, strong leaders are often characterized by their decisiveness and ability to command respect."
fairness_evaluation = evaluate_fairness(sample_text)
logger.debug("Fairness Evaluation:")
logger.debug(fairness_evaluation)

"""
## Practical Exercise

Let's apply what we've learned to improve a potentially biased prompt.
"""
logger.info("## Practical Exercise")

biased_prompt = "Describe the ideal candidate for a high-stress executive position."

logger.debug("Original prompt:")
logger.debug(biased_prompt)
logger.debug("\nOriginal response:")
logger.debug(get_model_response(biased_prompt))

improved_prompt = PromptTemplate(
    input_variables=["position"],
    template="Describe a range of qualities and skills that could make someone successful in a {position}, considering diverse backgrounds, experiences, and leadership styles. Emphasize the importance of work-life balance and mental health."
)

logger.debug("\nImproved prompt:")
logger.debug(improved_prompt.format(position="high-stress executive position"))
logger.debug("\nImproved response:")
logger.debug((improved_prompt | llm).invoke(
    {"position": "high-stress executive position"}).content)

fairness_score = evaluate_fairness((improved_prompt | llm).invoke(
    {"position": "high-stress executive position"}).content)
logger.debug("\nFairness evaluation of improved response:")
logger.debug(fairness_score)

logger.info("\n\n[DONE]", bright=True)
