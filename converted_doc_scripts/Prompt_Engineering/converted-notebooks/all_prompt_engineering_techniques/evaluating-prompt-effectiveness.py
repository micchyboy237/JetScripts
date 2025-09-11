from dotenv import load_dotenv
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import CustomLogger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Evaluating Prompt Effectiveness

## Overview
This tutorial focuses on methods and techniques for evaluating the effectiveness of prompts in AI language models. We'll explore various metrics for measuring prompt performance and discuss both manual and automated evaluation techniques.

## Motivation
As prompt engineering becomes increasingly crucial in AI applications, it's essential to have robust methods for assessing prompt effectiveness. This enables developers and researchers to optimize their prompts, leading to better AI model performance and more reliable outputs.

## Key Components
1. Metrics for measuring prompt performance
2. Manual evaluation techniques
3. Automated evaluation techniques
4. Practical examples using Ollama and LangChain

## Method Details
We'll start by setting up our environment and introducing key metrics for evaluating prompts. We'll then explore manual evaluation techniques, including human assessment and comparative analysis. Next, we'll delve into automated evaluation methods, utilizing techniques like perplexity scoring and automated semantic similarity comparisons. Throughout the tutorial, we'll provide practical examples using Ollama's GPT models and LangChain library to demonstrate these concepts in action.

## Conclusion
By the end of this tutorial, you'll have a comprehensive understanding of how to evaluate prompt effectiveness using both manual and automated techniques. You'll be equipped with practical tools and methods to optimize your prompts, leading to more efficient and accurate AI model interactions.

## Setup

First, let's import the necessary libraries and set up our environment.
"""
logger.info("# Evaluating Prompt Effectiveness")


load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

llm = ChatOllama(model="llama3.2")

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')


def semantic_similarity(text1, text2):
    """Calculate semantic similarity between two texts using cosine similarity."""
    embeddings = sentence_model.encode([text1, text2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]


"""
## Metrics for Measuring Prompt Performance

Let's define some key metrics for evaluating prompt effectiveness:
"""
logger.info("## Metrics for Measuring Prompt Performance")


def relevance_score(response, expected_content):
    """Calculate relevance score based on semantic similarity to expected content."""
    return semantic_similarity(response, expected_content)


def consistency_score(responses):
    """Calculate consistency score based on similarity between multiple responses."""
    if len(responses) < 2:
        return 1.0  # Perfect consistency if there's only one response
    similarities = []
    for i in range(len(responses)):
        for j in range(i+1, len(responses)):
            similarities.append(semantic_similarity(
                responses[i], responses[j]))
    return np.mean(similarities)


def specificity_score(response):
    """Calculate specificity score based on response length and unique word count."""
    words = response.split()
    unique_words = set(words)
    return len(unique_words) / len(words) if words else 0


"""
## Manual Evaluation Techniques

Manual evaluation involves human assessment of prompt-response pairs. Let's create a function to simulate this process:
"""
logger.info("## Manual Evaluation Techniques")


def manual_evaluation(prompt, response, criteria):
    """Simulate manual evaluation of a prompt-response pair."""
    logger.debug(f"Prompt: {prompt}")
    logger.debug(f"Response: {response}")
    logger.debug("\nEvaluation Criteria:")
    for criterion in criteria:
        score = float(input(f"Score for {criterion} (0-10): "))
        logger.debug(f"{criterion}: {score}/10")
    logger.debug("\nAdditional Comments:")
    comments = input("Enter any additional comments: ")
    logger.debug(f"Comments: {comments}")


prompt = "Explain the concept of machine learning in simple terms."
response = llm.invoke(prompt).content
criteria = ["Clarity", "Accuracy", "Simplicity"]
manual_evaluation(prompt, response, criteria)

"""
## Automated Evaluation Techniques

Now, let's implement some automated evaluation techniques:
"""
logger.info("## Automated Evaluation Techniques")


def automated_evaluation(prompt, response, expected_content):
    """Perform automated evaluation of a prompt-response pair."""
    relevance = relevance_score(response, expected_content)
    specificity = specificity_score(response)

    logger.debug(f"Prompt: {prompt}")
    logger.debug(f"Response: {response}")
    logger.debug(f"\nRelevance Score: {relevance:.2f}")
    logger.debug(f"Specificity Score: {specificity:.2f}")

    return {"relevance": relevance, "specificity": specificity}


prompt = "What are the three main types of machine learning?"
expected_content = "The three main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning."
response = llm.invoke(prompt).content
automated_evaluation(prompt, response, expected_content)

"""
## Comparative Analysis

Let's compare the effectiveness of different prompts for the same task:
"""
logger.info("## Comparative Analysis")


def compare_prompts(prompts, expected_content):
    """Compare the effectiveness of multiple prompts for the same task."""
    results = []
    for prompt in prompts:
        response = llm.invoke(prompt).content
        evaluation = automated_evaluation(prompt, response, expected_content)
        results.append({"prompt": prompt, **evaluation})

    sorted_results = sorted(
        results, key=lambda x: x['relevance'], reverse=True)

    logger.debug("Prompt Comparison Results:")
    for i, result in enumerate(sorted_results, 1):
        logger.debug(f"\n{i}. Prompt: {result['prompt']}")
        logger.debug(f"   Relevance: {result['relevance']:.2f}")
        logger.debug(f"   Specificity: {result['specificity']:.2f}")

    return sorted_results


prompts = [
    "List the types of machine learning.",
    "What are the main categories of machine learning algorithms?",
    "Explain the different approaches to machine learning."
]
expected_content = "The main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning."
compare_prompts(prompts, expected_content)

"""
## Putting It All Together

Now, let's create a comprehensive prompt evaluation function that combines both manual and automated techniques:
"""
logger.info("## Putting It All Together")


def evaluate_prompt(prompt, expected_content, manual_criteria=['Clarity', 'Accuracy', 'Relevance']):
    """Perform a comprehensive evaluation of a prompt using both manual and automated techniques."""
    response = llm.invoke(prompt).content

    logger.debug("Automated Evaluation:")
    auto_results = automated_evaluation(prompt, response, expected_content)

    logger.debug("\nManual Evaluation:")
    manual_evaluation(prompt, response, manual_criteria)

    return {"prompt": prompt, "response": response, **auto_results}


prompt = "Explain the concept of overfitting in machine learning."
expected_content = "Overfitting occurs when a model learns the training data too well, including its noise and fluctuations, leading to poor generalization on new, unseen data."
evaluate_prompt(prompt, expected_content)

logger.info("\n\n[DONE]", bright=True)
