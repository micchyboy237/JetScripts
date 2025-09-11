from dotenv import load_dotenv
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import CustomLogger
from langchain.output_parsers import RegexParser
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
# Constrained and Guided Generation Tutorial

## Overview

This tutorial explores the concepts of constrained and guided generation in the context of large language models. We'll focus on techniques to set up constraints for model outputs and implement rule-based generation using Ollama's GPT models and the LangChain library.

## Motivation

While large language models are powerful tools for generating text, they sometimes produce outputs that are too open-ended or lack specific desired characteristics. Constrained and guided generation techniques allow us to exert more control over the model's outputs, making them more suitable for specific tasks or adhering to certain rules and formats.

## Key Components

1. Setting up constraints for model outputs
2. Implementing rule-based generation
3. Using LangChain's PromptTemplate for structured prompts
4. Leveraging Ollama's GPT models for text generation

## Method Details

We'll use a combination of prompt engineering techniques and LangChain's utilities to implement constrained and guided generation:

1. We'll start by setting up the environment and importing necessary libraries.
2. We'll create structured prompts using LangChain's PromptTemplate to guide the model's output.
3. We'll implement constraints by specifying rules and formats in our prompts.
4. We'll use Ollama's GPT model to generate text based on our constrained prompts.
5. We'll explore different techniques for rule-based generation, including output parsing and regex-based validation.

## Conclusion

By the end of this tutorial, you'll have a solid understanding of how to implement constrained and guided generation techniques. These skills will enable you to create more controlled and specific outputs from large language models, making them more suitable for a wide range of applications where precise and rule-adherent text generation is required.

## Setup

First, let's import the necessary libraries and set up our environment.
"""
logger.info("# Constrained and Guided Generation Tutorial")


load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

llm = ChatOllama(model="llama3.2")


def display_output(output):
    """Display the model's output in a formatted manner."""
    logger.debug("Model Output:")
    logger.debug("-" * 40)
    logger.debug(output)
    logger.debug("-" * 40)
    logger.debug()


"""
## Setting Up Constraints for Model Outputs

Let's start by creating a constrained prompt that generates a product description with specific requirements.
"""
logger.info("## Setting Up Constraints for Model Outputs")

constrained_prompt = PromptTemplate(
    input_variables=["product", "target_audience", "tone", "word_limit"],
    template="""Create a product description for {product} targeted at {target_audience}.
    Use a {tone} tone and keep it under {word_limit} words.
    The description should include:
    1. A catchy headline
    2. Three key features
    3. A call to action

    Product Description:
    """
)

input_variables = {
    "product": "smart water bottle",
    "target_audience": "health-conscious millennials",
    "tone": "casual and friendly",
    "word_limit": "75"
}

chain = constrained_prompt | llm
output = chain.invoke(input_variables).content
display_output(output)

"""
## Implementing Rule-Based Generation

Now, let's implement a rule-based generation system for creating structured job postings.
"""
logger.info("## Implementing Rule-Based Generation")

job_posting_prompt = PromptTemplate(
    input_variables=["job_title", "company", "location", "experience"],
    template="""Create a job posting for a {job_title} position at {company} in {location}.
    The candidate should have {experience} years of experience.
    Follow these rules:
    1. Start with a brief company description (2 sentences)
    2. List 5 key responsibilities, each starting with an action verb
    3. List 5 required qualifications, each in a single sentence
    4. End with a standardized equal opportunity statement

    Format the output as follows:
    COMPANY: [Company Description]

    RESPONSIBILITIES:
    - [Responsibility 1]
    - [Responsibility 2]
    - [Responsibility 3]
    - [Responsibility 4]
    - [Responsibility 5]

    QUALIFICATIONS:
    - [Qualification 1]
    - [Qualification 2]
    - [Qualification 3]
    - [Qualification 4]
    - [Qualification 5]

    EEO: [Equal Opportunity Statement]
    """
)

input_variables = {
    "job_title": "Senior Software Engineer",
    "company": "TechInnovate Solutions",
    "location": "San Francisco, CA",
    "experience": "5+"
}

chain = job_posting_prompt | llm
output = chain.invoke(input_variables).content
display_output(output)

"""
## Using Regex Parser for Structured Output

Let's use a regex parser to ensure our output adheres to a specific structure.
"""
logger.info("## Using Regex Parser for Structured Output")

regex_parser = RegexParser(
    regex=r"COMPANY:\s*([\s\S]*?)\n\s*RESPONSIBILITIES:\s*([\s\S]*?)\n\s*QUALIFICATIONS:\s*([\s\S]*?)\n\s*EEO:\s*([\s\S]*)",
    output_keys=["company_description", "responsibilities",
                 "qualifications", "eeo_statement"]
)

parsed_job_posting_prompt = PromptTemplate(
    input_variables=["job_title", "company", "location", "experience"],
    template="""Create a job posting for a {job_title} position at {company} in {location}.
    The candidate should have {experience} years of experience.
    Follow these rules:
    1. Start with a brief company description (2 sentences)
    2. List 5 key responsibilities, each starting with an action verb
    3. List 5 required qualifications, each in a single sentence
    4. End with a standardized equal opportunity statement

    Format the output EXACTLY as follows:
    COMPANY: [Company Description]

    RESPONSIBILITIES:
    - [Responsibility 1]
    - [Responsibility 2]
    - [Responsibility 3]
    - [Responsibility 4]
    - [Responsibility 5]

    QUALIFICATIONS:
    - [Qualification 1]
    - [Qualification 2]
    - [Qualification 3]
    - [Qualification 4]
    - [Qualification 5]

    EEO: [Equal Opportunity Statement]
    """
)


def clean_output(output):
    for key, value in output.items():
        if isinstance(value, str):
            output[key] = re.sub(r'\n\s*', '\n', value.strip())
    return output


chain = parsed_job_posting_prompt | llm
raw_output = chain.invoke(input_variables).content

parsed_output = regex_parser.parse(raw_output)
cleaned_output = clean_output(parsed_output)

logger.debug("Parsed Output:")
for key, value in cleaned_output.items():
    logger.debug(f"{key.upper()}:")
    logger.debug(value)
    logger.debug()

"""
## Implementing Additional Constraints

Let's create a more complex constrained generation task: generating a product review with specific criteria.
"""
logger.info("## Implementing Additional Constraints")

review_prompt = PromptTemplate(
    input_variables=["product", "rating", "pros", "cons", "word_limit"],
    template="""Write a product review for {product} with the following constraints:
    1. The review should have a {rating}-star rating (out of 5)
    2. Include exactly {pros} pros and {cons} cons
    3. Use between 2 and 3 sentences for each pro and con
    4. The entire review should be under {word_limit} words
    5. End with a one-sentence recommendation

    Format the review as follows:
    Rating: [X] out of 5 stars

    Pros:
    1. [Pro 1]
    2. [Pro 2]
    ...

    Cons:
    1. [Con 1]
    2. [Con 2]
    ...

    Recommendation: [One-sentence recommendation]
    """
)

input_variables = {
    "product": "Smartphone X",
    "rating": "4",
    "pros": "3",
    "cons": "2",
    "word_limit": "200"
}

chain = review_prompt | llm
output = chain.invoke(input_variables).content
display_output(output)

logger.info("\n\n[DONE]", bright=True)
