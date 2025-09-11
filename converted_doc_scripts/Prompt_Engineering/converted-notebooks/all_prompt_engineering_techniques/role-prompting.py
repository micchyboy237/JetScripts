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
# Role Prompting Tutorial

## Overview

This tutorial explores the concept of role prompting in AI language models, focusing on how to assign specific roles to AI models and craft effective role descriptions. We'll use Ollama's GPT model and the LangChain library to demonstrate these techniques.

## Motivation

Role prompting is a powerful technique in prompt engineering that allows us to guide AI models to adopt specific personas or expertise. This approach can significantly enhance the quality and relevance of AI-generated responses, making them more suitable for specific tasks or domains.

## Key Components

1. Role Assignment: Techniques for assigning roles to AI models
2. Role Description Crafting: Strategies for creating effective and detailed role descriptions
3. Context Setting: Methods to provide necessary background information for the role
4. Task Specification: Approaches to clearly define tasks within the assigned role

## Method Details

Our approach involves the following steps:

1. Setting up the environment with necessary libraries (Ollama, LangChain)
2. Creating role-based prompts using LangChain's PromptTemplate
3. Assigning roles to the AI model through carefully crafted prompts
4. Demonstrating how different roles affect the model's responses
5. Exploring techniques for refining and improving role descriptions

We'll use various examples to illustrate how role prompting can be applied in different scenarios, such as technical writing, creative storytelling, and professional advice-giving.

## Conclusion

By the end of this tutorial, you will have a solid understanding of role prompting techniques and how to effectively implement them using Ollama and LangChain. You'll be equipped with the skills to craft compelling role descriptions and leverage them to enhance AI model performance in various applications.

## Setup

First, let's import the necessary libraries and set up our environment.
"""
logger.info("# Role Prompting Tutorial")


load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

llm = ChatOllama(model="llama3.2")

"""
## Basic Role Assignment

Let's start with a simple example of role assignment. We'll create a prompt that assigns the role of a technical writer to the AI model.
"""
logger.info("## Basic Role Assignment")

tech_writer_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""You are a technical writer specializing in creating clear and concise documentation for software products.
    Your task is to write a brief explanation of {topic} for a user manual.
    Please provide a 2-3 sentence explanation that is easy for non-technical users to understand."""
)

chain = tech_writer_prompt | llm
response = chain.invoke({"topic": "cloud computing"})
logger.debug(response.content)

"""
## Crafting Effective Role Descriptions

Now, let's explore how to craft more detailed and effective role descriptions. We'll create a prompt for a financial advisor role with a more comprehensive description.
"""
logger.info("## Crafting Effective Role Descriptions")

financial_advisor_prompt = PromptTemplate(
    input_variables=["client_situation"],
    template="""You are a seasoned financial advisor with over 20 years of experience in personal finance, investment strategies, and retirement planning.
    You have a track record of helping clients from diverse backgrounds achieve their financial goals.
    Your approach is characterized by:
    1. Thorough analysis of each client's unique financial situation
    2. Clear and jargon-free communication of complex financial concepts
    3. Ethical considerations in all recommendations
    4. A focus on long-term financial health and stability

    Given the following client situation, provide a brief (3-4 sentences) financial advice:
    {client_situation}

    Your response should reflect your expertise and adhere to your characteristic approach."""
)

chain = financial_advisor_prompt | llm
response = chain.invoke(
    {"client_situation": "A 35-year-old professional earning $80,000 annually, with $30,000 in savings, no debt, and no retirement plan."})
logger.debug(response.content)

"""
## Comparing Responses with Different Roles

To demonstrate how different roles can affect the AI's responses, let's create prompts for three different roles and compare their outputs on the same topic.
"""
logger.info("## Comparing Responses with Different Roles")

roles = [
    ("Scientist", "You are a research scientist specializing in climate change. Explain the following concept in scientific terms:"),
    ("Teacher", "You are a middle school science teacher. Explain the following concept in simple terms suitable for 12-year-old students:"),
    ("Journalist", "You are a journalist writing for a popular science magazine. Explain the following concept in an engaging and informative manner for a general adult audience:")
]

topic = "The greenhouse effect"

for role, description in roles:
    role_prompt = PromptTemplate(
        input_variables=["topic"],
        template=f"{description} {{topic}}"
    )
    chain = role_prompt | llm
    response = chain.invoke({"topic": topic})
    logger.debug(f"\n{role}'s explanation:\n")
    logger.debug(response.content)
    logger.debug("-" * 50)

"""
## Refining Role Descriptions

Let's explore how to refine role descriptions for more specific outcomes. We'll use a creative writing example, focusing on different storytelling styles.
"""
logger.info("## Refining Role Descriptions")

storyteller_prompt = PromptTemplate(
    input_variables=["style", "scenario"],
    template="""You are a master storyteller known for your ability to adapt to various narrative styles.
    Your current task is to write in the style of {style}.
    Key characteristics of this style include:
    1. {style_char1}
    2. {style_char2}
    3. {style_char3}

    Write a short paragraph (3-4 sentences) in this style about the following scenario:
    {scenario}

    Ensure your writing clearly reflects the specified style."""
)

styles = [
    {
        "name": "Gothic horror",
        "char1": "Atmospheric and ominous descriptions",
        "char2": "Themes of decay, death, and the supernatural",
        "char3": "Heightened emotions and sense of dread"
    },
    {
        "name": "Minimalist realism",
        "char1": "Sparse, concise language",
        "char2": "Focus on everyday, ordinary events",
        "char3": "Subtle implications rather than explicit statements"
    }
]

scenario = "A person enters an empty house at twilight"

for style in styles:
    chain = storyteller_prompt | llm
    response = chain.invoke({
        "style": style["name"],
        "style_char1": style["char1"],
        "style_char2": style["char2"],
        "style_char3": style["char3"],
        "scenario": scenario
    })
    logger.debug(f"\n{style['name']} version:\n")
    logger.debug(response.content)
    logger.debug("-" * 50)

logger.info("\n\n[DONE]", bright=True)
