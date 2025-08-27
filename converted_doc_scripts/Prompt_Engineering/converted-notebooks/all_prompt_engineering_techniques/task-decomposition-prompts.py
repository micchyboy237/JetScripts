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
# Task Decomposition in Prompts Tutorial

## Overview

This tutorial explores the concept of task decomposition in prompt engineering, focusing on techniques for breaking down complex tasks and chaining subtasks in prompts. These techniques are essential for effectively leveraging large language models to solve multi-step problems and perform complex reasoning tasks.

## Motivation

As AI language models become more advanced, they are increasingly capable of handling complex tasks. However, these models often perform better when given clear, step-by-step instructions. Task decomposition is a powerful technique that allows us to break down complex problems into smaller, more manageable subtasks. This approach not only improves the model's performance but also enhances the interpretability and reliability of the results.

## Key Components

1. **Breaking Down Complex Tasks**: Techniques for analyzing and dividing complex problems into simpler subtasks.
2. **Chaining Subtasks**: Methods for sequentially connecting multiple subtasks to solve a larger problem.
3. **Prompt Design for Subtasks**: Crafting effective prompts for each decomposed subtask.
4. **Result Integration**: Combining the outputs from individual subtasks to form a comprehensive solution.

## Method Details

The tutorial employs a step-by-step approach to demonstrate task decomposition:

1. **Problem Analysis**: We start by examining a complex task and identifying its component parts.
2. **Subtask Definition**: We define clear, manageable subtasks that collectively address the main problem.
3. **Prompt Engineering**: For each subtask, we create targeted prompts that guide the AI model.
4. **Sequential Execution**: We implement a chain of prompts, where the output of one subtask feeds into the next.
5. **Result Synthesis**: Finally, we combine the outputs from all subtasks to form a comprehensive solution.

Throughout the tutorial, we use practical examples to illustrate these concepts, demonstrating how task decomposition can be applied to various domains such as analysis, problem-solving, and creative tasks.

## Conclusion

By the end of this tutorial, learners will have gained practical skills in:
- Analyzing complex tasks and breaking them down into manageable subtasks
- Designing effective prompts for each subtask
- Chaining prompts to guide an AI model through a multi-step reasoning process
- Integrating results from multiple subtasks to solve complex problems

These skills will enable more effective use of AI language models for complex problem-solving and enhance the overall quality and reliability of AI-assisted tasks.

## Setup

First, let's import the necessary libraries and set up our environment.
"""
logger.info("# Task Decomposition in Prompts Tutorial")


load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

llm = ChatOllama(model="llama3.2")

def run_prompt(prompt, **kwargs):
    """Helper function to run a prompt through the language model.

    Args:
        prompt (str): The prompt template string.
        **kwargs: Keyword arguments to fill the prompt template.

    Returns:
        str: The model's response.
    """
    prompt_template = PromptTemplate(template=prompt, input_variables=list(kwargs.keys()))
    chain = prompt_template | llm
    return chain.invoke(kwargs).content

"""
## Breaking Down Complex Tasks

Let's start with a complex task and break it down into subtasks. We'll use the example of analyzing a company's financial health.
"""
logger.info("## Breaking Down Complex Tasks")

complex_task = """
Analyze the financial health of a company based on the following data:
- Revenue: $10 million
- Net Income: $2 million
- Total Assets: $15 million
- Total Liabilities: $7 million
- Cash Flow from Operations: $3 million
"""

decomposition_prompt = """
Break down the task of analyzing a company's financial health into 3 subtasks. For each subtask, provide a brief description of what it should accomplish.

Task: {task}

Subtasks:
1.
"""

subtasks = run_prompt(decomposition_prompt, task=complex_task)
logger.debug(subtasks)

"""
## Chaining Subtasks in Prompts

Now that we have our subtasks, let's create individual prompts for each and chain them together.
"""
logger.info("## Chaining Subtasks in Prompts")

def analyze_profitability(revenue, net_income):
    """Analyze the company's profitability.

    Args:
        revenue (float): Company's revenue.
        net_income (float): Company's net income.

    Returns:
        str: Analysis of the company's profitability.
    """
    prompt = """
    Analyze the company's profitability based on the following data:
    - Revenue: ${revenue} million
    - Net Income: ${net_income} million

    Calculate the profit margin and provide a brief analysis of the company's profitability.
    """
    return run_prompt(prompt, revenue=revenue, net_income=net_income)

def analyze_liquidity(total_assets, total_liabilities):
    """Analyze the company's liquidity.

    Args:
        total_assets (float): Company's total assets.
        total_liabilities (float): Company's total liabilities.

    Returns:
        str: Analysis of the company's liquidity.
    """
    prompt = """
    Analyze the company's liquidity based on the following data:
    - Total Assets: ${total_assets} million
    - Total Liabilities: ${total_liabilities} million

    Calculate the current ratio and provide a brief analysis of the company's liquidity.
    """
    return run_prompt(prompt, total_assets=total_assets, total_liabilities=total_liabilities)

def analyze_cash_flow(cash_flow):
    """Analyze the company's cash flow.

    Args:
        cash_flow (float): Company's cash flow from operations.

    Returns:
        str: Analysis of the company's cash flow.
    """
    prompt = """
    Analyze the company's cash flow based on the following data:
    - Cash Flow from Operations: ${cash_flow} million

    Provide a brief analysis of the company's cash flow health.
    """
    return run_prompt(prompt, cash_flow=cash_flow)

profitability_analysis = analyze_profitability(10, 2)
liquidity_analysis = analyze_liquidity(15, 7)
cash_flow_analysis = analyze_cash_flow(3)

logger.debug("Profitability Analysis:\n", profitability_analysis)
logger.debug("\nLiquidity Analysis:\n", liquidity_analysis)
logger.debug("\nCash Flow Analysis:\n", cash_flow_analysis)

"""
## Integrating Results

Finally, let's integrate the results from our subtasks to provide an overall analysis of the company's financial health.
"""
logger.info("## Integrating Results")

def integrate_results(profitability, liquidity, cash_flow):
    """Integrate the results from subtasks to provide an overall analysis.

    Args:
        profitability (str): Profitability analysis.
        liquidity (str): Liquidity analysis.
        cash_flow (str): Cash flow analysis.

    Returns:
        str: Overall analysis of the company's financial health.
    """
    prompt = """
    Based on the following analyses, provide an overall assessment of the company's financial health:

    Profitability Analysis:
    {profitability}

    Liquidity Analysis:
    {liquidity}

    Cash Flow Analysis:
    {cash_flow}

    Summarize the key points and give an overall evaluation of the company's financial position.
    """
    return run_prompt(prompt, profitability=profitability, liquidity=liquidity, cash_flow=cash_flow)

overall_analysis = integrate_results(profitability_analysis, liquidity_analysis, cash_flow_analysis)
logger.debug("Overall Financial Health Analysis:\n", overall_analysis)

logger.info("\n\n[DONE]", bright=True)