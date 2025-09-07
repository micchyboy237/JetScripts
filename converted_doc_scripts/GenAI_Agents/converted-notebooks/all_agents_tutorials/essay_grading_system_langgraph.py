from dotenv import load_dotenv
from jet.llm.ollama.base_langchain import ChatOllama
from jet.logger import CustomLogger
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict
import os
import re
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# Essay Grading System using LangGraph

## Overview
This notebook presents an automated essay grading system implemented using LangGraph and an LLM model. The system evaluates essays based on four key criteria: relevance, grammar, structure, and depth of analysis.

## Motivation
Automated essay grading systems can significantly streamline the assessment process in educational settings, providing consistent and objective evaluations. This implementation aims to demonstrate how large language models and graph-based workflows can be combined to create a sophisticated grading system.

## Key Components
1. State Graph: Defines the workflow of the grading process
2. LLM Model: Provides the underlying language understanding and analysis
3. Grading Functions: Separate functions for each evaluation criterion
4. Conditional Logic: Determines the flow of the grading process based on interim scores

## Method
The system follows a step-by-step approach to grade essays:

1. Content Relevance: Assesses how well the essay addresses the given topic
2. Grammar Check: Evaluates the essay's language usage and grammatical correctness
3. Structure Analysis: Examines the organization and flow of ideas in the essay
4. Depth of Analysis: Gauges the level of critical thinking and insight presented

Each step is conditionally executed based on the scores from previous steps, allowing for early termination of low-quality essays. The final score is a weighted average of all individual component scores.

## Conclusion
This notebook demonstrates a flexible and extensible approach to automated essay grading. By leveraging the power of large language models and a graph-based workflow, it offers a nuanced evaluation of essays that considers multiple aspects of writing quality. This system could be further refined and adapted for various educational contexts, potentially improving the efficiency and consistency of essay assessments.

## System Workflow

<div style="text-align: center;">

<img src="../images/essay_grading_system_langgraph.svg" alt="essay grading system langgraph" style="width:80%; height:auto;">
</div>

## Setup and Imports

This cell imports necessary libraries and sets up the Ollama API key.
"""
logger.info("# Essay Grading System using LangGraph")


load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

"""
## State Definition

This cell defines the State class, which represents the state of our grading process.
"""
logger.info("## State Definition")

class State(TypedDict):
    """Represents the state of the essay grading process."""
    essay: str
    relevance_score: float
    grammar_score: float
    structure_score: float
    depth_score: float
    final_score: float

"""
## Language Model Initialization

This cell initializes the ChatOllama model.
"""
logger.info("## Language Model Initialization")

llm = ChatOllama(model="llama3.2")

"""
## Grading Functions

This cell defines the functions used in the grading process, including score extraction and individual grading components.
"""
logger.info("## Grading Functions")

def extract_score(content: str) -> float:
    """Extract the numeric score from the LLM's response."""
    match = re.search(r'Score:\s*(\d+(\.\d+)?)', content)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not extract score from: {content}")

def check_relevance(state: State) -> State:
    """Check the relevance of the essay."""
    prompt = ChatPromptTemplate.from_template(
        "Analyze the relevance of the following essay to the given topic. "
        "Provide a relevance score between 0 and 1. "
        "Your response should start with 'Score: ' followed by the numeric score, "
        "then provide your explanation.\n\nEssay: {essay}"
    )
    result = llm.invoke(prompt.format(essay=state["essay"]))
    try:
        state["relevance_score"] = extract_score(result.content)
    except ValueError as e:
        logger.debug(f"Error in check_relevance: {e}")
        state["relevance_score"] = 0.0
    return state

def check_grammar(state: State) -> State:
    """Check the grammar of the essay."""
    prompt = ChatPromptTemplate.from_template(
        "Analyze the grammar and language usage in the following essay. "
        "Provide a grammar score between 0 and 1. "
        "Your response should start with 'Score: ' followed by the numeric score, "
        "then provide your explanation.\n\nEssay: {essay}"
    )
    result = llm.invoke(prompt.format(essay=state["essay"]))
    try:
        state["grammar_score"] = extract_score(result.content)
    except ValueError as e:
        logger.debug(f"Error in check_grammar: {e}")
        state["grammar_score"] = 0.0
    return state

def analyze_structure(state: State) -> State:
    """Analyze the structure of the essay."""
    prompt = ChatPromptTemplate.from_template(
        "Analyze the structure of the following essay. "
        "Provide a structure score between 0 and 1. "
        "Your response should start with 'Score: ' followed by the numeric score, "
        "then provide your explanation.\n\nEssay: {essay}"
    )
    result = llm.invoke(prompt.format(essay=state["essay"]))
    try:
        state["structure_score"] = extract_score(result.content)
    except ValueError as e:
        logger.debug(f"Error in analyze_structure: {e}")
        state["structure_score"] = 0.0
    return state

def evaluate_depth(state: State) -> State:
    """Evaluate the depth of analysis in the essay."""
    prompt = ChatPromptTemplate.from_template(
        "Evaluate the depth of analysis in the following essay. "
        "Provide a depth score between 0 and 1. "
        "Your response should start with 'Score: ' followed by the numeric score, "
        "then provide your explanation.\n\nEssay: {essay}"
    )
    result = llm.invoke(prompt.format(essay=state["essay"]))
    try:
        state["depth_score"] = extract_score(result.content)
    except ValueError as e:
        logger.debug(f"Error in evaluate_depth: {e}")
        state["depth_score"] = 0.0
    return state

def calculate_final_score(state: State) -> State:
    """Calculate the final score based on individual component scores."""
    state["final_score"] = (
        state["relevance_score"] * 0.3 +
        state["grammar_score"] * 0.2 +
        state["structure_score"] * 0.2 +
        state["depth_score"] * 0.3
    )
    return state

"""
## Workflow Definition

This cell defines the grading workflow using StateGraph.
"""
logger.info("## Workflow Definition")

workflow = StateGraph(State)

workflow.add_node("check_relevance", check_relevance)
workflow.add_node("check_grammar", check_grammar)
workflow.add_node("analyze_structure", analyze_structure)
workflow.add_node("evaluate_depth", evaluate_depth)
workflow.add_node("calculate_final_score", calculate_final_score)

workflow.add_conditional_edges(
    "check_relevance",
    lambda x: "check_grammar" if x["relevance_score"] > 0.5 else "calculate_final_score"
)
workflow.add_conditional_edges(
    "check_grammar",
    lambda x: "analyze_structure" if x["grammar_score"] > 0.6 else "calculate_final_score"
)
workflow.add_conditional_edges(
    "analyze_structure",
    lambda x: "evaluate_depth" if x["structure_score"] > 0.7 else "calculate_final_score"
)
workflow.add_conditional_edges(
    "evaluate_depth",
    lambda x: "calculate_final_score"
)

workflow.set_entry_point("check_relevance")

workflow.add_edge("calculate_final_score", END)

app = workflow.compile()

"""
## Essay Grading Function

This cell defines the main function to grade an essay using the defined workflow.
"""
logger.info("## Essay Grading Function")

def grade_essay(essay: str) -> dict:
    """Grade the given essay using the defined workflow."""
    initial_state = State(
        essay=essay,
        relevance_score=0.0,
        grammar_score=0.0,
        structure_score=0.0,
        depth_score=0.0,
        final_score=0.0
    )
    result = app.invoke(initial_state)
    return result

"""
## Sample Essay

This cell contains a sample essay for testing the grading system.
"""
logger.info("## Sample Essay")

sample_essay = """
    The Impact of Artificial Intelligence on Modern Society

    Artificial Intelligence (AI) has become an integral part of our daily lives,
    revolutionizing various sectors including healthcare, finance, and transportation.
    This essay explores the profound effects of AI on modern society, discussing both
    its benefits and potential challenges.

    One of the most significant impacts of AI is in the healthcare industry.
    AI-powered diagnostic tools can analyze medical images with high accuracy,
    often surpassing human capabilities. This leads to earlier detection of diseases
    and more effective treatment plans. Moreover, AI algorithms can process vast
    amounts of medical data to identify patterns and insights that might escape
    human observation, potentially leading to breakthroughs in drug discovery and
    personalized medicine.

    In the financial sector, AI has transformed the way transactions are processed
    and monitored. Machine learning algorithms can detect fraudulent activities in
    real-time, enhancing security for consumers and institutions alike. Robo-advisors
    use AI to provide personalized investment advice, democratizing access to
    financial planning services.

    The transportation industry is another area where AI is making significant strides.
    Self-driving cars, powered by complex AI systems, promise to reduce accidents
    caused by human error and provide mobility solutions for those unable to drive.
    In logistics, AI optimizes routing and inventory management, leading to more
    efficient supply chains and reduced environmental impact.

    However, the rapid advancement of AI also presents challenges. There are concerns
    about job displacement as AI systems become capable of performing tasks
    traditionally done by humans. This raises questions about the need for retraining
    and reskilling the workforce to adapt to an AI-driven economy.

    Privacy and ethical concerns also arise with the increasing use of AI. The vast
    amount of data required to train AI systems raises questions about data privacy
    and consent. Additionally, there are ongoing debates about the potential biases
    in AI algorithms and the need for transparent and accountable AI systems.

    In conclusion, while AI offers tremendous benefits and has the potential to solve
    some of humanity's most pressing challenges, it also requires careful consideration
    of its societal implications. As we continue to integrate AI into various aspects
    of our lives, it is crucial to strike a balance between technological advancement
    and ethical considerations, ensuring that the benefits of AI are distributed
    equitably across society.
    """

"""
## Grading the Sample Essay

This cell demonstrates how to use the grading system on the sample essay and display the results.
"""
logger.info("## Grading the Sample Essay")

result = grade_essay(sample_essay)

logger.debug(f"Final Essay Score: {result['final_score']:.2f}\n")
logger.debug(f"Relevance Score: {result['relevance_score']:.2f}")
logger.debug(f"Grammar Score: {result['grammar_score']:.2f}")
logger.debug(f"Structure Score: {result['structure_score']:.2f}")
logger.debug(f"Depth Score: {result['depth_score']:.2f}")

logger.info("\n\n[DONE]", bright=True)