from IPython.display import display, Image
from dotenv import load_dotenv
from jet.llm.mlx.adapters.mlx_langchain_llm_adapter import ChatMLX
from jet.logger import CustomLogger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import StateGraph, END
from typing import Dict, TypedDict
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Building an Intelligent Customer Support Agent with LangGraph

## Overview
This tutorial demonstrates how to create an intelligent customer support agent using LangGraph, a powerful tool for building complex language model workflows. The agent is designed to categorize customer queries, analyze sentiment, and provide appropriate responses or escalate issues when necessary.

## Motivation
In today's fast-paced business environment, efficient and accurate customer support is crucial. Automating the initial stages of customer interaction can significantly reduce response times and improve overall customer satisfaction. This project aims to showcase how advanced language models and graph-based workflows can be combined to create a sophisticated support system that can handle a variety of customer inquiries.

## Key Components
1. **State Management**: Using TypedDict to define and manage the state of each customer interaction.
2. **Query Categorization**: Classifying customer queries into Technical, Billing, or General categories.
3. **Sentiment Analysis**: Determining the emotional tone of customer queries.
4. **Response Generation**: Creating appropriate responses based on the query category and sentiment.
5. **Escalation Mechanism**: Automatically escalating queries with negative sentiment to human agents.
6. **Workflow Graph**: Utilizing LangGraph to create a flexible and extensible workflow.

## Method Details
1. **Initialization**: Set up the environment and import necessary libraries.
2. **State Definition**: Create a structure to hold query information, category, sentiment, and response.
3. **Node Functions**: Implement separate functions for categorization, sentiment analysis, and response generation.
4. **Graph Construction**: Use StateGraph to define the workflow, adding nodes and edges to represent the support process.
5. **Conditional Routing**: Implement logic to route queries based on their category and sentiment.
6. **Workflow Compilation**: Compile the graph into an executable application.
7. **Execution**: Process customer queries through the workflow and retrieve results.

## Conclusion
This tutorial demonstrates the power and flexibility of LangGraph in creating complex, AI-driven workflows. By combining natural language processing capabilities with a structured graph-based approach, we've created a customer support agent that can efficiently handle a wide range of queries. This system can be further extended and customized to meet specific business needs, potentially integrating with existing customer support tools and databases for even more sophisticated interactions.

The approach showcased here has broad applications beyond customer support, illustrating how language models can be effectively orchestrated to solve complex, multi-step problems in various domains.

## Install dependancies to run this tutorial
"""
logger.info("# Building an Intelligent Customer Support Agent with LangGraph")

# !pip install -q langgraph langchain-core langchain-openai python-dotenv ipython

"""
## Import necessary libraries
"""
logger.info("## Import necessary libraries")


load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

"""
## Define State Structure

We define a `State` class to hold the query, category, sentiment, and response for each customer interaction.
"""
logger.info("## Define State Structure")


class State(TypedDict):
    query: str
    category: str
    sentiment: str
    response: str


"""
## Define Node Functions

These functions represent the different stages of processing a customer query.
"""
logger.info("## Define Node Functions")


def categorize(state: State) -> State:
    """Categorize the customer query into Technical, Billing, or General."""
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following customer query into one of these categories: "
        "Technical, Billing, General. Query: {query}"
    )
    chain = prompt | ChatMLX(temperature=0)
    category = chain.invoke({"query": state["query"]}).content
    return {"category": category}


def analyze_sentiment(state: State) -> State:
    """Analyze the sentiment of the customer query as Positive, Neutral, or Negative."""
    prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of the following customer query. "
        "Respond with either 'Positive', 'Neutral', or 'Negative'. Query: {query}"
    )
    chain = prompt | ChatMLX(temperature=0)
    sentiment = chain.invoke({"query": state["query"]}).content
    return {"sentiment": sentiment}


def handle_technical(state: State) -> State:
    """Provide a technical support response to the query."""
    prompt = ChatPromptTemplate.from_template(
        "Provide a technical support response to the following query: {query}"
    )
    chain = prompt | ChatMLX(temperature=0)
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}


def handle_billing(state: State) -> State:
    """Provide a billing support response to the query."""
    prompt = ChatPromptTemplate.from_template(
        "Provide a billing support response to the following query: {query}"
    )
    chain = prompt | ChatMLX(temperature=0)
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}


def handle_general(state: State) -> State:
    """Provide a general support response to the query."""
    prompt = ChatPromptTemplate.from_template(
        "Provide a general support response to the following query: {query}"
    )
    chain = prompt | ChatMLX(temperature=0)
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}


def escalate(state: State) -> State:
    """Escalate the query to a human agent due to negative sentiment."""
    return {"response": "This query has been escalated to a human agent due to its negative sentiment."}


def route_query(state: State) -> str:
    """Route the query based on its sentiment and category."""
    if state["sentiment"] == "Negative":
        return "escalate"
    elif state["category"] == "Technical":
        return "handle_technical"
    elif state["category"] == "Billing":
        return "handle_billing"
    else:
        return "handle_general"


"""
## Create and Configure the Graph

Here we set up the LangGraph, defining nodes and edges to create our customer support workflow.
"""
logger.info("## Create and Configure the Graph")

workflow = StateGraph(State)

workflow.add_node("categorize", categorize)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("handle_technical", handle_technical)
workflow.add_node("handle_billing", handle_billing)
workflow.add_node("handle_general", handle_general)
workflow.add_node("escalate", escalate)

workflow.add_edge("categorize", "analyze_sentiment")
workflow.add_conditional_edges(
    "analyze_sentiment",
    route_query,
    {
        "handle_technical": "handle_technical",
        "handle_billing": "handle_billing",
        "handle_general": "handle_general",
        "escalate": "escalate"
    }
)
workflow.add_edge("handle_technical", END)
workflow.add_edge("handle_billing", END)
workflow.add_edge("handle_general", END)
workflow.add_edge("escalate", END)

workflow.set_entry_point("categorize")

app = workflow.compile()

"""
## Visualize the Graph

This cell generates and displays a visual representation of our LangGraph workflow.
"""
logger.info("## Visualize the Graph")

display(
    Image(
        app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
)

"""
## Run Customer Support Function

This function processes a customer query through our LangGraph workflow.
"""
logger.info("## Run Customer Support Function")


def run_customer_support(query: str) -> Dict[str, str]:
    """Process a customer query through the LangGraph workflow.

    Args:
        query (str): The customer's query

    Returns:
        Dict[str, str]: A dictionary containing the query's category, sentiment, and response
    """
    results = app.invoke({"query": query})
    return {
        "category": results["category"],
        "sentiment": results["sentiment"],
        "response": results["response"]
    }


"""
## Test the Customer Support Agent

Let's test our customer support agent with a sample queries for each kind of query type.
"""
logger.info("## Test the Customer Support Agent")

query = "My internet connection keeps dropping. Can you help?"
result = run_customer_support(query)
logger.debug(f"Query: {query}")
logger.debug(f"Category: {result['category']}")
logger.debug(f"Sentiment: {result['sentiment']}")
logger.debug(f"Response: {result['response']}")
logger.debug("\n")


query = "I need help talking to chatGPT"
result = run_customer_support(query)
logger.debug(f"Query: {query}")
logger.debug(f"Category: {result['category']}")
logger.debug(f"Sentiment: {result['sentiment']}")
logger.debug(f"Response: {result['response']}")
logger.debug("\n")


query = "where can i find my receipt?"
result = run_customer_support(query)
logger.debug(f"Query: {query}")
logger.debug(f"Category: {result['category']}")
logger.debug(f"Sentiment: {result['sentiment']}")
logger.debug(f"Response: {result['response']}")
logger.debug("\n")


query = "What are your business hours?"
result = run_customer_support(query)
logger.debug(f"Query: {query}")
logger.debug(f"Category: {result['category']}")
logger.debug(f"Sentiment: {result['sentiment']}")
logger.debug(f"Response: {result['response']}")

logger.info("\n\n[DONE]", bright=True)
