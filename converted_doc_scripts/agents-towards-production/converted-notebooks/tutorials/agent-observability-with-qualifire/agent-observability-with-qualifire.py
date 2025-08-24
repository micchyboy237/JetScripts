from LangGraph.chat_models import init_chat_model
from LangGraph_community.agent_toolkits import SQLDatabaseToolkit
from LangGraph_community.utilities import SQLDatabase
from jet.logger import CustomLogger
from langgraph.prebuilt import create_react_agent
import os
import qualifire
import requests
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
![](https://europe-west1-atp-views-tracker.cloudfunctions.net/working-analytics?notebook=tutorials--agent-observability-with-qualifire--agent-observability-with-qualifire)

# Agent observability with Qualifire 🔥

This notebook walks you through integrating Qualifire with a LangGraph agent to achieve comprehensive observability, including logging, tracing, and insights via OpenTelemetry.

## Overview

Modern AI applications increasingly rely on sophisticated, multi-step AI agents. These agents often involve multiple LLM calls, interactions with various tools, and complex decision-making processes. Gaining clear visibility into these intricate workflows is a significant challenge. On top of all of that you might also encounter hallucinations, poor tool selection quality and other AI related risks.

## Why Qualifire for Agent Observability?

- **End-to-End Tracing**: Track every step of your agent's execution, from initial prompt to final output
- **Real-Time Monitoring**: Get immediate insights into your agent's performance and behavior
- **Debug & Troubleshoot**: Quickly identify and resolve issues in your agent's decision-making process
- **Quality Assurance**: Monitor for hallucinations and ensure high-quality tool selection
- **OpenTelemetry Integration**: Leverage industry-standard observability practices

## Key Methods

1. **Tracing Setup**: Implement distributed tracing to track agent workflows
2. **Logging Integration**: Capture detailed logs of agent operations
3. **Performance Monitoring**: Track response times and resource usage
4. **Quality Metrics**: Measure and monitor agent decision quality

## What you will learn

1. Setup tracing and observability in your LangGraph agent
2. Debug and troubleshoot your agent
3. Get real-time agent observability using Qualifire

<img src="./assets/freddie-observer.png" alt="Freddie Observer" width="200px">

## 1. Setup and Prerequisites

### 1.1. Install Dependencies
"""
logger.info("# Agent observability with Qualifire 🔥")

# !pip install -q -r requirements.txt

"""
### 1.2. Sign up for Qualifire and Get API Key

Before proceeding, make sure you have a Qualifire account and both MLX and Qualifire API keys.

1. Sign up at [https://app.qualifire.ai](https://app.qualifire.ai?utm=agents-towards-production)
2. Complete the onboarding and create your Qualifire API key.

<img src="./assets/api-key-form.png" alt="Freddie Observer" >

<img src="./assets/new-api-key.png" alt="Freddie Observer">

3. Once you see the "waiting for your events..." screen you can proceed with this tutorial.

<img src="./assets/wait-for-logs.png" >

### 1.3. Initialize Qualifire

Add both your Qualifire and MLX API keys below to initialize the Qualifire SDK. This step is crucial as it sets up the automatic OpenTelemetry instrumentation.
The `qualifire.init()` call is sufficient to automatically instrument and configure OpenTelemetry for tracing the LangGraph agent.
"""
logger.info("### 1.2. Sign up for Qualifire and Get API Key")

global QUALIFIRE_API_KEY
QUALIFIRE_API_KEY = "YOUR QUALIFIRE API KEY" #@param {type:"string"}

# global OPENAI_API_KEY
# OPENAI_API_KEY = "YOUR OPENAI API KEY" #@param {type:"string"}

"""
### Check that the Qualifire API key is set.
"""
logger.info("### Check that the Qualifire API key is set.")


if QUALIFIRE_API_KEY == "YOUR QUALIFIRE API KEY":
    logger.debug("Please replace 'YOUR QUALIFIRE API KEY' with your actual key.")
else:
    qualifire.init(
        api_key=QUALIFIRE_API_KEY,
    )
    logger.debug("Qualifire SDK Initialized Successfully!")

"""
### Check that the MLX API key is set.
"""
logger.info("### Check that the MLX API key is set.")

# if OPENAI_API_KEY == "YOUR OPENAI API KEY":
    logger.debug("Please replace 'YOUR OPENAI API KEY' with your actual key for the agent to run.")
else:
    logger.debug("MLX API Key set.")
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

"""
## 2. Define and run the LangGraph agent

The following code defines a LangGraph agent that interacts with a SQL database. We will run this agent to generate some activity, which Qualifire will then trace.

### 2.1. Agent Code
This is an example agent but feel free to replace it with your own agent.


This example agent is a simple chatbot that can answer questions about a given database. in our example we will use the Chinook database. 
The Chinook data model represents a digital media store, including tables for artists, albums, media tracks, invoices and customers.

We are able to ask all sorts of questions about the database, in our example we will ask about the sales agent who made the most sales in 2009.

The AI will then research the database, read schemas and tables and then answer the question.

ℹ️ Note: You don't need to read the example agent code, it is just here to show you an example of how to build an agent.
"""
logger.info("## 2. Define and run the LangGraph agent")



logger.debug("Imported necessary libraries for the agent.")

# if QUALIFIRE_API_KEY == "YOUR QUALIFIRE API KEY" or OPENAI_API_KEY == "YOUR OPENAI API KEY":
#     logger.debug("\nERROR: API keys are not set. Please set QUALIFIRE_API_KEY and OPENAI_API_KEY in the cells above.\n")
else:
    logger.debug("\nAPI keys seem to be set. Proceeding with agent initialization.\n")

    llm = init_chat_model(
        "ollama:gpt-4.1", # This can be ANY model supported by your init_chat_model
#         api_key=OPENAI_API_KEY,
        base_url="https://proxy.qualifire.ai/api/providers/openai/", # Needed for moderation we'll discuss that in the next tutorial
        default_headers={
            "X-Qualifire-API-Key": QUALIFIRE_API_KEY
        },
    )
    logger.debug("LLM Initialized.")

    db_file_path = "./assets/Chinook.db"

    db = SQLDatabase.from_uri(f"sqlite:///{db_file_path}")

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    logger.debug("\nAvailable Tools:")
    for tool in tools:
        logger.debug(f"- {tool.name}: {tool.description}")

    system_prompt_string = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
        dialect=db.dialect,
        top_k=5,
    )

    agent_executor = create_react_agent(
        llm,
        tools,
        prompt=system_prompt_string,
    )

    logger.debug("\nAgent executor created.")

    question = "Which sales agent made the most in sales in 2009?"
    logger.debug(f"\nQuestion: {question}\n")
    logger.debug("Streaming agent response:")
    logger.debug("-" * 30)

    try:
        for step in agent_executor.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values", # "values" provides the full state at each step
        ):
            if "messages" in step and step["messages"]:
                step["messages"][-1].pretty_logger.debug()
                logger.debug("-" * 30)
    except Exception as e:
        logger.debug(f"An error occurred during agent execution: {e}")
        logger.debug("This might be due to API key issues, network problems, or unexpected agent behavior.")
        logger.debug("Please check your API keys and network connection.")
        logger.debug("If the agent itself has an issue, the Qualifire traces (if captured) might provide insights.")

"""
## 3. View Traces in Qualifire

After running the agent, Qualifire (if initialized correctly) will have captured the traces of its execution.

1.  Go to the Qualifire platform: [https://app.qualifire.ai/traces](https://app.qualifire.ai/traces?utm=agents-towards-production)
2.  You should see the trace from your agent run. It might take a few moments for traces to appear.
3.  Click on "results" to explore the trace and see the detailed logs, execution flow, and any insights Qualifire provides.


### What to look for:
*   **Our new trace:** The trace you just created should appear in the list of traces.
*   **Spans:** Each significant operation (LLM call, tool execution, database query) should appear as a span.
*   **LLM Interaction Details:** For spans representing LLM calls, look for attributes containing the prompt, response, token counts, and model used.
*   **Tool Calls:** Observe how tools are called, their inputs, and outputs.
*   **Errors:** If any errors occurred, they should be visible within the relevant spans.

### The traces table

<img src="./assets/traces-table.png">

### The trace overview

<img src="./assets/full-trace.png">

## 4. Understanding the Observability Data

Qualifire leverages OpenTelemetry to capture a rich set of observability data. Here's what you can typically analyze:

*   **End-to-End Trace:** Visualize the entire lifecycle of a request to your agent. This includes the initial input, calls to the LLM, any tools used by the agent (like the SQLDatabaseToolkit), and the final response.

*   **LLM Interactions:**
    *   **Prompts & Completions:** See the exact prompts sent to the LLM and the completions received.
    *   **Model & Parameters:** Confirm which model was used (e.g., `gpt-4.1`) and other parameters like temperature or max tokens if they are captured.
    *   **Token Usage:** Monitor token consumption for cost management and to ensure you're within context limits.
*   **Tool Usage:**
    *   **Tool Input/Output:** See what data was passed to each tool and what the tool returned. This helps verify if tools are behaving as expected.
*   **Performance Metrics:**
    *   **Latency Analysis:** Pinpoint which parts of your agent's workflow are taking the most time (e.g., LLM response time, database query time, tool execution time).
*   **Error Analysis:**
    *   **Error Messages:** When errors occur, Qualifire can capture detailed error messages, associating them with the specific operation that failed.

## 5. Conclusion

In this tutorial, you've learned how to:
1.  Initialize the Qualifire SDK in your Python application with a single line of code.
2.  Run a LangGraph agent, with Qualifire automatically capturing observability data via OpenTelemetry in the background.
3.  Navigate to the Qualifire platform to view and analyze the traces, logs, and insights generated by your agent.

Using Qualifire provides deep visibility into your agent's operations, making it easier to debug issues, optimize performance, understand LLM interactions, and ensure your agent is behaving as expected. This is a crucial step towards building robust, production-ready AI agents.

We encourage you to explore further:
*   Test with different types of agents and LLMs.
*   Examine the various details Qualifire provides for different operations.
*   Consider how these observability features can fit into your MLOps lifecycle for agents.

### Thank you for completing the tutorial! 🙏
we'd like to offer you 1 free month of the Pro plan to help you get started with Qualifire. use code `NIR1MONTH` at checkout

For more details visit [https://qualifire.ai](https://qualifire.ai?utm=agents-towards-production).
"""
logger.info("## 3. View Traces in Qualifire")

logger.info("\n\n[DONE]", bright=True)