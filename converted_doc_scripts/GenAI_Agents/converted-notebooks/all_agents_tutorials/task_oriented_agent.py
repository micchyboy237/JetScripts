from dotenv import load_dotenv
from jet.llm.mlx.adapters.mlx_langchain_llm_adapter import ChatMLX
from jet.logger import CustomLogger
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import PromptTemplate
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Tutorial: Building a Text Summarizer and Translator with LangChain

## Overview

This tutorial demonstrates how to create a language model application that summarizes text and translates the summary to Spanish using LangChain. The application uses a combination of custom functions, structured tools, and an agent to process input text efficiently.

## Motivation

In today's data-rich world, the ability to quickly summarize information and translate it into different languages is invaluable. This tutorial aims to show how to leverage language models and the LangChain framework to create a tool that can:

1. Summarize lengthy text
2. Translate the summary to Spanish
3. Do both tasks in a single, streamlined process

This type of tool can be useful for various applications, including content curation, multilingual communication, and rapid information processing.

## Key Components

1. **Custom Functions**: For summarization and translation
2. **Structured Tools**: Wrappers for the custom functions
3. **Prompt Template**: Instructions for the agent
4. **Agent**: Orchestrates the use of tools based on the prompt
5. **Agent Executor**: Runs the agent with specified parameters

## Method Details

### 1. Custom Functions

Two main functions are defined:

- A summarization function that takes input text and returns a summary
- A translation function that takes input text and returns its Spanish translation

Both functions use a PromptTemplate and a language model to perform their tasks.

### 2. Structured Tools

The custom functions are wrapped as StructuredTool objects. This allows the agent to use these functions as tools, providing a name, description, and input schema for each.

### 3. Prompt Template

A PromptTemplate is created with detailed instructions for the agent. It outlines the steps the agent should follow:
1. Summarize the input text
2. Translate the summary to Spanish
3. Format the output with both the English summary and Spanish translation

### 4. Agent and Agent Executor

An agent is created using the tools and prompt. This agent is then wrapped in an AgentExecutor, which manages the execution of the agent. The executor is configured with parameters such as the maximum number of iterations and the early stopping method.

### 5. Running the Agent

A helper function is created to simplify running the agent. This function takes the agent executor and a query, runs the agent, and returns the output. The tutorial demonstrates this with a sample query about pangrams, showing how the entire pipeline works together to process the input text.

## Conclusion

This tutorial demonstrates how to create a powerful text processing tool using LangChain. By combining custom functions, structured tools, and an agent, we've created an application that can summarize text and translate the summary to Spanish in one seamless operation. This approach can be extended to include other languages or text processing tasks, making it a versatile foundation for various natural language processing applications.

The strength of this approach lies in its modularity and flexibility. By using LangChain's components, we can easily modify or extend the functionality of our application. For instance, we could add more languages, implement different summarization techniques, or even incorporate other text processing tasks like sentiment analysis or keyword extraction.

### Import necessary libraries
"""
logger.info(
    "# Tutorial: Building a Text Summarizer and Translator with LangChain")


"""
### Load environment variables and initialize the language model
"""
logger.info("### Load environment variables and initialize the language model")

load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
llm = ChatMLX(model="llama-3.2-3b-instruct", max_tokens=1000, temperature=0)

"""
### first let's define the functions that the agent can use:
"""
logger.info("### first let's define the functions that the agent can use:")


def summarize(text):
    prompt = PromptTemplate(
        input_variables=["text"],  # Specify the input variable
        # Define the template for summarization
        template="Summarize the following text:\n\n{text}\n\nSummary:"
    )
    chain = prompt | llm  # Create a chain by piping the prompt to the language model
    # Invoke the chain with the input text and return the content of the response
    return chain.invoke({"text": text}).content


def translate(text):
    prompt = PromptTemplate(
        input_variables=["text"],  # Specify the input variable
        # Define the template for translation
        template="Translate the following text to Spanish:\n\n{text}\n\nTranslation:"
    )
    chain = prompt | llm  # Create a chain by piping the prompt to the language model
    # Invoke the chain with the input text and return the content of the response
    return chain.invoke({"text": text}).content


class TextInput(BaseModel):
    # Define a text field with a description
    text: str = Field(description="The text to summarize or translate")


text = "The quick brown fox jumps over the lazy dog."
logger.debug(summarize(text))
logger.debug(translate(text))

"""
### Define the tools for the agent
"""
logger.info("### Define the tools for the agent")

tools = [
    StructuredTool.from_function(
        func=summarize,  # The function to be wrapped as a tool
        name="Summarize",  # Name of the tool
        description="Useful for summarizing text",  # Description of what the tool does
        args_schema=TextInput  # The Pydantic model defining the input schema
    ),
    StructuredTool.from_function(
        func=translate,  # The function to be wrapped as a tool
        name="Translate",  # Name of the tool
        # Description of what the tool does
        description="Useful for translating text to Spanish",
        args_schema=TextInput  # The Pydantic model defining the input schema
    )
]

"""
### Initialize the agent
"""
logger.info("### Initialize the agent")

prompt = PromptTemplate(
    # Define the input variables for the prompt
    input_variables=["input", "agent_scratchpad"],
    template="""Summarize the following text and then translate the summary to Spanish:

Text: {input}

Use the following steps:
1. Use the Summarize tool to summarize the text. Pass the entire text as the 'text' argument.
2. Use the Translate tool to translate the summary to Spanish. Pass the summary as the 'text' argument.
3. Immediately after using both tools, respond with the final result in the following format:
   Summary (English): [English summary]
   Translation (Spanish): [Spanish translation]

Do not use any tools after providing the formatted output.

{agent_scratchpad}"""  # Define the template for the agent's instructions
)

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,  # The agent to execute
    tools=tools,  # The tools available to the agent
    verbose=True,  # Enable verbose output
    max_iterations=3,  # Set maximum number of iterations
    early_stopping_method="force"  # Force stop after max_iterations
)


def run_agent_with_query(agent_executor, query):
    """
    Execute the agent with a given query and return the output.

    Args:
        agent_executor (AgentExecutor): The configured AgentExecutor to run.
        query (str): The input text to be processed by the agent.

    Returns:
        str: The output generated by the agent after processing the query.
    """
    result = agent_executor.invoke({"input": query})

    return result['output']


"""
###  Example usage
"""
logger.info("###  Example usage")

query = """The quick brown fox jumps over the lazy dog. This sentence is often used as a pangram in typography
to display font examples, as it contains every letter of the English alphabet. However, it's not the only pangram
in existence. Another example is 'Pack my box with five dozen liquor jugs', which is shorter but less commonly used."""

result = run_agent_with_query(agent_executor, query)

logger.debug("\nQuery:")
logger.debug(query)

logger.debug("\nResult:")
logger.debug(result)

logger.info("\n\n[DONE]", bright=True)
