from dotenv import load_dotenv
from jet.llm.ollama.base_langchain import ChatOllama
from jet.logger import CustomLogger
from langchain.prompts import PromptTemplate
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# Simple Question-Answering Agent Tutorial

## Overview
This tutorial introduces a basic Question-Answering (QA) agent using LangChain and Ollama's language model. The agent is designed to understand user queries and provide relevant, concise answers.

## Motivation
In the era of AI-driven interactions, creating a simple QA agent serves as a fundamental stepping stone towards more complex AI systems. This project aims to:
- Demonstrate the basics of AI-driven question-answering
- Introduce key concepts in building AI agents
- Provide a foundation for more advanced agent architectures

## Key Components
1. **Language Model**: Utilizes Ollama's GPT model for natural language understanding and generation.
2. **Prompt Template**: Defines the structure and context for the agent's responses.
3. **LLMChain**: Combines the language model and prompt template for streamlined processing.

## Method Details

### 1. Setup and Initialization
- Import necessary libraries (LangChain, dotenv)
- Load environment variables for API key management
- Initialize the Ollama language model

### 2. Defining the Prompt Template
- Create a template that instructs the AI on its role and response format
- Use the PromptTemplate class to structure the input

### 3. Creating the LLMChain
- Combine the language model and prompt template into an LLMChain
- This chain manages the flow from user input to AI response

### 4. Implementing the Question-Answering Function
- Define a function that takes a user question as input
- Use the LLMChain to process the question and generate an answer

### 5. User Interaction
- In a Jupyter notebook environment, provide cells for:
  - Example usage with a predefined question
  - Interactive input for user questions

## Conclusion
This Simple Question-Answering Agent serves as an entry point into the world of AI agents. By understanding and implementing this basic model, you've laid the groundwork for more sophisticated systems. Future enhancements could include:
- Adding memory to maintain context across multiple questions
- Integrating external knowledge bases for more informed responses
- Implementing more complex decision-making processes

As you progress through more advanced tutorials in this repository, you'll build upon these fundamental concepts to create increasingly capable and intelligent AI agents.

### Import necessary libraries
"""
logger.info("# Simple Question-Answering Agent Tutorial")

load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

"""
### initialize the language model
"""
logger.info("### initialize the language model")

llm = ChatOllama(model="llama3.2")

"""
### Define the prompt template
"""
logger.info("### Define the prompt template")

template = """
You are a helpful AI assistant. Your task is to answer the user's question to the best of your ability.

User's question: {question}

Please provide a clear and concise answer:
"""

prompt = PromptTemplate(template=template, input_variables=["question"])

"""
### Create the LLMChain
"""
logger.info("### Create the LLMChain")

qa_chain = prompt | llm

"""
### Define the get_answer function
"""
logger.info("### Define the get_answer function")

def get_answer(question):
    """
    Get an answer to the given question using the QA chain.
    """
    input_variables = {"question": question}
    response = qa_chain.invoke(input_variables).content
    return response

"""
### Cell 6: Example usage
"""
logger.info("### Cell 6: Example usage")

question = "What is the capital of France?"
answer = get_answer(question)
logger.debug(f"Question: {question}")
logger.debug(f"Answer: {answer}")

"""
### Interactive cell for user questions
"""
logger.info("### Interactive cell for user questions")

user_question = input("Enter your question: ")
user_answer = get_answer(user_question)
logger.debug(f"Answer: {user_answer}")

logger.info("\n\n[DONE]", bright=True)