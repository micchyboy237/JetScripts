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
# Introduction to Prompt Engineering Tutorial

## Overview

This tutorial provides a comprehensive introduction to the fundamental concepts of prompt engineering in the context of AI and language models. It is designed to give learners a solid foundation in understanding how to effectively communicate with and leverage large language models through carefully crafted prompts.

## Motivation

As AI language models become increasingly sophisticated and widely used, the ability to interact with them effectively becomes a crucial skill. Prompt engineering is the key to unlocking the full potential of these models, allowing users to guide AI outputs, improve response quality, and tackle complex tasks. This tutorial aims to equip learners with the essential knowledge and skills to begin their journey in prompt engineering.

## Key Components

The tutorial covers several key components of prompt engineering:

1. **Basic Concepts**: An introduction to what prompt engineering is and why it's important.
2. **Prompt Structures**: Exploration of different ways to structure prompts for various outcomes.
3. **Importance of Prompt Engineering**: Discussion on how prompt engineering impacts AI model performance.
4. **Role in AI and Language Models**: Examination of how prompt engineering fits into the broader context of AI applications.
5. **Practical Examples**: Hands-on demonstrations of prompt engineering techniques.

## Method Details

The tutorial employs a mix of theoretical explanations and practical demonstrations to convey the concepts of prompt engineering:

1. **Setup and Environment**: The lesson begins by setting up the necessary tools, including the Ollama API and LangChain library. This provides a practical environment for experimenting with prompts.

2. **Basic Concept Exploration**: Through simple examples, learners are introduced to how different prompts can yield varying responses from the AI model. This illustrates the fundamental principle of prompt engineering.

3. **Structured Prompts**: The tutorial demonstrates how to create more complex, structured prompts using LangChain's PromptTemplate. This shows how to create reusable prompt structures with variable inputs.

4. **Comparative Analysis**: By presenting multiple prompts on the same topic, the lesson highlights how subtle changes in prompt structure and wording can significantly affect the AI's output.

5. **Problem-Solving Application**: The tutorial explores how prompt engineering can be applied to break down complex problems, guiding the AI through a step-by-step reasoning process.

6. **Limitation Mitigation**: Examples are provided to show how careful prompt design can help overcome some limitations of AI models, such as improving factual accuracy.

Throughout these methods, the tutorial emphasizes the importance of clarity, specificity, and thoughtful design in creating effective prompts.

## Conclusion

This introductory tutorial on prompt engineering lays the groundwork for understanding and applying this crucial skill in AI interactions. By the end of the lesson, learners will have gained:

1. A clear understanding of what prompt engineering is and why it's important.
2. Insight into how different prompt structures can influence AI outputs.
3. Practical experience in crafting prompts for various purposes.
4. Awareness of the role prompt engineering plays in enhancing AI model performance.
5. A foundation for exploring more advanced prompt engineering techniques.

The skills and knowledge gained from this tutorial will enable learners to more effectively harness the power of AI language models, setting the stage for more advanced applications and explorations in the field of artificial intelligence.

## Setup

First, let's import the necessary libraries
"""
logger.info("# Introduction to Prompt Engineering Tutorial")


load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY') # Ollama API key
llm = ChatOllama(model="llama3.2")

"""
## Basic Concepts and Importance

Prompt engineering is the practice of designing and optimizing input prompts for language models to generate desired outputs. It's a crucial skill for effectively leveraging AI models in various applications.

Let's explore the concept with a simple example:
"""
logger.info("## Basic Concepts and Importance")

basic_prompt = "Explain the concept of prompt engineering in one sentence."
logger.debug(llm.invoke(basic_prompt).content)

"""
Now, let's see how a more structured prompt can yield a more detailed response:
"""
logger.info(
    "Now, let's see how a more structured prompt can yield a more detailed response:")

structured_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Provide a definition of {topic}, explain its importance, and list three key benefits."
)

# Combine the prompt template with the language model
chain = structured_prompt | llm
input_variables = {"topic": "prompt engineering"}  # Define the input variables
# Invoke the chain with the input variables
output = chain.invoke(input_variables).content
logger.debug(output)

"""
### Importance of Prompt Engineering

Prompt engineering is important because it allows us to:
1. Improve the quality and relevance of AI-generated outputs
2. Guide language models to perform specific tasks more effectively
3. Overcome limitations and biases in AI models
4. Customize AI responses for different use cases and audiences

Let's demonstrate how different prompts can lead to different outputs on the same topic:
"""
logger.info("### Importance of Prompt Engineering")

prompts = [
    "List 3 applications of AI in healthcare.",
    "Explain how AI is revolutionizing healthcare, with 3 specific examples.",
    "You are a doctor. Describe 3 ways AI has improved your daily work in the hospital."
]

for i, prompt in enumerate(prompts, 1):
    logger.debug(f"\nPrompt {i}:")
    logger.debug(prompt)
    logger.debug("\nResponse:")
    logger.debug(llm.invoke(prompt).content)
    logger.debug("-" * 50)

"""
## Role in AI and Language Models

Prompt engineering plays a crucial role in enhancing the performance and applicability of AI and language models. It helps in:

1. Tailoring model outputs to specific needs
2. Improving the accuracy and relevance of responses
3. Enabling complex task completion
4. Reducing biases and improving fairness in AI outputs

Let's explore how prompt engineering can help in overcoming some limitations of language models:
"""
logger.info("## Role in AI and Language Models")

fact_check_prompt = PromptTemplate(
    input_variables=["statement"],
    template="""Evaluate the following statement for factual accuracy. If it's incorrect, provide the correct information:
    Statement: {statement}
    Evaluation:"""
)

chain = fact_check_prompt | llm
logger.debug(chain.invoke("The capital of France is London.").content)

"""
### Improving Complex Problem-Solving

Prompt engineering can also help in breaking down complex problems and guiding the model through a step-by-step reasoning process:
"""
logger.info("### Improving Complex Problem-Solving")

problem_solving_prompt = PromptTemplate(
    input_variables=["problem"],
    template="""Solve the following problem step by step:
    Problem: {problem}
    Solution:
    1)"""
)

chain = problem_solving_prompt | llm
logger.debug(chain.invoke(
    "Calculate the compound interest on $1000 invested for 5 years at an annual rate of 5%, compounded annually.").content)

logger.info("\n\n[DONE]", bright=True)
