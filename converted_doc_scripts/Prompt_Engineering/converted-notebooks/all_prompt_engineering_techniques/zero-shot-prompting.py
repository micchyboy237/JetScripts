from dotenv import load_dotenv
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import CustomLogger
from langchain.chains import LLMChain
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
# Zero-Shot Prompting Tutorial

## Overview

This tutorial provides a comprehensive introduction to zero-shot prompting, a powerful technique in prompt engineering that allows language models to perform tasks without specific examples or prior training. We'll explore how to design effective zero-shot prompts and implement strategies using Ollama's GPT models and the LangChain library.

## Motivation

Zero-shot prompting is crucial in modern AI applications as it enables language models to generalize to new tasks without the need for task-specific training data or fine-tuning. This capability significantly enhances the flexibility and applicability of AI systems, allowing them to adapt to a wide range of scenarios and user needs with minimal setup.

## Key Components

1. **Understanding Zero-Shot Learning**: An introduction to the concept and its importance in AI.
2. **Prompt Design Principles**: Techniques for crafting effective zero-shot prompts.
3. **Task Framing**: Methods to frame various tasks for zero-shot performance.
4. **Ollama Integration**: Using Ollama's GPT models for zero-shot tasks.
5. **LangChain Implementation**: Leveraging LangChain for structured zero-shot prompting.

## Method Details

The tutorial will cover several methods for implementing zero-shot prompting:

1. **Direct Task Specification**: Crafting prompts that clearly define the task without examples.
2. **Role-Based Prompting**: Assigning specific roles to the AI to guide its responses.
3. **Format Specification**: Providing output format guidelines in the prompt.
4. **Multi-step Reasoning**: Breaking down complex tasks into simpler zero-shot steps.
5. **Comparative Analysis**: Evaluating different zero-shot prompt structures for the same task.

Throughout the tutorial, we'll use Python code with Ollama and LangChain to demonstrate these techniques practically.

## Conclusion

By the end of this tutorial, learners will have gained:

1. A solid understanding of zero-shot prompting and its applications.
2. Practical skills in designing effective zero-shot prompts for various tasks.
3. Experience in implementing zero-shot techniques using Ollama and LangChain.
4. Insights into the strengths and limitations of zero-shot approaches.
5. A foundation for further exploration and innovation in prompt engineering.

This knowledge will empower learners to leverage AI models more effectively across a wide range of applications, enhancing their ability to solve novel problems and create more flexible AI systems.

## Setup

Let's start by importing the necessary libraries and setting up our environment.
"""
logger.info("# Zero-Shot Prompting Tutorial")


load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

llm = ChatOllama(model="llama3.2")


def create_chain(prompt_template):
    """
    Create a LangChain chain with the given prompt template.

    Args:
        prompt_template (str): The prompt template string.

    Returns:
        LLMChain: A LangChain chain object.
    """
    prompt = PromptTemplate.from_template(prompt_template)
    return prompt | llm


"""
## 1. Direct Task Specification

In this section, we'll explore how to craft prompts that clearly define the task without providing examples. This is the essence of zero-shot prompting.
"""
logger.info("## 1. Direct Task Specification")

direct_task_prompt = """Classify the sentiment of the following text as positive, negative, or neutral.
Do not explain your reasoning, just provide the classification.

Text: {text}

Sentiment:"""

direct_task_chain = create_chain(direct_task_prompt)

texts = [
    "I absolutely loved the movie! The acting was superb.",
    "The weather today is quite typical for this time of year.",
    "I'm disappointed with the service I received at the restaurant."
]

for text in texts:
    result = direct_task_chain.invoke({"text": text}).content
    logger.debug(f"Text: {text}")
    logger.debug(f"Sentiment: {result}")

"""
## 2. Format Specification

Providing output format guidelines in the prompt can help structure the AI's response in a zero-shot scenario.
"""
logger.info("## 2. Format Specification")

format_spec_prompt = """Generate a short news article about {topic}.
Structure your response in the following format:

Headline: [A catchy headline for the article]

Lead: [A brief introductory paragraph summarizing the key points]

Body: [2-3 short paragraphs providing more details]

Conclusion: [A concluding sentence or call to action]"""

format_spec_chain = create_chain(format_spec_prompt)

topic = "The discovery of a new earth-like exoplanet"
result = format_spec_chain.invoke({"topic": topic}).content
logger.debug(result)

"""
## 3. Multi-step Reasoning

For complex tasks, we can break them down into simpler zero-shot steps. This approach can improve the overall performance of the model.
"""
logger.info("## 3. Multi-step Reasoning")

multi_step_prompt = """Analyze the following text for its main argument, supporting evidence, and potential counterarguments.
Provide your analysis in the following steps:

1. Main Argument: Identify and state the primary claim or thesis.
2. Supporting Evidence: List the key points or evidence used to support the main argument.
3. Potential Counterarguments: Suggest possible objections or alternative viewpoints to the main argument.

Text: {text}

Analysis:"""

multi_step_chain = create_chain(multi_step_prompt)

text = """While electric vehicles are often touted as a solution to climate change, their environmental impact is not as straightforward as it seems.
The production of batteries for electric cars requires significant mining operations, which can lead to habitat destruction and water pollution.
Moreover, if the electricity used to charge these vehicles comes from fossil fuel sources, the overall carbon footprint may not be significantly reduced.
However, as renewable energy sources become more prevalent and battery technology improves, electric vehicles could indeed play a crucial role in combating climate change."""

result = multi_step_chain.invoke({"text": text}).content
logger.debug(result)

"""
## 4. Comparative Analysis

Let's compare different zero-shot prompt structures for the same task to evaluate their effectiveness.
"""
logger.info("## 4. Comparative Analysis")


def compare_prompts(task, prompt_templates):
    """
    Compare different prompt templates for the same task.

    Args:
        task (str): The task description or input.
        prompt_templates (dict): A dictionary of prompt templates with their names as keys.
    """
    logger.debug(f"Task: {task}\n")
    for name, template in prompt_templates.items():
        chain = create_chain(template)
        result = chain.invoke({"task": task}).content
        logger.debug(f"{name} Prompt Result:")
        logger.debug(result)
        logger.debug("\n" + "-"*50 + "\n")


task = "Explain concisely the concept of blockchain technology"

prompt_templates = {
    "Basic": "Explain {task}.",
    "Structured": """Explain {task} by addressing the following points:
1. Definition
2. Key features
3. Real-world applications
4. Potential impact on industries"""
}

compare_prompts(task, prompt_templates)

logger.info("\n\n[DONE]", bright=True)
