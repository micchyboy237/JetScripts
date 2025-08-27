from dotenv import load_dotenv
from jet.logger import CustomLogger
from jinja2 import Template
import openai
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Prompt Templates and Variables Tutorial (Using Jinja2)

## Overview

This tutorial provides a comprehensive introduction to creating and using prompt templates with variables in the context of AI language models. It focuses on leveraging Python and the Jinja2 templating engine to create flexible, reusable prompt structures that can incorporate dynamic content. The tutorial demonstrates how to interact with Ollama's GPT models using these advanced prompting techniques.

## Motivation

As AI language models become increasingly sophisticated, the ability to craft effective prompts becomes crucial for obtaining desired outputs. Prompt templates and variables offer several advantages:

1. **Reusability**: Templates can be reused across different contexts, saving time and ensuring consistency.
2. **Flexibility**: Variables allow for dynamic content insertion, making prompts adaptable to various scenarios.
3. **Complexity Management**: Templates can handle complex structures, including conditional logic and loops, enabling more sophisticated interactions with AI models.
4. **Scalability**: As applications grow, well-structured templates make it easier to manage and maintain large numbers of prompts.

This tutorial aims to equip learners with the knowledge and skills to create powerful, flexible prompt templates, enhancing their ability to work effectively with AI language models.

## Key Components

The tutorial covers several key components:

1. **PromptTemplate Class**: A custom class that wraps Jinja2's Template class, providing a simple interface for creating and using templates.
2. **Jinja2 Templating**: Utilization of Jinja2 for advanced templating features, including variable insertion, conditional statements, and loops.
3. **Ollama API Integration**: Direct use of the Ollama API for sending prompts and receiving responses from GPT models.
4. **Variable Handling**: Techniques for incorporating variables into templates and managing dynamic content.
5. **Conditional Logic**: Implementation of if-else statements within templates to create context-aware prompts.
6. **Advanced Formatting**: Methods for structuring complex prompts, including list formatting and multi-part instructions.

## Method Details

The tutorial employs a step-by-step approach to introduce and demonstrate prompt templating concepts:

1. **Setup and Environment**: The lesson begins by setting up the necessary libraries, including Jinja2 and the Ollama API client.

2. **Basic Template Creation**: Introduction to creating simple templates with single and multiple variables using the custom PromptTemplate class.

3. **Variable Insertion**: Demonstration of how to insert variables into templates using Jinja2's `{{ variable }}` syntax.

4. **Conditional Content**: Exploration of using if-else statements in templates to create prompts that adapt based on provided variables.

5. **List Processing**: Techniques for handling lists of items within templates, including iteration and formatting.

6. **Advanced Templating**: Demonstration of more complex template structures, including nested conditions, loops, and multi-part prompts.

7. **Dynamic Instruction Generation**: Creation of templates that can generate structured instructions based on multiple input variables.

8. **API Integration**: Throughout the tutorial, examples show how to use the templates with the Ollama API to generate responses from GPT models.

The methods are presented with practical examples, progressing from simple to more complex use cases. Each concept is explained theoretically and then demonstrated with a practical application.

## Conclusion

This tutorial provides a solid foundation in creating and using prompt templates with variables, leveraging the power of Jinja2 for advanced templating features. By the end of the lesson, learners will have gained:

1. Understanding of the importance and applications of prompt templates in AI interactions.
2. Practical skills in creating reusable, flexible prompt templates.
3. Knowledge of how to incorporate variables and conditional logic into prompts.
4. Experience in structuring complex prompts for various use cases.
5. Insight into integrating templated prompts with the Ollama API.

These skills enable more sophisticated and efficient interactions with AI language models, opening up possibilities for creating more advanced, context-aware AI applications. The techniques learned can be applied to a wide range of scenarios, from simple query systems to complex, multi-turn conversational agents.

## Setup
"""
logger.info("# Prompt Templates and Variables Tutorial (Using Jinja2)")


load_dotenv()

# openai.api_key = os.getenv('OPENAI_API_KEY')

def get_completion(prompt, model="llama3.2"):
    ''' Get a completion from the Ollama API
    Args:
        prompt (str): The prompt to send to the API
        model (str): The model to use for the completion
    Returns:
        str: The completion text
    '''
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

"""
## 1. Creating Reusable Prompt Templates

We'll create a PromptTemplate class that uses Jinja2 for templating:
"""
logger.info("## 1. Creating Reusable Prompt Templates")

class PromptTemplate:
    ''' A class to represent a template for generating prompts with variables
    Attributes:
        template (str): The template string with variables
        input_variables (list): A list of the variable names in the template
    '''
    def __init__(self, template, input_variables):
        self.template = Template(template)
        self.input_variables = input_variables

    def format(self, **kwargs):
        return self.template.render(**kwargs)

simple_template = PromptTemplate(
    template="Provide a brief explanation of {{ topic }}.",
    input_variables=["topic"]
)

complex_template = PromptTemplate(
    template="Explain the concept of {{ concept }} in the field of {{ field }} to a {{ audience }} audience, concisely.",
    input_variables=["concept", "field", "audience"]
)

logger.debug("Simple Template Result:")
prompt = simple_template.format(topic="photosynthesis")
logger.debug(get_completion(prompt))

logger.debug("\n" + "-"*50 + "\n")

logger.debug("Complex Template Result:")
prompt = complex_template.format(
    concept="neural networks",
    field="artificial intelligence",
    audience="beginner"
)
logger.debug(get_completion(prompt))

"""
## 2. Using Variables for Dynamic Content

Now let's explore more advanced uses of variables, including conditional content:
"""
logger.info("## 2. Using Variables for Dynamic Content")

conditional_template = PromptTemplate(
    template="My name is {{ name }} and I am {{ age }} years old. "
              "{% if profession %}I work as a {{ profession }}.{% else %}I am currently not employed.{% endif %} "
              "Can you give me career advice based on this information? answer concisely.",
    input_variables=["name", "age", "profession"]
)

logger.debug("Conditional Template Result (with profession):")
prompt = conditional_template.format(
    name="Alex",
    age="28",
    profession="software developer"
)
logger.debug(get_completion(prompt))

logger.debug("\nConditional Template Result (without profession):")
prompt = conditional_template.format(
    name="Sam",
    age="22",
    profession=""
)
logger.debug(get_completion(prompt))

logger.debug("\n" + "-"*50 + "\n")

list_template = PromptTemplate(
    template="Categorize these items into groups: {{ items }}. Provide the categories and the items in each category.",
    input_variables=["items"]
)

logger.debug("List Template Result:")
prompt = list_template.format(
    items="apple, banana, carrot, hammer, screwdriver, pliers, novel, textbook, magazine"
)
logger.debug(get_completion(prompt))

"""
## Advanced Template Techniques

Let's explore some more advanced techniques for working with prompt templates and variables:
"""
logger.info("## Advanced Template Techniques")

list_format_template = PromptTemplate(
    template="Analyze the following list of items:\n"
              "{% for item in items.split(',') %}"
              "- {{ item.strip() }}\n"
              "{% endfor %}"
              "\nProvide a summary of the list and suggest any patterns or groupings.",
    input_variables=["items"]
)


logger.debug("Formatted List Template Result:")
prompt = list_format_template.format(
    items="Python, JavaScript, HTML, CSS, React, Django, Flask, Node.js"
)
logger.debug(get_completion(prompt))

logger.debug("\n" + "-"*50 + "\n")

dynamic_instruction_template = PromptTemplate(
    template="Task: {{ task }}\n"
              "Context: {{ context }}\n"
              "Constraints: {{ constraints }}\n\n"
              "Please provide a solution that addresses the task, considers the context, and adheres to the constraints.",
    input_variables=["task", "context", "constraints"]
)

logger.debug("Dynamic Instruction Template Result:")
prompt = dynamic_instruction_template.format(
    task="Design a logo for a tech startup",
    context="The startup focuses on AI-driven healthcare solutions",
    constraints="Must use blue and green colors, and should be simple enough to be recognizable when small"
)
logger.debug(get_completion(prompt))

logger.info("\n\n[DONE]", bright=True)