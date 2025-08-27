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
# Prompts for Specific Tasks

## Overview

This tutorial explores the creation and use of prompts for specific tasks in natural language processing. We'll focus on four key areas: text summarization, question-answering, code generation, and creative writing. Using Ollama's GPT model and the LangChain library, we'll demonstrate how to craft effective prompts for each of these tasks.

## Motivation

As language models become more advanced, the ability to design task-specific prompts becomes increasingly valuable. Well-crafted prompts can significantly enhance the performance of AI models across various applications, from summarizing long documents to generating code and fostering creativity in writing. This tutorial aims to provide practical insights into prompt engineering for these diverse tasks.

## Key Components

1. Text Summarization Prompts: Techniques for condensing long texts while retaining key information.
2. Question-Answering Prompts: Strategies for extracting specific information from given contexts.
3. Code Generation Prompts: Methods for guiding AI models to produce accurate and functional code.
4. Creative Writing Prompts: Approaches to stimulating imaginative and engaging written content.

## Method Details

This tutorial uses the Ollama GPT-4 model through the LangChain library. For each task type, we'll follow these steps:

1. Design a prompt template tailored to the specific task.
2. Implement the prompt using LangChain's PromptTemplate.
3. Execute the prompt with sample inputs.
4. Analyze the output and discuss potential improvements or variations.

We'll explore how different prompt structures and phrasings can influence the model's output for each task type. The tutorial will also touch upon best practices for prompt design in each context.

## Conclusion

By the end of this tutorial, you'll have a solid understanding of how to create effective prompts for text summarization, question-answering, code generation, and creative writing tasks. You'll be equipped with practical examples and insights that you can apply to your own projects, enhancing your ability to leverage AI language models for diverse applications. Remember that prompt engineering is both an art and a science - experimentation and iteration are key to finding the most effective prompts for your specific needs.

## Setup

First, let's import the necessary libraries and set up our environment.
"""
logger.info("# Prompts for Specific Tasks")


load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

llm = ChatOllama(model="llama3.2")

"""
## 1. Text Summarization Prompts

Let's start with creating a prompt for text summarization. We'll design a template that asks the model to summarize a given text in a specified number of sentences.
"""
logger.info("## 1. Text Summarization Prompts")

summarization_template = PromptTemplate(
    input_variables=["text", "num_sentences"],
    template="Summarize the following text in {num_sentences} sentences:\n\n{text}"
)

long_text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans.
AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving".
This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.
AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars, automated decision-making and competing at the highest level in strategic game systems.
As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect.
"""

summarization_chain = summarization_template | llm
summary = summarization_chain.invoke({"text": long_text, "num_sentences": 3}).content

logger.debug("Summary:")
logger.debug(summary)

"""
## 2. Question-Answering Prompts

Next, let's create a prompt for question-answering tasks. We'll design a template that takes a context and a question as inputs.
"""
logger.info("## 2. Question-Answering Prompts")

qa_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Context: {context}\n\nQuestion: {question}\n\nAnswer:"
)

context = """
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
Constructed from 1887 to 1889 as the entrance arch to the 1889 World's Fair, it was initially criticized by some of France's leading artists and intellectuals for its design, but it has become a global cultural icon of France and one of the most recognizable structures in the world.
The Eiffel Tower is the most-visited paid monument in the world; 6.91 million people ascended it in 2015.
The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris.
"""

question = "How tall is the Eiffel Tower and what is its equivalent in building stories?"

qa_chain = qa_template | llm
answer = qa_chain.invoke({"context": context, "question": question}).content

logger.debug("Answer:")
logger.debug(answer)

"""
## 3. Code Generation Prompts

Now, let's create a prompt for code generation. We'll design a template that takes a programming language and a task description as inputs.
"""
logger.info("## 3. Code Generation Prompts")

code_gen_template = PromptTemplate(
    input_variables=["language", "task"],
    template="Generate {language} code for the following task:\n\n{task}\n\nCode:"
)

language = "Python"
task = "Create a function that takes a list of numbers and returns the average of the even numbers in the list."

code_gen_chain = code_gen_template | llm
generated_code = code_gen_chain.invoke({"language": language, "task": task}).content

logger.debug("Generated Code:")
logger.debug(generated_code)

"""
## 4. Creative Writing Prompts

Finally, let's create a prompt for creative writing tasks. We'll design a template that takes a genre, a setting, and a theme as inputs.
"""
logger.info("## 4. Creative Writing Prompts")

creative_writing_template = PromptTemplate(
    input_variables=["genre", "setting", "theme"],
    template="Write a short {genre} story set in {setting} that explores the theme of {theme}. The story should be approximately 150 words long.\n\nStory:"
)

genre = "science fiction"
setting = "a space station orbiting a distant planet"
theme = "the nature of humanity"

creative_writing_chain = creative_writing_template | llm
story = creative_writing_chain.invoke({"genre": genre, "setting": setting, "theme": theme}).content

logger.debug("Generated Story:")
logger.debug(story)

logger.info("\n\n[DONE]", bright=True)