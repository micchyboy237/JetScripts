from dotenv import load_dotenv
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import CustomLogger
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Prompt Length and Complexity Management

## Overview

This tutorial explores techniques for managing prompt length and complexity when working with large language models (LLMs). We'll focus on two key aspects: balancing detail and conciseness in prompts, and strategies for handling long contexts.

## Motivation

Effective prompt engineering often requires finding the right balance between providing enough context for the model to understand the task and keeping prompts concise for efficiency. Additionally, many real-world applications involve processing long documents or complex multi-step tasks, which can exceed the context window of LLMs. Learning to manage these challenges is crucial for building robust AI applications.

## Key Components

1. Balancing detail and conciseness in prompts
2. Strategies for handling long contexts
3. Practical examples using Ollama's GPT model and LangChain

## Method Details

We'll start by examining techniques for crafting prompts that provide sufficient context without unnecessary verbosity. This includes using clear, concise language and leveraging prompt templates for consistency.

Next, we'll explore strategies for handling long contexts, such as:
- Chunking: Breaking long texts into smaller, manageable pieces
- Summarization: Condensing long texts while retaining key information
- Iterative processing: Handling complex tasks through multiple API calls

Throughout the tutorial, we'll use practical examples to demonstrate these concepts, utilizing Ollama's GPT model via the LangChain library.

## Conclusion

By the end of this tutorial, you'll have a solid understanding of how to manage prompt length and complexity effectively. These skills will enable you to create more efficient and robust AI applications, capable of handling a wide range of text processing tasks.

## Setup

First, let's import the necessary libraries and set up our environment.
"""
logger.info("# Prompt Length and Complexity Management")


load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

llm = ChatOllama(model="llama3.2")

logger.debug("Setup complete!")

"""
## Balancing Detail and Conciseness

Let's start by examining how to balance detail and conciseness in prompts. We'll compare responses from a detailed prompt and a concise prompt.
"""
logger.info("## Balancing Detail and Conciseness")

detailed_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""Please provide a comprehensive explanation of {topic}. Include its definition,
    historical context, key components, practical applications, and any relevant examples.
    Also, discuss any controversies or debates surrounding the topic, and mention potential
    future developments or trends."""
)

concise_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Briefly explain {topic} and its main importance."
)

topic = "artificial intelligence"

logger.debug("Detailed response:")
logger.debug(llm.invoke(detailed_prompt.format(topic=topic)).content)

logger.debug("\nConcise response:")
logger.debug(llm.invoke(concise_prompt.format(topic=topic)).content)

"""
### Analysis of Prompt Balance

Let's analyze the differences between the detailed and concise prompts, and discuss strategies for finding the right balance.
"""
logger.info("### Analysis of Prompt Balance")

analysis_prompt = PromptTemplate(
    input_variables=["topic", "detailed_response", "concise_response"],
    template="""Compare the following two responses on {topic}:

Detailed response:
{detailed_response}

Concise response:
{concise_response}

Analyze the differences in terms of:
1. Information coverage
2. Clarity and focus
3. Potential use cases for each type of response

Then, suggest strategies for balancing detail and conciseness in prompts."""
)

detailed_response = llm.invoke(detailed_prompt.format(topic=topic)).content
concise_response = llm.invoke(concise_prompt.format(topic=topic)).content

analysis = llm.invoke(analysis_prompt.format(
    topic=topic,
    detailed_response=detailed_response,
    concise_response=concise_response
)).content

logger.debug(analysis)

"""
## Strategies for Handling Long Contexts

Now, let's explore strategies for handling long contexts, which often exceed the token limits of language models.

### 1. Chunking

Chunking involves breaking long texts into smaller, manageable pieces. Let's demonstrate this using a long text passage.
"""
logger.info("## Strategies for Handling Long Contexts")

long_text = """
Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines that can simulate human cognitive processes.
The field of AI has a rich history dating back to the 1950s, with key milestones such as the development of the first neural networks and expert systems.
AI encompasses a wide range of subfields, including machine learning, natural language processing, computer vision, and robotics.
Practical applications of AI include speech recognition, image classification, autonomous vehicles, and medical diagnosis.
AI has the potential to revolutionize many industries, from healthcare and finance to transportation and entertainment.
However, there are ongoing debates and controversies surrounding AI, such as concerns about job displacement, bias in algorithms, and the ethical implications of autonomous systems.
Looking ahead, the future of AI holds promise for advancements in areas like explainable AI, AI ethics, and human-AI collaboration.
The intersection of AI with other technologies like blockchain, quantum computing, and biotechnology will likely shape the future of the field.
But as AI continues to evolve, it is essential to consider the societal impact and ethical implications of these technologies.
One of the key challenges for AI researchers and developers is to strike a balance between innovation and responsibility, ensuring that AI benefits society as
a whole while minimizing potential risks.
If managed effectively, AI has the potential to transform our world in ways we can only begin to imagine.
Though the future of AI is uncertain, one thing is clear: the impact of artificial intelligence will be profound and far-reaching.
"""

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

chunks = text_splitter.split_text(long_text)

logger.debug(f"Number of chunks: {len(chunks)}")
logger.debug(f"First chunk: {chunks[0][:200]}...")

"""
### 2. Summarization

Summarization can be used to condense long texts while retaining key information. Let's use LangChain's summarization chain to demonstrate this.
"""
logger.info("### 2. Summarization")


doc_chunks = [Document(page_content=chunk) for chunk in chunks]

chain = load_summarize_chain(llm, chain_type="map_reduce")

summary_result = chain.invoke(doc_chunks)

logger.debug("Summary:")
logger.debug(summary_result['output_text'])

"""
### 3. Iterative Processing

For complex tasks that require multiple steps, we can use iterative processing. Let's demonstrate this with a multi-step analysis task.
"""
logger.info("### 3. Iterative Processing")


def iterative_analysis(text, steps):
    """
    Perform iterative analysis on a given text.

    Args:
    text (str): The text to analyze.
    steps (list): List of analysis steps to perform.

    Returns:
    str: The final analysis result.
    """
    result = text
    for step in steps:
        prompt = PromptTemplate(
            input_variables=["text"],
            template=f"Analyze the following text. {step}\n\nText: {{text}}\n\nAnalysis:"
        )
        result = llm.invoke(prompt.format(text=result)).content
    return result


analysis_steps = [
    "Identify the main topics discussed.",
    "Summarize the key points for each topic.",
    "Provide a brief conclusion based on the analysis."
]

final_analysis = iterative_analysis(long_text, analysis_steps)
logger.debug("Final Analysis:")
logger.debug(final_analysis)

"""
## Practical Tips for Managing Prompt Length and Complexity

Let's conclude with some practical tips for managing prompt length and complexity in real-world applications.
"""
logger.info("## Practical Tips for Managing Prompt Length and Complexity")

tips_prompt = """
Based on the examples and strategies we've explored for managing prompt length and complexity,
provide a list of 5 practical tips for developers working with large language models.
Each tip should be concise and actionable.
"""

tips = llm.invoke(tips_prompt).content
logger.debug(tips)

logger.info("\n\n[DONE]", bright=True)
