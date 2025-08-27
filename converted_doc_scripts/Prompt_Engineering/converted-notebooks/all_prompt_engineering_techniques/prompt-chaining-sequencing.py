from dotenv import load_dotenv
from jet.llm.ollama.base_langchain import ChatOllama
from jet.logger import CustomLogger
from langchain.prompts import PromptTemplate
import os
import re
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Prompt Chaining and Sequencing Tutorial

## Overview

This tutorial explores the concepts of prompt chaining and sequencing in the context of working with large language models. We'll use Ollama's GPT models and the LangChain library to demonstrate how to connect multiple prompts and build logical flows for more complex AI-driven tasks.

## Motivation

As AI applications become more sophisticated, there's often a need to break down complex tasks into smaller, manageable steps. Prompt chaining and sequencing allow us to guide language models through a series of interrelated prompts, enabling more structured and controlled outputs. This approach is particularly useful for tasks that require multiple stages of processing or decision-making.

## Key Components

1. **Basic Prompt Chaining**: Connecting the output of one prompt to the input of another.
2. **Sequential Prompting**: Creating a logical flow of prompts to guide the AI through a multi-step process.
3. **Dynamic Prompt Generation**: Using the output of one prompt to dynamically generate the next prompt.
4. **Error Handling and Validation**: Implementing checks and balances within the prompt chain.

## Method Details

We'll start by setting up our environment with the necessary libraries. Then, we'll explore basic prompt chaining by connecting two simple prompts. We'll move on to more complex sequential prompting, where we'll guide the AI through a multi-step analysis process. Next, we'll demonstrate how to dynamically generate prompts based on previous outputs. Finally, we'll implement error handling and validation techniques to make our prompt chains more robust.

Throughout the tutorial, we'll use practical examples to illustrate these concepts, such as a multi-step text analysis task and a dynamic question-answering system.

## Conclusion

By the end of this tutorial, you'll have a solid understanding of how to implement prompt chaining and sequencing in your AI applications. These techniques will enable you to tackle more complex tasks, improve the coherence and relevance of AI-generated content, and create more interactive and dynamic AI-driven experiences.

## Setup

Let's start by importing the necessary libraries and setting up our environment.
"""
logger.info("# Prompt Chaining and Sequencing Tutorial")



load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

llm = ChatOllama(model="llama3.2")

"""
## Basic Prompt Chaining

Let's start with a simple example of prompt chaining. We'll create two prompts: one to generate a short story, and another to summarize it.
"""
logger.info("## Basic Prompt Chaining")

story_prompt = PromptTemplate(
    input_variables=["genre"],
    template="Write a short {genre} story in 3-4 sentences."
)

summary_prompt = PromptTemplate(
    input_variables=["story"],
    template="Summarize the following story in one sentence:\n{story}"
)

def story_chain(genre):
    """Generate a story and its summary based on a given genre.

    Args:
        genre (str): The genre of the story to generate.

    Returns:
        tuple: A tuple containing the generated story and its summary.
    """
    story = (story_prompt | llm).invoke({"genre": genre}).content
    summary = (summary_prompt | llm).invoke({"story": story}).content
    return story, summary

genre = "science fiction"
story, summary = story_chain(genre)
logger.debug(f"Story: {story}\n\nSummary: {summary}")

"""
## Sequential Prompting

Now, let's create a more complex sequence of prompts for a multi-step analysis task. We'll analyze a given text for its main theme, tone, and key takeaways.
"""
logger.info("## Sequential Prompting")

theme_prompt = PromptTemplate(
    input_variables=["text"],
    template="Identify the main theme of the following text:\n{text}"
)

tone_prompt = PromptTemplate(
    input_variables=["text"],
    template="Describe the overall tone of the following text:\n{text}"
)

takeaway_prompt = PromptTemplate(
    input_variables=["text", "theme", "tone"],
    template="Given the following text with the main theme '{theme}' and tone '{tone}', what are the key takeaways?\n{text}"
)

def analyze_text(text):
    """Perform a multi-step analysis of a given text.

    Args:
        text (str): The text to analyze.

    Returns:
        dict: A dictionary containing the theme, tone, and key takeaways of the text.
    """
    theme = (theme_prompt | llm).invoke({"text": text}).content
    tone = (tone_prompt | llm).invoke({"text": text}).content
    takeaways = (takeaway_prompt | llm).invoke({"text": text, "theme": theme, "tone": tone}).content
    return {"theme": theme, "tone": tone, "takeaways": takeaways}

sample_text = "The rapid advancement of artificial intelligence has sparked both excitement and concern among experts. While AI promises to revolutionize industries and improve our daily lives, it also raises ethical questions about privacy, job displacement, and the potential for misuse. As we stand on the brink of this technological revolution, it's crucial that we approach AI development with caution and foresight, ensuring that its benefits are maximized while its risks are minimized."

analysis = analyze_text(sample_text)
for key, value in analysis.items():
    logger.debug(f"{key.capitalize()}: {value}\n")

"""
## Dynamic Prompt Generation

In this section, we'll create a dynamic question-answering system that generates follow-up questions based on previous answers.
"""
logger.info("## Dynamic Prompt Generation")

answer_prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question concisely:\n{question}"
)

follow_up_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="Based on the question '{question}' and the answer '{answer}', generate a relevant follow-up question."
)

def dynamic_qa(initial_question, num_follow_ups=3):
    """Conduct a dynamic Q&A session with follow-up questions.

    Args:
        initial_question (str): The initial question to start the Q&A session.
        num_follow_ups (int): The number of follow-up questions to generate.

    Returns:
        list: A list of dictionaries containing questions and answers.
    """
    qa_chain = []
    current_question = initial_question

    for _ in range(num_follow_ups + 1):  # +1 for the initial question
        answer = (answer_prompt | llm).invoke({"question": current_question}).content
        qa_chain.append({"question": current_question, "answer": answer})

        if _ < num_follow_ups:  # Generate follow-up for all but the last iteration
            current_question = (follow_up_prompt | llm).invoke({"question": current_question, "answer": answer}).content

    return qa_chain

initial_question = "What are the potential applications of quantum computing?"
qa_session = dynamic_qa(initial_question)

for i, qa in enumerate(qa_session):
    logger.debug(f"Q{i+1}: {qa['question']}")
    logger.debug(f"A{i+1}: {qa['answer']}\n")

"""
## Error Handling and Validation

In this final section, we'll implement error handling and validation in our prompt chains to make them more robust.
"""
logger.info("## Error Handling and Validation")

generate_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Generate a 4-digit number related to the topic: {topic}. Respond with ONLY the number, no additional text."
)

validate_prompt = PromptTemplate(
    input_variables=["number", "topic"],
    template="Is the number {number} truly related to the topic '{topic}'? Answer with 'Yes' or 'No' and explain why."
)

def extract_number(text):
    """Extract a 4-digit number from the given text.

    Args:
        text (str): The text to extract the number from.

    Returns:
        str or None: The extracted 4-digit number, or None if no valid number is found.
    """
    match = re.search(r'\b\d{4}\b', text)
    return match.group() if match else None

def robust_number_generation(topic, max_attempts=3):
    """Generate a topic-related number with validation and error handling.

    Args:
        topic (str): The topic to generate a number for.
        max_attempts (int): Maximum number of generation attempts.

    Returns:
        str: A validated 4-digit number or an error message.
    """
    for attempt in range(max_attempts):
        try:
            response = (generate_prompt | llm).invoke({"topic": topic}).content
            number = extract_number(response)

            if not number:
                raise ValueError(f"Failed to extract a 4-digit number from the response: {response}")

            validation = (validate_prompt | llm).invoke({"number": number, "topic": topic}).content
            if validation.lower().startswith("yes"):
                return number
            else:
                logger.debug(f"Attempt {attempt + 1}: Number {number} was not validated. Reason: {validation}")
        except Exception as e:
            logger.debug(f"Attempt {attempt + 1} failed: {str(e)}")

    return "Failed to generate a valid number after multiple attempts."

topic = "World War II"
result = robust_number_generation(topic)
logger.debug(f"Final result for topic '{topic}': {result}")

logger.info("\n\n[DONE]", bright=True)