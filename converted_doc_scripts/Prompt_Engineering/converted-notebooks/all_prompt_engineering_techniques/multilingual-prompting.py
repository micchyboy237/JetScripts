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
# Multilingual and Cross-lingual Prompting

## Overview

This tutorial explores the concepts and techniques of multilingual and cross-lingual prompting in the context of large language models. We'll focus on designing prompts that work effectively across multiple languages and implement techniques for language translation tasks.

## Motivation

As AI language models become increasingly sophisticated, there's a growing need to leverage their capabilities across linguistic boundaries. Multilingual and cross-lingual prompting techniques allow us to create more inclusive and globally accessible AI applications, breaking down language barriers and enabling seamless communication across diverse linguistic landscapes.

## Key Components

1. Multilingual Prompt Design: Strategies for creating prompts that work effectively in multiple languages.
2. Language Detection and Adaptation: Techniques for identifying the input language and adapting the model's response accordingly.
3. Cross-lingual Translation: Methods for using language models to perform translation tasks between different languages.
4. Prompt Templating for Multilingual Support: Using LangChain's PromptTemplate for creating flexible, language-aware prompts.
5. Handling Non-Latin Scripts: Considerations and techniques for working with languages that use non-Latin alphabets.

## Method Details

We'll use Ollama's GPT-4 model via the LangChain library to demonstrate multilingual and cross-lingual prompting techniques. Our approach includes:

1. Setting up the environment with necessary libraries and API keys.
2. Creating multilingual prompts using LangChain's PromptTemplate.
3. Implementing language detection and response adaptation.
4. Designing prompts for cross-lingual translation tasks.
5. Handling various writing systems and scripts.
6. Exploring techniques for improving translation quality and cultural sensitivity.

Throughout the tutorial, we'll provide examples in multiple languages to illustrate the concepts and techniques discussed.

## Conclusion

By the end of this tutorial, you will have gained practical skills in designing and implementing multilingual and cross-lingual prompts. These techniques will enable you to create more inclusive and globally accessible AI applications, leveraging the power of large language models across diverse linguistic contexts. The knowledge gained here forms a foundation for developing sophisticated, language-aware AI systems capable of breaking down communication barriers on a global scale.

## Setup

First, let's import the necessary libraries and set up our environment.
"""
logger.info("# Multilingual and Cross-lingual Prompting")


load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

llm = ChatOllama(model="llama3.2")


def print_response(response):
    logger.debug(response.content)


"""
## Multilingual Prompt Design

Let's start by creating a multilingual greeting prompt that adapts to different languages.
"""
logger.info("## Multilingual Prompt Design")

multilingual_greeting = PromptTemplate(
    input_variables=["language"],
    template="Greet the user in {language} and provide a short introduction about the weather in a country where this language is spoken."
)

languages = ["English", "Spanish", "French", "German", "Japanese"]

for lang in languages:
    prompt = multilingual_greeting.format(language=lang)
    response = llm.invoke(prompt)
    logger.debug(f"{lang}:")
    print_response(response)
    logger.debug()

"""
## Language Detection and Adaptation

Now, let's create a prompt that can detect the input language and respond accordingly.
"""
logger.info("## Language Detection and Adaptation")

language_adaptive_prompt = PromptTemplate(
    input_variables=["user_input"],
    template="""Detect the language of the following input and respond in the same language:
    User input: {user_input}
    Your response (in the detected language):"""
)

inputs = [
    "Hello, how are you?",
    "Hola, ¿cómo estás?",
    "Bonjour, comment allez-vous ?",
    "こんにちは、お元気ですか？",
    "Здравствуйте, как дела?"
]

for user_input in inputs:
    prompt = language_adaptive_prompt.format(user_input=user_input)
    response = llm.invoke(prompt)
    logger.debug(f"Input: {user_input}")
    logger.debug("Response:")
    print_response(response)
    logger.debug()

"""
## Cross-lingual Translation

Let's implement a prompt for cross-lingual translation tasks.
"""
logger.info("## Cross-lingual Translation")

translation_prompt = PromptTemplate(
    input_variables=["source_lang", "target_lang", "text"],
    template="""Translate the following text from {source_lang} to {target_lang}:
    {source_lang} text: {text}
    {target_lang} translation:"""
)

translations = [
    {"source_lang": "English", "target_lang": "French",
        "text": "The quick brown fox jumps over the lazy dog."},
    {"source_lang": "Spanish", "target_lang": "German", "text": "La vida es bella."},
    {"source_lang": "Japanese", "target_lang": "English", "text": "桜の花が満開です。"}
]

for t in translations:
    prompt = translation_prompt.format(**t)
    response = llm.invoke(prompt)
    logger.debug(f"From {t['source_lang']} to {t['target_lang']}:")
    logger.debug(f"Original: {t['text']}")
    logger.debug("Translation:")
    print_response(response)
    logger.debug()

"""
## Handling Non-Latin Scripts

Let's create a prompt that can work with non-Latin scripts and provide transliteration.
"""
logger.info("## Handling Non-Latin Scripts")

non_latin_prompt = PromptTemplate(
    input_variables=["text", "script"],
    template="""Provide the following information for the given text:
    1. The original text
    2. The name of the script/writing system
    3. A transliteration to Latin alphabet
    4. An English translation

    Text: {text}
    Script: {script}
    """
)

non_latin_texts = [
    {"text": "こんにちは、世界", "script": "Japanese"},
    {"text": "Здравствуй, мир", "script": "Cyrillic"},
    {"text": "नमस्ते दुनिया", "script": "Devanagari"}
]

for text in non_latin_texts:
    prompt = non_latin_prompt.format(**text)
    response = llm.invoke(prompt)
    print_response(response)
    logger.debug()

"""
## Improving Translation Quality and Cultural Sensitivity

Finally, let's create a prompt that focuses on maintaining cultural context and idioms in translation.
"""
logger.info("## Improving Translation Quality and Cultural Sensitivity")

cultural_translation_prompt = PromptTemplate(
    input_variables=["source_lang", "target_lang", "text"],
    template="""Translate the following text from {source_lang} to {target_lang}, paying special attention to cultural context and idiomatic expressions. Provide:
    1. A direct translation
    2. A culturally adapted translation (if different)
    3. Explanations of any cultural nuances or idioms

    {source_lang} text: {text}
    {target_lang} translation and explanation:"""
)

cultural_texts = [
    {"source_lang": "English", "target_lang": "Japanese",
        "text": "It's raining cats and dogs."},
    {"source_lang": "French", "target_lang": "English",
        "text": "Je suis dans le pétrin."},
    {"source_lang": "Spanish", "target_lang": "German",
        "text": "Cuesta un ojo de la cara."}
]

for text in cultural_texts:
    prompt = cultural_translation_prompt.format(**text)
    response = llm.invoke(prompt)
    logger.debug(f"From {text['source_lang']} to {text['target_lang']}:")
    logger.debug(f"Original: {text['text']}")
    logger.debug("Translation and Explanation:")
    print_response(response)
    logger.debug()

logger.info("\n\n[DONE]", bright=True)
