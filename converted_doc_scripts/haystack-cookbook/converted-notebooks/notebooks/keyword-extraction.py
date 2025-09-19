from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OllamaFunctionCallingAdapterChatGenerator
from haystack.dataclasses import ChatMessage
from jet.logger import CustomLogger
import json
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
# Keyword Extraction with LLM Chat Generator
This notebook demonstrates how to extract keywords and key phrases from text using Haystack’s `ChatPromptBuilder` together with an LLM via `OllamaFunctionCallingAdapterChatGenerator`. We will:

- Define a prompt that instructs the model to identify single- and multi-word keywords.

- Capture each keyword’s character offsets.

- Assign a relevance score (0–1).

- Parse and display the results as JSON.

### Install packages and setup OllamaFunctionCalling API key
"""
logger.info("# Keyword Extraction with LLM Chat Generator")

# !pip install haystack-ai

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass("Enter OllamaFunctionCalling API key:")

"""
### Import Required Libraries
"""
logger.info("### Import Required Libraries")




"""
### Prepare Text 
Collect your text you want to analyze.
"""
logger.info("### Prepare Text")

text_to_analyze = "Artificial intelligence models like large language models are increasingly integrated into various sectors including healthcare, finance, education, and customer service. They can process natural language, generate text, translate languages, and extract meaningful insights from unstructured data. When performing key word extraction, these systems identify the most significant terms, phrases, or concepts that represent the core meaning of a document. Effective extraction must balance between technical terminology, domain-specific jargon, named entities, action verbs, and contextual relevance. The process typically involves tokenization, stopword removal, part-of-speech tagging, frequency analysis, and semantic relationship mapping to prioritize terms that most accurately capture the document's essential information and main topics."

"""
### Build the Prompt
We construct a single-message template that instructs the model to extract keywords, their positions and scores and return the output as JSON object.
"""
logger.info("### Build the Prompt")

messages = [
    ChatMessage.from_user(
        '''
You are a keyword extractor. Extract the most relevant keywords and phrases from the following text. For each keyword:
1. Find single and multi-word keywords that capture important concepts
2. Include the starting position (index) where each keyword appears in the text
3. Assign a relevance score between 0 and 1 for each keyword
4. Focus on nouns, noun phrases, and important terms

Text to analyze: {{text}}

Return the results as a JSON array in this exact format:
{
  "keywords": [
    {
      "keyword": "example term",
      "positions": [5],
      "score": 0.95
    },
    {
      "keyword": "another keyword",
      "positions": [20],
      "score": 0.85
    }
  ]
}

Important:
- Each keyword must have its EXACT character position in the text (counting from 0)
- Scores should reflect the relevance (0–1)
- Include both single words and meaningful phrases
- List results from highest to lowest score
'''
    )
]

builder = ChatPromptBuilder(template=messages, required_variables='*')
prompt = builder.run(text=text_to_analyze)

"""
### Initialize the Generator and Extract Keywords
We use OllamaFunctionCallingAdapterChatGenerator (e.g., llama3.2) to send our prompt and request a JSON-formatted response.
"""
logger.info("### Initialize the Generator and Extract Keywords")

extractor = OllamaFunctionCallingAdapterChatGenerator(model="llama3.2")

results = extractor.run(
    messages=prompt["prompt"],
    generation_kwargs={"response_format": {"type": "json_object"}}
)

output_str = results["replies"][0].text

"""
### Parse and Display Results
Finally, convert the returned JSON string into a Python object and iterate over the extracted keywords.
"""
logger.info("### Parse and Display Results")

try:
    data = json.loads(output_str)
    for kw in data["keywords"]:
        logger.debug(f'Keyword: {kw["keyword"]}')
        logger.debug(f' Positions: {kw["positions"]}')
        logger.debug(f' Score: {kw["score"]}\n')
except json.JSONDecodeError:
    logger.debug("Failed to parse the output as JSON. Raw output:", output_str)

logger.info("\n\n[DONE]", bright=True)