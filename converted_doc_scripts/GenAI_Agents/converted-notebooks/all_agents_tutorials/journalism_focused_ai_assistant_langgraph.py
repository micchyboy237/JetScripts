from IPython.display import display, Image
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from functools import lru_cache
from jet.llm.mlx.adapters.mlx_langchain_llm_adapter import ChatMLX
from jet.llm.ollama.base_langchain import ChatOllama
from jet.logger import CustomLogger
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langgraph.graph import StateGraph, END
from pathlib import Path
from typing import Optional, List, TypedDict
import os
import pprint
import shutil
import time


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Journalism-Focused AI Assistant

![Journalism Focused AI Assistant](../images/journalism_focused_ai_assistant_langgraph.png)

## Overview

This notebook introduces an AI-powered assistant designed specifically for journalists, tackling challenges like misinformation, biased reporting, and information overload. With tools for fact-checking, tone analysis, summarization, and more, it uses AI tools to enhance the accuracy and efficiency of journalistic work.


## Motivation

Journalism plays a vital role in upholding democracy, but modern challenges like the flood of online misinformation and subtle biases can undermine trust in reporting. Journalists often face the daunting task of making sense of huge volumes of data under tight deadlines. This notebook equips journalists with tools to:

- **Verify claims** through reliable fact-checking.
- **Detect tone and biases** to maintain balanced storytelling.
- **Simplify the review process** with concise and accurate summaries, as well as grammar checks.

The ultimate goal is to support ethical reporting and uphold the integrity of the information we rely on every day.

## Key Components

1. **Language Models**: Get insights and generate responses using advanced models like `Llama 3.1/3.2` and `llama-3.2-3b-instruct`.
2. **Web Search Integration**: Fetch reliable data from `DuckDuckGo’s` search API to strengthen the research process.
3. **Document Parsing**: Extract text from PDFs and web pages with tools like PyMuPDFLoader and WebBaseLoader, enhanced by BeautifulSoupTransformer.
4. **Structured Outputs**: Receive responses in a clean, JSON format for consistency and precision.
5. **Text Splitting and Summarization**: Break down long articles into digestible summaries using RecursiveCharacterTextSplitter.
6. **Tailored Prompts and Examples**: Use custom prompts and few-shot prompting to guide the AI in providing meaningful results.
7. **LangGraph Workflow**: Tie everything together into a seamless, easy-to-use workflow.

## Method Details

### Setting Up the Environment
- Import necessary libraries.
- Configure any API keys and data sources.

### Summarization
- Pinpoint key ideas in long articles or reports.
- Generate clear, concise summaries for quicker understanding.

### Fact-Checking
- Input claims or statements to analyze.
- Search credible sources and compile relevant evidence.
- Categorize claims (e.g., confirmed, refuted, or unverifiable) and provide detailed explanations.

### Tone and Bias Analysis
- Process text to determine sentiment—positive, neutral, or negative.
- Spot and highlight biased language or phrasing.

### Quote Extraction
- Detect direct quotes and their sources to add transparency to your reporting.

### Grammar and Bias Review
- Identify grammar errors and subtle biases, ensuring content is polished and fair.

### LangGraph Workflow Integration
- Use LangGraph to connect all these tools into one powerful workflow:
- Define nodes for each task, from analysis to report generation.
- Pass data seamlessly between tasks for smooth processing.
- Test the workflow on a sample article.

### Report Generation
- Combine all findings into a well-organized report that’s easy to read and share.

### Additional Considerations
- Discuss limitations, potential improvements, or specific use cases.

## This Journalism-Focused AI Assistant is all about helping journalists do their best work by:
- **Improving Accuracy**: Fact-checking tools ensure your claims are backed by evidence.
- **Boosting Efficiency**: Summarization and workflows save valuable time.
- **Adding Transparency**: Features like quote extraction and structured reports build trust.
- **Promoting Ethical Reporting**: Tone and bias analysis helps maintain objectivity.

By bringing these features together, this tool empowers journalists to focus on what they do best: telling meaningful stories.

## Setup and Imports

First, we'll import the necessary modules and set up our environment.
"""
logger.info("# Journalism-Focused AI Assistant")


load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

data_path = Path(os.getcwd()).parent / "data"

ddgs = DDGS()

"""
## Initialize Language Models

We will initialize the language models that can be used for testing.

For running Llama models, we use Ollama. A detailed tutorial on this is beyond the scope of this notebook, but you can refer to their repository for a [quickstart guide on Ollama](https://github.com/ollama/ollama).
"""
logger.info("## Initialize Language Models")

llm = ChatMLX(model="llama-3.2-3b-instruct", temperature=0)

"""
## Data Setup

For this, I used the `gpt-4o` model to generate a sample article using the prompt below. Note that the information mentioned in the article is not to be taken seriously. The article will serve as input for summarization, fact-checking, tone analysis, quote extraction, and grammar and bias analysis modules, providing a basis for refining prompt responses.

We will take advantage of the `document_loaders` `langchain` module, specifically the `PyMuPDFLoader` for loading the text from a PDF file. 

`Prompt`:
Write an article designed for classification purposes, containing a variety of claims that fit into distinct but subtly presented categories: well-known and confirmed facts, refuted claims, unverifiable statements requiring further research, and vague or speculative assertions. The article should flow naturally without explicitly labeling these categories but ensure that each type of claim is clearly identifiable through its content and context. Use varied tones, including positive, critical, biased, or opinionated language, to differentiate the claims. Incorporate quotes to enhance realism and include occasional minor grammar errors or awkward phrasing for added authenticity.
"""
logger.info("## Data Setup")

file_path = data_path / "Sample AI Generated Article.pdf"

pages = []
loader = PyMuPDFLoader(file_path)
for page in loader.lazy_load():
    pages.append(page)

"""
#### Check the file contents and remove the new lines and tabs

For printing throughout the tutorial, I will sometimes use the built-in `pprint` package as it formats text and different data types in more human-readable forms.
"""
logger.info("#### Check the file contents and remove the new lines and tabs")


def clean_page_content(page_content: str) -> str:
    """
    Clean the page content by removing new lines and tabs
    """
    page_content = page_content.replace("\n", " ")
    page_content = page_content.replace("\t", " ")
    return page_content


logger.debug("Page content before cleaning")
for page in pages:
    pprint.plogger.debug(page.page_content[:100])


logger.debug("\nPage content after cleaning")
formatted_pages = []
for page in pages:
    page_content = clean_page_content(page.page_content)
    formatted_pages.append(page_content)
    pprint.plogger.debug(page_content[:200])

full_article_text = " ".join(formatted_pages)

"""
## Helper Functions
"""
logger.info("## Helper Functions")


def chunk_large_text(text, chunk_size=100000, overlap=1000):
    """
    Splits the input text into manageable chunks while maintaining context overlap.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)


"""
## Summarizing Long Articles

Summarizing lengthy articles requires handling the constraints of token limits in models like `llama-3.2-3b-instruct` or `Llama 3.2/3.1`. These models can process only a limited amount of text at once, making **text splitting** a crucial step to ensure the analysis is thorough and no critical information is overlooked. While this notebook may not reach those token limits, the text-splitting approach is demonstrated as a proof of concept.

### Why Use Text Splitting?

Text splitting ensures the model can process lengthy content effectively by:
- Dividing large articles into manageable chunks (e.g., 100,000 tokens).
- Maintaining context through overlapping sections (e.g., 1,000 tokens).
- Preventing request rejections due to exceeding token limits.

### Tools and Techniques

- **Splitter**: The `RecursiveCharacterTextSplitter` from `langchain` is used for breaking down text into smaller chunks while preserving context and readability.
- **Custom Prompts**: Focus prompts on extracting key events, individuals, and statistics for precise summarization.
- **Step-by-Step Workflow**:
  1. Split the text into chunks.
  2. Summarize each chunk individually using the AI model.
  3. Combine the individual summaries into a cohesive and concise final summary.

### Benefits of This Approach

By leveraging text splitting and summarization workflows, even the longest articles can be processed effectively:
- **Accuracy**: Ensures no critical details are missed during summarization.
- **Efficiency**: Breaks down complex tasks into manageable pieces for faster results.
- **Consistency**: Maintains flow and context across the final summary.

This approach provides reliable and concise summaries for articles of any length, making it a powerful tool for handling extensive content with ease.

### Prepare the summarization pipelines
"""
logger.info("## Summarizing Long Articles")

summarization_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "Summarize the provided article text focusing on the main events, key people involved, "
        "and any important statistics in 150-200 words. Use a neutral tone suitable for a journalistic report:\n\n"
        "Article text:\n{text}\n\n"
    )
)

combine_summarization_prompt = PromptTemplate(
    input_variables=["summaries"],
    template=(
        "Combine the provided summaries into a single coherent summary that captures the main events, key people involved, "
        "and important statistics in 150-200 words. Use a neutral tone suitable for a journalistic report:\n\n"
        "Summaries:\n{summaries}\n\n"
    )
)

summarization_pipeline = summarization_prompt | llm

combine_summarization_pipeline = combine_summarization_prompt | llm

"""
### Summarization helper functions
"""
logger.info("### Summarization helper functions")


def combine_summaries(summaries: List[str]):
    """
    Combines multiple summaries into a single coherent summary.
    """
    if len(summaries) == 1:
        return summaries[0]

    summaries_text = ""
    for i, summary in enumerate(summaries):
        summaries_text += f"Summary {i + 1}:\n{summary}\n\n"

    full_summary = combine_summarization_pipeline.invoke(
        {"summaries": summaries_text})

    return full_summary


def summarize_article(article_text: str, article_chunks=None):
    """
    Summarize a full article text by splitting it into manageable chunks and generating summaries for each chunk.
    The individual summaries are then combined into a single coherent summary.
    """
    if not article_chunks:
        article_chunks = chunk_large_text(article_text)

    summaries = []
    for chunk in article_chunks:
        summary = summarization_pipeline.invoke({"text": chunk})
        summaries.append(summary.content)

    full_summary = combine_summaries(summaries)

    return full_summary


"""
### Sample usage
"""
logger.info("### Sample usage")

summary_result = summarize_article(full_article_text)

logger.debug("\n\nSummary of the article")
pprint.plogger.debug(summary_result)

"""
## Fact-Checking Articles

This section outlines a structured approach to verify claims in articles, ensuring accuracy and credibility.

### Key Components

- **Structured Outputs**: Results are formatted as JSON with:
  - **Statement**: The claim being analyzed.
  - **Status**: `confirmed`, `refuted`, `unverifiable`, or `vague`.
  - **Explanation**: A rationale for the status.
  - **Keywords**: Suggested for further investigation.
  - Implemented using the `with_structured_output` method for consistency.

- **Search Integration**:
  - DuckDuckGo’s API retrieves relevant search results.
  - Tools like `WebBaseLoader` and `BeautifulSoupTransformer` extract and clean web content

- **Tailored Prompting**: A custom prompt guides the AI to analyze claims, flag inaccuracies, and suggest further research paths.

### Workflow

1. Extract claims from the text.
2. Fetch evidence using web search.
3. Analyze and categorize claims with supporting explanations.
4. Output findings in a structured, clear format.

### Benefits

This approach combines AI precision with real-time data to deliver:
- **Transparency**: Claims are backed by evidence.
- **Efficiency**: Automated workflows save time.
- **Consistency**: Standardized outputs ensure reliability.

A seamless tool to enhance content credibility and journalistic integrity.

### The models used for the structured output
"""
logger.info("## Fact-Checking Articles")


class FactCheckStatement(TypedDict):
    """
    Represents a single fact-check statement structure.
    """
    statement: str
    status: str
    explanation: str
    suggested_keywords: List[str]


class FactCheckResult(TypedDict):
    """
    Represents the result of a fact-checking process.
    """
    result: List[FactCheckStatement]


"""
#### Define the internet search helper functions
"""
logger.info("#### Define the internet search helper functions")


@lru_cache
def search_ddg(keywords: str, max_results: int = 1):
    """
        This function searches DuckDuckGo for the given keywords and returns the top max_results results.
    """
    try:
        text_results = ddgs.text(keywords=keywords, max_results=max_results)
    except Exception as e:
        logger.debug("Keywords", keywords)
        logger.debug("Error: ", str(e))
        time.sleep(5)
        try:
            text_results = ddgs.text(
                keywords=keywords, max_results=max_results)
        except Exception as e:
            logger.debug("Error: ", str(e))
            return [{}]

    return text_results


def search_and_summarize(keywords: str, max_results: int = 1):
    """
        Search for the given keywords using DuckDuckGo and summarize the content of the top search results.
    """
    text_results = search_ddg(keywords, max_results)

    results = []
    for result in text_results:
        loader = WebBaseLoader([str(result['href'])])
        html_content = str(loader.scrape())
        bs_transformer = BeautifulSoupTransformer()
        html_transform = (
            bs_transformer.remove_unwanted_tags(
                html_content, ["script", "style", "noscript"])
        )

        html_transform = bs_transformer.extract_tags(
            html_transform, ["p"], remove_comments=True)
        html_transform = bs_transformer.remove_unnecessary_lines(
            html_transform)

        summary_result = summarize_article(page_content)

        results.append({
            "title": result['title'],
            "url":  result['href'],
            "summary": summary_result
        })

    return results


"""
### Prepare the fact-checking pipeline
"""
logger.info("### Prepare the fact-checking pipeline")

fact_checking_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "Fact-check the texts provided. For each statement, identify any factual inaccuracies, misleading information, "
        "unsupported claims, or vague language lacking specific details. Confirm accuracy for each claim where possible, "
        "or provide suggestions for further searches. Flag statements as 'vague' if they are overly broad or lacking "
        "critical specifics (e.g., missing names, dates, or descriptions of technologies)."
        "Suggest keyword if you can't confirm or refute the statement.\n\n"
        "{text}\n\n"
        "Return the results in this JSON format:\n"
        "{{\n"
        "  \"results\": [\n"
        "    {{\n"
        "      \"statement\": \"<Original statement>\",\n"
        "      \"status\": \"<confirmed | refuted | unverifiable | vague>\",\n"
        "      \"explanation\": \"<Brief explanation of findings or reason for vagueness>\",\n"
        "      \"suggested_keywords\": [\"<keyword1>\", \"<keyword2>\"]\n"
        "    }},\n"
        "    {{...}}\n"
        "  ]\n"
        "}}\n"
    )
)

structured_output_llm = llm.with_structured_output(FactCheckResult)

fact_checking_pipeline = fact_checking_prompt | structured_output_llm


def fact_check_article(article_text: str, chunks=None):
    """
    Fact-check the given text by identifying factual inaccuracies, misleading information, unsupported claims, or vague language.
    """
    if not chunks:
        chunks = chunk_large_text(article_text)

    fact_check_results = []
    for chunk in chunks:
        fact_check_result = fact_checking_pipeline.invoke({"text": chunk})
        for statement in fact_check_result["result"]:
            suggested_keywords = statement.get('suggested_keywords', [])
            if suggested_keywords:
                statement['search_results'] = [
                    search_and_summarize(keyword) for keyword in suggested_keywords
                ]

        fact_check_results.extend(fact_check_result["result"])

    return fact_check_results


"""
### Sample usage
"""
logger.info("### Sample usage")

fact_check_results = fact_check_article(full_article_text)

logger.debug("\n\n Unverifiable facts")
for statement in fact_check_results:
    if statement["status"] == "unverifiable":
        pprint.plogger.debug(statement)

"""
For our case, it seems that multiple sources return similar information. This redundancy can be useful for verifying the consistency and reliability of the data, but it also highlights the importance of cross-referencing to ensure accuracy.

## Tone and Sentiment Analysis

This module evaluates the tone of an article to determine if it’s neutral, positive, critical, or opinionated. By identifying tone, journalists can uncover biases and better understand the mood conveyed in the content, ensuring more balanced reporting.

### Prepare the tone and sentiment analysis pipeline
"""
logger.info("## Tone and Sentiment Analysis")

tone_analysis_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "Analyze the tones of the following article. Does it appear neutral, positive, critical, or opinionated? "
        "Provide a short explanation for each detected tone. "
        "Use specific examples from the article to support your analysis.\\n\n{text}"
    )
)

tone_pipeline = tone_analysis_prompt | llm


def tone_analysis_article(article_text: str, chunks=None):
    """
    Analyze the tones of the given article text.
    """
    if not chunks:
        chunks = chunk_large_text(article_text)

    tone_results = []
    for chunk in chunks:
        tone_result = tone_pipeline.invoke({"text": chunk})
        tone_results.append(tone_result.content)

    return tone_results


"""
### Sample usage
"""
logger.info("### Sample usage")

tone_analysis_results = tone_analysis_article(full_article_text)

logger.debug("\n\n Tone analysis of the article")
pprint.plogger.debug(tone_analysis_results)

"""
## Quote Extraction

This module identifies and extracts key quotes from an article, offering insights into important viewpoints and statements. Extracted quotes help journalists highlight perspectives and enrich their reporting with direct references.

### Prepare the quote extraction pipeline
"""
logger.info("## Quote Extraction")

quote_extraction_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "Identify direct quotes in the following content, noting the speaker's name "
        "and the context of each quote. If there are no quotes, return 'No quotes found'.\n\n"
        "Text: {text}"
    )
)

quote_extraction_pipeline = quote_extraction_prompt | llm


def quote_extraction_article(article_text: str, chunks=None):
    """
    Extract direct quotes from the given article text.
    """
    if not chunks:
        chunks = chunk_large_text(article_text)

    quote_results = []
    for chunk in chunks:
        quote_result = quote_extraction_pipeline.invoke({"text": chunk})
        quote_results.append(quote_result.content)

    return quote_results


"""
### Sample usage
"""
logger.info("### Sample usage")

quote_extraction_results = quote_extraction_article(full_article_text)

pprint.plogger.debug(quote_extraction_results)

"""
## Grammar and Bias Analysis

This module reviews articles for grammatical accuracy and detects potential biases, helping journalists maintain credibility and ensure neutrality in their reporting.

### Prepare the grammar and bias analysis pipeline
"""
logger.info("## Grammar and Bias Analysis")

review_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "Review the following article for grammar, spelling, punctuation, and bias. "
        "Provide feedback on each aspect in form of a list of the issues found and some suggestions for improvement.\n\n"
        "{text}"
    )
)

grammar_and_bias_review = review_prompt | llm


def grammary_and_bias_analysis_article(article_text: str, chunks=None):
    """
    Review the given article text for grammar, spelling, punctuation, and bias.
    """
    if not chunks:
        chunks = chunk_large_text(article_text)

    review_results = []
    for chunk in chunks:
        review_result = grammar_and_bias_review.invoke({"text": chunk})
        review_results.append(review_result.content)

    return review_results


"""
### Sample usage
"""
logger.info("### Sample usage")

grammary_and_bias_analysis_results = grammary_and_bias_analysis_article(
    full_article_text)

pprint.plogger.debug(grammary_and_bias_analysis_results)

"""
## LangGraph Workflow Integration

This section combines all modules into a unified workflow using LangGraph. By integrating summarization, fact-checking, tone analysis, quote extraction, and grammar and bias review, it creates a robust AI assistant tailored for journalism.

### Define State Structure

The state structure is defined using the `State` class, which is a `TypedDict`. This structure holds various elements of the article analysis process, including the current query, article text, chunks of the article, actions to be performed, and results from different analysis modules. The state structure ensures that all necessary information is organized and accessible throughout the workflow.

The `State` class includes the following fields:
- `current_query`: The user's current query or request.
- `article_text`: The full text of the article to be analyzed.
- `chunks`: List of text chunks for processing.
- `actions`: List of actions to be performed based on the user's query.
- `summary_result`: Result of the summarization process.
- `fact_check_result`: Result of the fact-checking process.
- `tone_analysis_result`: Result of the tone analysis process.
- `quote_extraction_result`: Result of the quote extraction process.
- `grammar_and_bias_review_result`: Result of the grammar and bias review process.
- `review_result`: Overall review result combining all analyses.
"""
logger.info("## LangGraph Workflow Integration")


class State(TypedDict):
    current_query: str
    article_text: str
    chunks: List[str]
    actions: List[str]
    summary_result: Optional[str]
    fact_check_result: Optional[TypedDict]
    tone_analysis_result: Optional[str]
    quote_extraction_result: Optional[str]
    grammar_and_bias_review_result: Optional[str]
    review_result: Optional[str]


"""
## Pipeline for Identifying Action Types

This step helps the assistant figure out what you’re asking for—whether it’s summarizing content, fact-checking, analyzing tone, extracting quotes, reviewing grammar and bias, or determining that no action is needed.

Using a well-crafted prompt template, the assistant organizes its response in a clear JSON format, making it easy to understand and act on. To make sure the responses are accurate, I’ve included `few-shot prompting`. By showing the model examples of what good responses look like, it learns to interpret requests more effectively.
"""
logger.info("## Pipeline for Identifying Action Types")


class SystemAction(TypedDict):
    actions: List[str]


action_prompt = PromptTemplate(
    input_variables=["input_text"],
    template=(
        "Identify the user's intended actions based on their input and return the actions in the following JSON format:\n"
        "{{\n"
        '  "actions": ["<summarization | fact-checking | tone-analysis | quote-extraction | grammar-and-bias-review | no-action-required | invalid>"]\n'
        "}}\n\n"
        "Guidelines:\n"
        "- If the user requests all actions or says 'everything' or 'full report,' respond with the list of all individual actions:\n"
        '{{\n'
        '    "actions": ["summarization", "fact-checking", "tone-analysis", "quote-extraction", "grammar-and-bias-review"]\n'
        "}}\n"
        "- If the user input requests multiple specific actions, list each action requested (e.g., 'summarization' and 'tone analysis' together as ['summarization', 'tone-analysis']).\n"
        "- If the user’s input does not relate to any accessible action, respond with:\n"
        '{{\n'
        '    "actions": ["invalid"]\n'
        "}}\n"
        "- If the user's input does not require any specific action, or wants to end the conversation, respond with:\n"
        '{{\n'
        '    "actions": ["no-action-required"]\n'
        "}}\n\n"
        "Important:\n"
        "- Only list all actions ('summarization', 'fact-checking', 'tone-analysis', 'quote-extraction', 'grammar-and-bias-review') if the user explicitly requests a comprehensive overview or all actions.\n"
        "- List only the actions explicitly requested by the user without inferring additional ones.\n\n"
        "Examples:\n"
        "- User input: 'Can you summarize the main points of this article for me?'\n"
        '  System action: {{ "actions": ["summarization"] }}\n'
        "- User input: 'I need to verify some claims in this article. Can you fact-check it?'\n"
        '  System action: {{ "actions": ["fact-checking"] }}\n'
        "- User input: 'Could you tell me the tone conveyed by this article?'\n"
        '  System action: {{ "actions": ["tone-analysis"] }}\n'
        "- User input: 'Identify any key quotes in this article that stand out.'\n"
        '  System action: {{ "actions": ["quote-extraction"] }}\n'
        "- User input: 'Can you check the grammar and point out any bias in this article?'\n"
        '  System action: {{ "actions": ["grammar-and-bias-review"] }}\n'
        "- User input: 'Please provide a comprehensive analysis, including all aspects.'\n"
        '  System action: {{ "actions": ["summarization", "fact-checking", "tone-analysis", "quote-extraction", "grammar-and-bias-review"] }}\n'
        "- User input: 'I want a tone analysis and quote extraction, please.'\n"
        '  System action: {{ "actions": ["tone-analysis", "quote-extraction"] }}\n'
        "- User input: 'I have another question that’s not related to these functions.'\n"
        '  System action: {{ "actions": ["invalid"] }}\n\n'
        "Input text:\n{input_text}"
    )
)


action_pipeline = action_prompt | llm.with_structured_output(SystemAction)


def get_user_actions(input_text: str) -> List[str]:
    """
    Identify the user's intended actions based on their input.
    """
    system_actions = action_pipeline.invoke({"input_text": input_text})

    return system_actions["actions"]


"""
### Sample usage
"""
logger.info("### Sample usage")

user_inputs = [
    "Can you summarize this article for me?",
    "I'm not sure about the accuracy of this article. Can you summarize it and fact-check it?",
    "I want to know the tone of this article.",
    "Can you extract any quotes from this article?",
    "I need a review of this article for grammar and bias.",
    "Can you provide a full report on this article?",
    "I would like to know everything about this article.",
    "I need help with something else.",
    "Ok, that's enough for now.",
    "What is the weather like today?",
]

for user_input in user_inputs:
    logger.debug(f"\n\nUser input: '{user_input}'")
    user_actions = get_user_actions(user_input)
    logger.debug(f"System actions: {user_actions}")

"""
### Define Node Functions

The following functions are defined to handle various actions such as summarization, fact-checking, tone analysis, quote extraction, and grammar and bias review. These functions utilize the previously defined pipelines and helper functions to process the article text and return the desired results.

- `get_or_create_chunks(state: State)`: Splits the article text into manageable chunks if not already done.
- `categorize_user_input(state: State)`: Categorizes the user input into specific actions based on keywords.
- `summarization_node(state: State)`: Handles the summarization action.
- `fact_checking_node(state: State)`: Handles the fact-checking action.
- `tone_analysis_node(state: State)`: Handles the tone analysis action.
- `quote_extraction_node(state: State)`: Handles the quote extraction action.
- `grammar_and_bias_review_node(state: State)`: Handles the grammar and bias review action.
"""
logger.info("### Define Node Functions")


def get_or_create_chunks(state: State):
    """
    This function gets the article text from the state and splits it into manageable chunks.
    The chunks are stored in the state to avoid recomputing them multiple times.
    """
    article_text = state["article_text"]
    chunks = state.get("chunks", [])
    if not chunks:
        chunks = chunk_large_text(article_text)
        state["chunks"] = chunks

    return chunks


def categorize_user_input(state: State) -> State:
    """
    This node handles the categorization of the user input to identify the intended actions.
    """
    query = state["current_query"]
    actions = get_user_actions(query)
    return {"actions": actions}


def summarization_node(state):
    """
    This node generates a summary of the article text.
    """
    chunks = get_or_create_chunks(state)
    article_text = state["article_text"]
    summary_result = summarize_article(article_text, chunks)
    return {"summary_result": summary_result}


def fact_checking_node(state: State) -> State:
    chunks = get_or_create_chunks(state)
    article_text = state["article_text"]
    fact_checking_results = fact_check_article(article_text, chunks)
    return {"fact_check_result": fact_checking_results}


def tone_analysis_node(state: State) -> State:
    chunks = get_or_create_chunks(state)
    article_text = state["article_text"]
    tone_analysis_results = tone_analysis_article(article_text, chunks)
    return {"tone_analysis_result": tone_analysis_results}


def quote_extraction_node(state: State) -> State:
    chunks = get_or_create_chunks(state)
    article_text = state["article_text"]
    quote_extraction_results = quote_extraction_article(article_text, chunks)
    return {"quote_extraction_result": quote_extraction_results}


def grammar_and_bias_review_node(state: State) -> State:
    chunks = get_or_create_chunks(state)
    article_text = state["article_text"]
    grammar_and_bias_review_results = (
        grammary_and_bias_analysis_article(article_text, chunks)
    )
    return {"grammar_and_bias_review_result": grammar_and_bias_review_results}


"""
### Routing function

The `route` function determines the next action based on the current state. It checks the actions specified in the state and returns the corresponding routes. If no specific actions are found, it returns `END`.
"""
logger.info("### Routing function")


def route(state: State) -> str:
    routes = []
    actions = state.get("actions", [])
    if "full report" in actions:
        routes = routes.values()
        routes.pop("no-action-required")
        routes.pop("invalid")
    else:
        for action in actions:
            if action in routes:
                routes.append(routes[action])

    if not routes:
        return END

    return routes


"""
### Create and Configure the Graph
Here we set up the LangGraph, defining nodes and edges to create our workflow.
"""
logger.info("### Create and Configure the Graph")

CATEGORY = "category"
SUMMARY = "summary"
FACT_CHECKING = "fact checking"
TONE_ANALYSIS = "tone analysis"
QUOTE_EXTRACTION = "quote extraction"
GRAMMAR_AND_BIAS_REVIEW = "grammar and bias review"

workflow = StateGraph(State)

workflow.add_node(CATEGORY, categorize_user_input)
workflow.add_node(SUMMARY, summarization_node)
workflow.add_node(FACT_CHECKING, fact_checking_node)
workflow.add_node(TONE_ANALYSIS, tone_analysis_node)
workflow.add_node(QUOTE_EXTRACTION, quote_extraction_node)
workflow.add_node(GRAMMAR_AND_BIAS_REVIEW, grammar_and_bias_review_node)

workflow.set_entry_point(CATEGORY)

workflow.add_conditional_edges(
    CATEGORY,
    lambda state: state["actions"],
    {
        "summarization": SUMMARY,
        "fact-checking": FACT_CHECKING,
        "tone-analysis": TONE_ANALYSIS,
        "quote-extraction": QUOTE_EXTRACTION,
        "grammar-and-bias-review": GRAMMAR_AND_BIAS_REVIEW,
        "no-action-required": END,
        "invalid": END,
    }
)

workflow.add_edge(SUMMARY, END)
workflow.add_edge(FACT_CHECKING, END)
workflow.add_edge(TONE_ANALYSIS, END)
workflow.add_edge(QUOTE_EXTRACTION, END)
workflow.add_edge(GRAMMAR_AND_BIAS_REVIEW, END)

journalist_assistant = workflow.compile()

display(
    Image(
        journalist_assistant.get_graph().draw_mermaid_png()
    )
)

"""
### Sample usage
"""
logger.info("### Sample usage")

full_report = journalist_assistant.invoke(
    {
        "current_query": "Can you provide a full report on this article?",
        "article_text": full_article_text,
    }
)

pprint.plogger.debug(full_report["fact_check_result"])

"""
### Report Generation
Format the results in markdown and save them to be easier to read.
"""
logger.info("### Report Generation")


def format_analysis_results(report_data):

    actions = report_data.get('actions', [])
    actions_text = "\n".join(
        f"- {action}" for action in actions) if actions else "No actions recorded."

    current_query = report_data.get('current_query', 'No query provided.')

    summary = report_data.get('summary_result', 'No summary available.')

    fact_check_result = report_data.get('fact_check_result', [])
    fact_check_text = ""
    if fact_check_result:
        for fact in fact_check_result:
            statement = fact.get('statement', 'No statement provided')
            status = fact.get('status', 'Unknown')
            explanation = fact.get('explanation', 'No explanation provided')
            fact_check_text += f"\n- **Statement:** {statement}\n  - **Status:** {status}\n  - **Explanation:** {explanation}\n"

            search_results = fact.get('search_results', [])
            if search_results:
                fact_check_text += "  - **Related Search Results:**\n"
                for result_set in search_results:
                    for result in result_set:
                        title = result.get('title', 'No title available')
                        summary = result.get(
                            'summary', 'No summary available').strip().replace('\n', ' ')
                        url = result.get('url', 'No URL available')
                        fact_check_text += f"    - **Title:** {title}\n\n      - **Summary:** {summary}\n\n      - **URL:** {url}\n\n"
    else:
        fact_check_text = "No fact-check results available."

    grammar_review = report_data.get('grammar_and_bias_review_result', [
                                     'No grammar or bias review available.'])
    grammar_review = "\n".join(grammar_review)

    tone_analysis = report_data.get('tone_analysis_result', [
                                    'No tone analysis available.'])
    tone_analysis = "\n".join(tone_analysis)

    quote_extraction = report_data.get(
        'quote_extraction_result', ['No quotes extracted.'])
    quote_extraction = "\n".join(quote_extraction)

    report_structured = f"""

{current_query}

{actions_text}

{summary if summary else "No summary available."}

{fact_check_text if fact_check_text else "No fact-check results available."}

{grammar_review if grammar_review else "No grammar or bias review available."}

{tone_analysis if tone_analysis else "No tone analysis available."}

{quote_extraction if quote_extraction else "No quotes found."}

---


"""

    return report_structured


report_content = format_analysis_results(full_report)

report_path = data_path / "ArticleAnalysis.md"
with open(report_path, "w") as report_file:
    report_file.write(report_content)

logger.debug(f"Report saved to {report_path} file")

"""
## Additional Considerations

While this Journalism-Focused AI Assistant provides powerful tools for enhancing journalistic practices, it’s important to acknowledge its limitations, potential areas for improvement, and specific scenarios where it excels.

### Limitations
- **Accuracy of Sources**: The reliability of fact-checking depends heavily on the quality and credibility of the sources retrieved via web searches. Results may vary depending on the availability of accurate information.
- **Bias in AI Outputs**: While tone and bias detection modules help mitigate issues, AI models themselves may occasionally introduce or miss biases due to training data limitations.
- **Real-Time Updates**: The system is not connected to live news feeds or continuously updating databases, which might limit its utility for breaking news scenarios.

### Potential Improvements
- **Real-Time Data Integration**: Connecting to live news feeds or APIs for real-time updates could enhance the relevance and timeliness of analyses.
- **Fine-Tuned Models**: Custom training or fine-tuning the AI models on journalistic datasets could improve the system’s understanding of nuances in reporting.
- **Enhanced Workflows**: Additional workflows for niche journalism use cases, such as investigative reporting or legal compliance, could increase flexibility.

### Specific Use Cases
- **Fact-Checking at Scale**: Ideal for verifying multiple claims in investigative journalism or research-heavy articles.
- **Tone and Bias Monitoring**: Helpful for ensuring balanced coverage and maintaining editorial neutrality.
- **Content Summarization**: Streamlines the review process for lengthy reports, interviews, or documents.
- **Transparent Reporting**: Quote extraction and grammar review make it easier to present clear and credible content.

By addressing these considerations, the tool can continue to evolve, becoming an even more indispensable resource for modern journalism.
"""
logger.info("## Additional Considerations")

logger.info("\n\n[DONE]", bright=True)
