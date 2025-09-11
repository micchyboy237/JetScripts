from dotenv import load_dotenv
from jet.llm.ollama.base_langchain import ChatOllama
from jet.logger import CustomLogger
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Any, Tuple, Optional
import nltk
import os
import re
import shutil
import traceback


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# Search and Summarize: AI-Powered Web Research Tool

## Overview
This Jupyter notebook implements an intelligent web research assistant that combines web search capabilities with AI-powered summarization. It automates the process of gathering information from the internet and distilling it into concise, relevant summaries, enhancing the efficiency of online research tasks.

## Motivation
In the age of information overload, efficiently extracting relevant knowledge from the vast expanse of the internet is increasingly challenging. This tool addresses several key pain points:

1. Time consumption in manual web searches
2. Information overload from multiple sources
3. Difficulty in quickly grasping key points from lengthy articles
4. Need for focused research on specific websites

By automating the search and summarization process, this tool aims to significantly reduce the time and cognitive load associated with web research, allowing users to quickly gain insights on any topic.

## Key Components
The notebook consists of several integral components:

1. **Web Search Module**: Utilizes DuckDuckGo's search API to fetch relevant web pages based on user queries.
2. **Result Parser**: Processes raw search results into a structured format for further analysis.
3. **Text Summarization Engine**: Leverages Ollama's language models to generate concise summaries of web content.
4. **Integration Layer**: Combines the search and summarization functionalities into a seamless workflow.

## Method Details

### Web Search Process
1. The user provides a search query and optionally specifies a target website.
2. If a specific site is given, the tool performs two searches:
   a. A site-specific search within the specified domain
   b. A general search excluding the specified site
3. Without a specific site, it conducts a general web search.
4. Search results are parsed to extract snippets, titles, and links.

### Summarization Approach
1. For each search result, the tool extracts the relevant text content.
2. The extracted text is sent to the AI model with a prompt requesting a concise summary.
3. The AI generates a summary in the form of 1-2 bullet points, capturing the key information.
4. Summaries are compiled along with their sources (title and link).

### Integration and Output
1. The tool combines the web search and summarization processes into a single function call.
2. It returns a formatted output containing summaries from multiple sources, each clearly attributed.
3. The output is designed to provide a quick overview of the topic, with links to full sources for further reading.

## Conclusion
This notebook demonstrates the power of combining web search capabilities with AI-driven summarization. It offers a practical solution for rapid information gathering and synthesis, applicable in various domains such as research, journalism, business intelligence, and general knowledge acquisition. By automating the tedious aspects of web research, it allows users to focus on higher-level analysis and decision-making based on quickly acquired, relevant information.

The modular design of this tool also allows for future enhancements, such as integration with different search engines, implementation of more advanced summarization techniques, or adaptation to specific domain knowledge requirements.

## Import Dependencies

This cell imports all necessary libraries and sets up the environment.
"""
logger.info("# Search and Summarize: AI-Powered Web Research Tool")


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

"""
## Initialize DuckDuckGo Search

This cell initializes the DuckDuckGo search tool.
"""
logger.info("## Initialize DuckDuckGo Search")

search = DuckDuckGoSearchResults()

"""
## Define Data Models

This cell defines the data model for text summarization.
"""
logger.info("## Define Data Models")


class SummarizeText(BaseModel):
    """Model for text to be summarized."""
    text: str = Field(..., title="Text to summarize",
                      description="The text to be summarized")


"""
## Helper Functions

This section contains helper functions for parsing search results and performing web searches.
"""
logger.info("## Helper Functions")


def parse_search_results(results_string: str) -> List[dict]:
    """Parse a string representation of search results into a list of dictionaries."""
    results = []
    entries = results_string.split(', snippet: ')
    for entry in entries[1:]:  # Skip the first split as it's empty
        parts = entry.split(', title: ')
        if len(parts) == 2:
            snippet = parts[0]
            title_link = parts[1].split(', link: ')
            if len(title_link) == 2:
                title, link = title_link
                results.append({
                    'snippet': snippet,
                    'title': title,
                    'link': link
                })
    return results


def perform_web_search(query: str, specific_site: Optional[str] = None) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Perform a web search based on a query, optionally including a specific website."""
    try:
        if specific_site:
            specific_query = f"site:{specific_site} {query}"
            logger.debug(f"Searching for: {specific_query}")
            specific_results = search.run(specific_query)
            logger.debug(f"Specific search results: {specific_results}")
            specific_parsed = parse_search_results(specific_results)

            general_query = f"-site:{specific_site} {query}"
            logger.debug(f"Searching for: {general_query}")
            general_results = search.run(general_query)
            logger.debug(f"General search results: {general_results}")
            general_parsed = parse_search_results(general_results)

            combined_results = (specific_parsed + general_parsed)[:3]
        else:
            logger.debug(f"Searching for: {query}")
            web_results = search.run(query)
            logger.debug(f"Web results: {web_results}")
            combined_results = parse_search_results(web_results)[:3]

        web_knowledge = [result.get('snippet', '')
                         for result in combined_results]
        sources = [(result.get('title', 'Untitled'), result.get('link', ''))
                   for result in combined_results]

        logger.debug(f"Processed web_knowledge: {web_knowledge}")
        logger.debug(f"Processed sources: {sources}")
        return web_knowledge, sources
    except Exception as e:
        logger.debug(f"Error in perform_web_search: {str(e)}")
        traceback.print_exc()
        return [], []


"""
## Text Summarization Function

This cell defines the function to summarize text using Ollama's language model.
"""
logger.info("## Text Summarization Function")


def summarize_text(text: str, source: Tuple[str, str]) -> str:
    """Summarize the given text using Ollama's language model."""
    try:
        llm = ChatOllama(model="llama3.2")
        prompt_template = "Please summarize the following text in 1-2 bullet points:\n\n{text}\n\nSummary:"
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["text"],
        )
        summary_chain = prompt | llm
        input_data = {"text": text}
        summary = summary_chain.invoke(input_data)

        summary_content = summary.content if hasattr(
            summary, 'content') else str(summary)

        formatted_summary = f"Source: {source[0]} ({source[1]})\n{summary_content.strip()}\n"
        return formatted_summary
    except Exception as e:
        logger.debug(f"Error in summarize_text: {str(e)}")
        return ""


"""
## Main Search and Summarize Function

This cell defines the main function that combines web search and text summarization.
"""
logger.info("## Main Search and Summarize Function")


def search_summarize(query: str, specific_site: Optional[str] = None) -> str:
    """Perform a web search and summarize the results."""
    web_knowledge, sources = perform_web_search(query, specific_site)

    if not web_knowledge or not sources:
        logger.debug("No web knowledge or sources found.")
        return ""

    summaries = [summarize_text(knowledge, source) for knowledge, source in zip(
        web_knowledge, sources) if summarize_text(knowledge, source)]

    combined_summary = "\n".join(summaries)
    return combined_summary


"""
## Example Usage

This cell demonstrates how to use the search_summarize function.
"""
logger.info("## Example Usage")

query = "What are the latest advancements in artificial intelligence?"
# Optional: specify a site or set to None
specific_site = "https://www.nature.com"
result = search_summarize(query, specific_site)
logger.debug(
    f"Summary of latest advancements in AI (including information from {specific_site if specific_site else 'various sources'}):")
logger.debug(result)

logger.info("\n\n[DONE]", bright=True)
