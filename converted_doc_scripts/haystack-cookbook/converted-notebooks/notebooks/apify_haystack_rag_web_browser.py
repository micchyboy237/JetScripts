from apify_haystack import ApifyDatasetFromActorCall
from haystack import Document
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OllamaFunctionCallingAdapterChatGenerator
from haystack.components.preprocessors import DocumentCleaner
from haystack.dataclasses import ChatMessage
from jet.logger import CustomLogger
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
# Search and browse the web with Apify and Haystack

Want to give any of your LLM applications the power to search and browse the web? In this cookbook, we'll show you how to use the [RAG Web Browser Actor](https://apify.com/apify/rag-web-browser) to search Google and extract content from web pages, then analyze the results using a large language model - all within the Haystack ecosystem using the apify-haystack integration.

This cookbook also demonstrates how to leverage the RAG Web Browser Actor with Haystack to create powerful web-aware applications. We'll explore multiple use cases showing how easy it is to:

1. [Search interesting topics](#search-interesting-topics)
2. [Analyze the results with OllamaFunctionCallingAdapterGenerator](#analyze-the-results-with-openaigenerator)
3. [Use the Haystack Pipeline for web search and analysis](#use-the-haystack-pipeline-for-web-search-and-analysis)
   
**We'll start by using the RAG Web Browser Actor to perform web searches and then use the OllamaFunctionCallingAdapterGenerator to analyze and summarize the web content**

## Install dependencies
"""
logger.info("# Search and browse the web with Apify and Haystack")

# !pip install -q apify-haystack==0.1.7 haystack-ai

"""
## Set up the API keys

You need to have an Apify account and obtain [APIFY_API_TOKEN](https://docs.apify.com/platform/integrations/api).

# You also need an OllamaFunctionCalling account and [OPENAI_API_KEY](https://platform.openai.com/docs/quickstart)
"""
logger.info("## Set up the API keys")

# from getpass import getpass

# os.environ["APIFY_API_TOKEN"] = getpass("Enter YOUR APIFY_API_TOKEN")
# os.environ["OPENAI_API_KEY"] = getpass("Enter YOUR OPENAI_API_KEY")

"""
## Search interesting topics

The [RAG Web Browser Actor](https://apify.com/apify/rag-web-browser) is designed to enhance AI and Large Language Model (LLM) applications by providing up-to-date web content. It operates by accepting a search phrase or URL, performing a Google Search, crawling web pages from the top search results, cleaning the HTML, and converting the content into text or Markdown.  

### Output Format
The output from the RAG Web Browser Actor is a JSON array, where each object contains:
- **crawl**: Details about the crawling process, including HTTP status code and load time.
- **searchResult**: Information from the search result, such as the title, description, and URL.
- **metadata**: Additional metadata like the page title, description, language code, and URL.
- **markdown**: The main content of the page, converted into Markdown format.

> For example, query: `rag web browser` returns:

```json
[
    {
        "crawl": {
            "httpStatusCode": 200,
            "httpStatusMessage": "OK",
            "loadedAt": "2024-11-25T21:23:58.336Z",
            "uniqueKey": "eM0RDxDQ3q",
            "requestStatus": "handled"
        },
        "searchResult": {
            "title": "apify/rag-web-browser",
            "description": "Sep 2, 2024 — The RAG Web Browser is designed for Large Language Model (LLM) applications ...",
            "url": "https://github.com/apify/rag-web-browser"
        },
        "metadata": {
            "title": "GitHub - apify/rag-web-browser: RAG Web Browser is an Apify Actor to feed your LLM applications ...",
            "description": "RAG Web Browser is an Apify Actor to feed your LLM applications ...",
            "languageCode": "en",
            "url": "https://github.com/apify/rag-web-browser"
        },
        "markdown": "# apify/rag-web-browser: RAG Web Browser is an Apify Actor ..."
    }
]
```

We will convert this JSON to a Haystack Document using the `dataset_mapping_function` as follows:
"""
logger.info("## Search interesting topics")


def dataset_mapping_function(dataset_item: dict) -> Document:
    return Document(
        content=dataset_item.get("markdown"),
        meta={
            "title": dataset_item.get("metadata", {}).get("title"),
            "url": dataset_item.get("metadata", {}).get("url"),
            "language": dataset_item.get("metadata", {}).get("languageCode")
        }
    )

"""
Now set up the `ApifyDatasetFromActorCall` component:
"""
logger.info("Now set up the `ApifyDatasetFromActorCall` component:")


document_loader = ApifyDatasetFromActorCall(
    actor_id="apify/rag-web-browser",
    run_input={
        "maxResults": 2,
        "outputFormats": ["markdown"],
        "requestTimeoutSecs": 30
    },
    dataset_mapping_function=dataset_mapping_function,
)

"""
Check out other `run_input` parameters at [Github for the RAG web browser](https://github.com/apify/rag-web-browser?tab=readme-ov-file#query-parameters).

Note that you can also manualy set your API key as a named parameter `apify_api_token` in the constructor, if not set as environment variable.

### Run the Actor and fetch results

Let's run the Actor with a sample query and fetch the results. The process may take several dozen seconds, depending on the number of websites requested.
"""
logger.info("### Run the Actor and fetch results")

query = "Artificial intelligence latest developments"

result = document_loader.run(run_input={"query": query})
documents = result.get("documents", [])

for doc in documents:
    logger.debug(f"Title: {doc.meta['title']}")
    logger.debug(f"Truncated content:  \n {doc.content[:100]} ...")
    logger.debug("---")

"""
## Analyze the results with OllamaFunctionCallingAdapterChatGenerator

Use the OllamaFunctionCallingAdapterChatGenerator to analyze and summarize the web content.
"""
logger.info("## Analyze the results with OllamaFunctionCallingAdapterChatGenerator")


generator = OllamaFunctionCallingAdapterChatGenerator(model="llama3.2")

for doc in documents:
    result = generator.run(messages=[ChatMessage.from_user(doc.content)])
    summary = result["replies"][0].text  # Accessing the generated text
    logger.debug(f"Summary for {doc.meta.get('title')} available from {doc.meta.get('url')}: \n{summary}\n ---")

"""
## Use the Haystack Pipeline for web search and analysis

Now let's create a more sophisticated pipeline that can handle different types of content and generate specialized analyses. We'll create a pipeline that:

1. Searches the web using RAG Web Browser
2. Cleans and filters the documents
3. Routes them based on content type
4. Generates customized summaries for different types of content
"""
logger.info("## Use the Haystack Pipeline for web search and analysis")


def dataset_mapping_function(dataset_item: dict) -> Document:
    max_chars = 10000
    content = dataset_item.get("markdown", "")
    return Document(
        content=content[:max_chars],
        meta={
            "title": dataset_item.get("metadata", {}).get("title"),
            "url": dataset_item.get("metadata", {}).get("url"),
            "language": dataset_item.get("metadata", {}).get("languageCode")
        }
    )

def create_pipeline(query: str) -> Pipeline:

    document_loader = ApifyDatasetFromActorCall(
        actor_id="apify/rag-web-browser",
        run_input={
            "query": query,
            "maxResults": 2,
            "outputFormats": ["markdown"]
        },
        dataset_mapping_function=dataset_mapping_function,
    )

    cleaner = DocumentCleaner(
        remove_empty_lines=True,
        remove_extra_whitespaces=True,
        remove_repeated_substrings=True
    )

    prompt_template = """
    Analyze the following content and provide:
    1. Key points and findings
    2. Practical implications
    3. Notable conclusions
    Be concise.

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Analysis:
    """

    prompt_builder = ChatPromptBuilder(template=[ChatMessage.from_user(prompt_template)], required_variables="*")

    generator = OllamaFunctionCallingAdapterChatGenerator(model="llama3.2")

    pipe = Pipeline()
    pipe.add_component("loader", document_loader)
    pipe.add_component("cleaner", cleaner)
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("generator", generator)

    pipe.connect("loader", "cleaner")
    pipe.connect("cleaner", "prompt_builder")
    pipe.connect("prompt_builder", "generator")

    return pipe

def research_topic(query: str) -> str:
    pipeline = create_pipeline(query)
    result = pipeline.run({})
    return result["generator"]["replies"][0].text

query = "latest developments in AI ethics"
analysis = research_topic(query)

logger.debug("Analysis Result:")
logger.debug(analysis)

"""
You can customize the pipeline further by:
- Adding more sophisticated routing logic
- Implementing additional preprocessing steps
- Creating specialized generators for different content types
- Adding error handling and retries
- Implementing caching for improved performance

This completes our exploration of using Apify's RAG Web Browser with Haystack for web-aware AI applications. The combination of web search capabilities with sophisticated content processing and analysis creates powerful possibilities for research, analysis and many other tasks.
"""
logger.info("You can customize the pipeline further by:")

logger.info("\n\n[DONE]", bright=True)