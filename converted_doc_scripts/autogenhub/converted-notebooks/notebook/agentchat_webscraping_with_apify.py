from apify_client import ApifyClient
from autogen import ConversableAgent, register_function
from jet.logger import CustomLogger
from typing_extensions import Annotated
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Web Scraping using Apify Tools

This notebook shows how to use Apify tools with AutoGen agents to
scrape data from a website and formate the output.

First we need to install the Apify SDK and the AutoGen library.
"""
logger.info("# Web Scraping using Apify Tools")

# ! pip install -qqq autogen apify-client

"""
Setting up the LLM configuration and the Apify API key is also required.
"""
logger.info("Setting up the LLM configuration and the Apify API key is also required.")


config_list = [
#     {"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY")},
]

apify_api_key = os.getenv("APIFY_API_KEY")

"""
Let's define the tool for scraping data from the website using Apify actor.
Read more about tool use in this [tutorial chapter](/docs/tutorial/tool-use).
"""
logger.info("Let's define the tool for scraping data from the website using Apify actor.")



def scrape_page(url: Annotated[str, "The URL of the web page to scrape"]) -> Annotated[str, "Scraped content"]:
    client = ApifyClient(token=apify_api_key)

    run_input = {
        "startUrls": [{"url": url}],
        "useSitemaps": False,
        "crawlerType": "playwright:firefox",
        "includeUrlGlobs": [],
        "excludeUrlGlobs": [],
        "ignoreCanonicalUrl": False,
        "maxCrawlDepth": 0,
        "maxCrawlPages": 1,
        "initialConcurrency": 0,
        "maxConcurrency": 200,
        "initialCookies": [],
        "proxyConfiguration": {"useApifyProxy": True},
        "maxSessionRotations": 10,
        "maxRequestRetries": 5,
        "requestTimeoutSecs": 60,
        "dynamicContentWaitSecs": 10,
        "maxScrollHeightPixels": 5000,
        "removeElementsCssSelector": """nav, footer, script, style, noscript, svg,
    [role=\"alert\"],
    [role=\"banner\"],
    [role=\"dialog\"],
    [role=\"alertdialog\"],
    [role=\"region\"][aria-label*=\"skip\" i],
    [aria-modal=\"true\"]""",
        "removeCookieWarnings": True,
        "clickElementsCssSelector": '[aria-expanded="false"]',
        "htmlTransformer": "readableText",
        "readableTextCharThreshold": 100,
        "aggressivePrune": False,
        "debugMode": True,
        "debugLog": True,
        "saveHtml": True,
        "saveMarkdown": True,
        "saveFiles": False,
        "saveScreenshots": False,
        "maxResults": 9999999,
        "clientSideMinChangePercentage": 15,
        "renderingTypeDetectionPercentage": 10,
    }

    run = client.actor("aYG0l9s7dbB7j3gbS").call(run_input=run_input)

    text_data = ""
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        text_data += item.get("text", "") + "\n"

    average_token = 0.75
    max_tokens = 20000  # slightly less than max to be safe 32k
    text_data = text_data[: int(average_token * max_tokens)]
    return text_data

"""
Create the agents and register the tool.
"""
logger.info("Create the agents and register the tool.")


scraper_agent = ConversableAgent(
    "WebScraper",
    llm_config={"config_list": config_list},
    system_message="You are a web scrapper and you can scrape any web page using the tools provided. "
    "Returns 'TERMINATE' when the scraping is done.",
)

user_proxy_agent = ConversableAgent(
    "UserProxy",
    llm_config=False,  # No LLM for this agent.
    human_input_mode="NEVER",
    code_execution_config=False,  # No code execution for this agent.
    is_termination_msg=lambda x: x.get("content", "") is not None and "terminate" in x["content"].lower(),
    default_auto_reply="Please continue if not finished, otherwise return 'TERMINATE'.",
)

register_function(
    scrape_page,
    caller=scraper_agent,
    executor=user_proxy_agent,
    name="scrape_page",
    description="Scrape a web page and return the content.",
)

"""
Start the conversation for scraping web data. We used the
`reflection_with_llm` option for summary method
to perform the formatting of the output into a desired format.
The summary method is called after the conversation is completed
given the complete history of the conversation.
"""
logger.info("Start the conversation for scraping web data. We used the")

chat_result = user_proxy_agent.initiate_chat(
    scraper_agent,
    message="Can you scrape agentops.ai for me?",
    summary_method="reflection_with_llm",
    summary_args={
        "summary_prompt": """Summarize the scraped content and format summary EXACTLY as follows:
---
*Company name*:
`Acme Corp`
---
*Website*:
`acmecorp.com`
---
*Description*:
`Company that does things.`
---
*Tags*:
`Manufacturing. Retail. E-commerce.`
---
*Takeaways*:
`Provides shareholders with value by selling products.`
---
*Questions*:
`What products do they sell? How do they make money? What is their market share?`
---
"""
    },
)

"""
The output is stored in the summary.
"""
logger.info("The output is stored in the summary.")

logger.debug(chat_result.summary)

logger.info("\n\n[DONE]", bright=True)