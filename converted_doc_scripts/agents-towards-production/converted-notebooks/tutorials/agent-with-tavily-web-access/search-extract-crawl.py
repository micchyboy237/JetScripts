from dotenv import load_dotenv
from jet.logger import CustomLogger
from tavily import TavilyClient
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
![](https://europe-west1-atp-views-tracker.cloudfunctions.net/working-analytics?notebook=tutorials--agent-with-tavily-web-access--search-extract-crawl)

# 1. Search, Extract, and Crawl the Web 🌐

Welcome! In this tutorial, you'll gain hands-on experience with the core capabilities of the Tavily API—searching the web with semantic understanding, extracting content from live web pages, and crawling entire websites. 

These skills are essential for anyone building AI agents or applications that need up-to-date, relevant information from the internet. By learning how to programmatically access and process real-time web data, you'll be able to bridge the gap between static language models and the dynamic world they operate in, making your agents smarter, more accurate, and context-aware.

We'll cover:
- How to perform web searches and retrieve the most relevant results
- How to extract clean, usable content from any URL
- How to crawl websites to gather comprehensive information
- How to fine-tune your queries with advanced parameters
 

---

## Getting Started

Follow these steps to set up:

1. **Sign up** for Tavily at [app.tavily.com](https://app.tavily.com/home/?utm_source=github&utm_medium=referral&utm_campaign=nir_diamant) to get your API key.

   *Refer to the screenshots linked below for step-by-step guidance:*

   - ![Screenshot: Signup Page](assets/sign-up.png)
   - ![Screenshot: Tavily API Keys Dashboard](assets/api-key.png)


2. **Copy your API key** from your Tavily account dashboard.

3. **Paste your API key** into the cell below and execute the cell.
"""
logger.info("# 1. Search, Extract, and Crawl the Web 🌐")

# !echo "TAVILY_API_KEY=<your-tavily-api-key>" >> .env

"""
Install dependencies in the cell below.
"""
logger.info("Install dependencies in the cell below.")

# %pip install --upgrade tavily-python --quiet

"""
### Setting Up Your Tavily API Client

The code below will instantiate the Tavily client with your API key.
"""
logger.info("### Setting Up Your Tavily API Client")

# import getpass

load_dotenv()

if not os.environ.get("TAVILY_API_KEY"):
#     os.environ["TAVILY_API_KEY"] = getpass.getpass("TAVILY_API_KEY:\n")

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

"""
## Search 🔍 

Let's run a basic web search query to retrieve up-to-date information about NYC.
"""
logger.info("## Search 🔍")

search_results = tavily_client.search(
    query="What happened in NYC today?", max_results=5
)

"""
This search invocation will return 5 results. Each result includes the web page's title, URL, a content snippet for RAG purposes, and a semantic score indicating how closely the result matches your query.
"""
logger.info("This search invocation will return 5 results. Each result includes the web page's title, URL, a content snippet for RAG purposes, and a semantic score indicating how closely the result matches your query.")

for result in search_results["results"]:
    logger.debug(result["title"])
    logger.debug(result["url"])
    logger.debug(result["content"])
    logger.debug(result["score"])
    logger.debug("\n")

"""
Let's experiment with different API parameter configurations to see Tavily in action. Try everything from broad topics to specific questions! You can adjust parameters such as the number of results, time range, and domain filters to tailor your search. For more information, read the search [API reference](https://docs.tavily.com/documentation/api-reference/endpoint/search?utm_source=github&utm_medium=referral&utm_campaign=nir_diamant) and [best practices guide](https://docs.tavily.com/documentation/best-practices/best-practices-search?utm_source=github&utm_medium=referral&utm_campaign=nir_diamant). Let's apply a time range filter, domain filter, and use the `news` search topic.
"""
logger.info("Let's experiment with different API parameter configurations to see Tavily in action. Try everything from broad topics to specific questions! You can adjust parameters such as the number of results, time range, and domain filters to tailor your search. For more information, read the search [API reference](https://docs.tavily.com/documentation/api-reference/endpoint/search?utm_source=github&utm_medium=referral&utm_campaign=nir_diamant) and [best practices guide](https://docs.tavily.com/documentation/best-practices/best-practices-search?utm_source=github&utm_medium=referral&utm_campaign=nir_diamant). Let's apply a time range filter, domain filter, and use the `news` search topic.")

search_results = tavily_client.search(
    query="Anthropic model release?",
    max_results=5,
    time_range="month",
    include_domains=["techcrunch.com"],
    topic="news",
)

"""
Notice that all the results are from `techcrunch.com` and are limited to the past month. By setting the `news` topic, our search is focused on trusted third-party news sources.
"""
logger.info("Notice that all the results are from `techcrunch.com` and are limited to the past month. By setting the `news` topic, our search is focused on trusted third-party news sources.")

for result in search_results["results"]:
    logger.debug(result["title"])
    logger.debug(result["url"])
    logger.debug(result["content"])
    logger.debug("\n")

"""
## Extract 📄 

Next, we'll use the Tavily extract endpoint to retrieve the complete content (i.e., `raw_content`) of each page using the URLs from our previous search results. Instead of just using the short content snippets from the search, this allows us to access the full text of each page. For efficiency, the extract endpoint can process up to 20 URLs at once in a single call. For more information, read the extract [API reference](https://docs.tavily.com/documentation/api-reference/endpoint/extract?utm_source=github&utm_medium=referral&utm_campaign=nir_diamant) and [best practices guide](https://docs.tavily.com/documentation/best-practices/best-practices-extract?utm_source=github&utm_medium=referral&utm_campaign=nir_diamant).
"""
logger.info("## Extract 📄")

extract_results = tavily_client.extract(
    urls=[result["url"] for result in search_results["results"]],
)

"""
Let's look at the raw content, which provides much more detailed and complete information than the short content snippets shown earlier. If you use raw content as input to LLMs, remember to consider your model's context window limits.
"""
logger.info("Let's look at the raw content, which provides much more detailed and complete information than the short content snippets shown earlier. If you use raw content as input to LLMs, remember to consider your model's context window limits.")

for result in extract_results["results"]:
    logger.debug(result["url"])
    logger.debug(result["raw_content"])
    logger.debug("\n")

"""
Rather than using the Extract endpoint to return raw page content, we can combine the search and extract endpoints into a API call by using the search endpoint with the `include_raw_content=True` parameter.
"""
logger.info("Rather than using the Extract endpoint to return raw page content, we can combine the search and extract endpoints into a API call by using the search endpoint with the `include_raw_content=True` parameter.")

search_results = tavily_client.search(
    query="Anthropic model release?",
    max_results=1,
    include_raw_content=True,
)

"""
Each search result now contains the web page's title, URL, semantic score, a content snippet, and the complete raw content. Tavily's flexible and modular API supports building a wide range of agentic systems, regardless of model size.
"""
logger.info("Each search result now contains the web page's title, URL, semantic score, a content snippet, and the complete raw content. Tavily's flexible and modular API supports building a wide range of agentic systems, regardless of model size.")

for result in search_results["results"]:
    logger.debug(result["url"])
    logger.debug(result["content"])
    logger.debug(result["score"])
    logger.debug(result["raw_content"])
    logger.debug("\n")

"""
## Crawl 🕸️ 

Now let’s use Tavily to crawl a webpage and extract all its links. Web crawling is the process of automatically navigating through websites by following hyperlinks to discover numerous web pages and URLs (think of it like falling down a Wikipedia rabbit hole 🐇—clicking from page to page, diving deeper into interconnected topics). For autonomous web agents, this capability is essential for accessing deep web data which might be difficult to retrieve via search. For more information, read the crawl [API reference](https://docs.tavily.com/documentation/api-reference/endpoint/crawl?utm_source=github&utm_medium=referral&utm_campaign=nir_diamant) and [best practices guide](https://docs.tavily.com/documentation/best-practices/best-practices-crawl?utm_source=github&utm_medium=referral&utm_campaign=nir_diamant).

Let's begin by crawling the Tavily website to gather all nested pages.
"""
logger.info("## Crawl 🕸️")

crawl_results = tavily_client.crawl(url="tavily.com")

"""
We can see all the nested URLs.
"""
logger.info("We can see all the nested URLs.")

for result in crawl_results["results"]:
    logger.debug(result["url"])

"""
The crawl endpoint also returns the raw page content of each URL.
"""
logger.info("The crawl endpoint also returns the raw page content of each URL.")

for result in crawl_results["results"]:
    logger.debug(result["url"])
    logger.debug(result["raw_content"])
    logger.debug("\n")

"""
If you're interested in just the links (without the full page content), use the Map endpoint. It's a faster and more cost-effective way to retrieve all the links from a site.
"""
logger.info("If you're interested in just the links (without the full page content), use the Map endpoint. It's a faster and more cost-effective way to retrieve all the links from a site.")

map_results = tavily_client.map(url="tavily.com")

"""
Let's view the results, which only contain the links in this case.
"""
logger.info("Let's view the results, which only contain the links in this case.")

map_results

"""
The `instructions` parameter of the crawl/map endpoint is a powerful feature that lets you guide the web crawl using natural language instructions.
"""
logger.info("The `instructions` parameter of the crawl/map endpoint is a powerful feature that lets you guide the web crawl using natural language instructions.")

guided_map_results = tavily_client.map(
    url="tavily.com", instructions="find only the developer docs"
)

"""
Now, the results will only include developer docs from the Tavily webpage.
"""
logger.info("Now, the results will only include developer docs from the Tavily webpage.")

guided_map_results

"""
Experiment with different URLs to see how Tavily maps the structure of different websites. How would you integrate this into your agentic systems...🤔?

## Conclusion & Next Steps
 
In this tutorial, you learned how to:
- Perform real-time web searches using the Tavily API
- Extract content from web pages
- Crawl and map websites to gather links and information
- Guide crawls with natural language instructions for targeted data extraction
 
These foundational skills enable your agents to access and utilize up-to-date web information, making them more powerful and context-aware. Feel free to experiment with the Tavily API in the [playground](https://app.tavily.com/playground?utm_source=github&utm_medium=referral&utm_campaign=nir_diamant) and read the [best practices guide](https://docs.tavily.com/documentation/best-practices/best-practices-search?utm_source=github&utm_medium=referral&utm_campaign=nir_diamant) to optimize for your use case.
 
**Ready to take the next step?**  
In **Tutorial #2: Building a Web Agent** that can search, extract, and crawl autonomously, you'll combine these capabilities to build a fully autonomous web agent. This agent will be able to reason, decide when to search, crawl, or extract, and integrate web data into its workflow—all powered by Tavily.
 
[👉 **Continue to Tutorial #2!**](./web-agent-tutorial.ipynb)
"""
logger.info("## Conclusion & Next Steps")

logger.info("\n\n[DONE]", bright=True)