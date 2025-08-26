import asyncio
from jet.transformers.formatters import format_json
from IPython.display import display, Image
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from enum import Enum
from google.colab import userdata
from gradio import ChatMessage
from importnb import Notebook
from jet.llm.ollama.base_langchain import ChatOllama
from jet.logger import CustomLogger
from langchain.tools import DuckDuckGoSearchResults
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import StateGraph, END, START, MessagesState
from langsmith import Client
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from lxml import html
from patchright.async_api import async_playwright
from pydantic import BaseModel
from typing import Literal, Optional
from typing import TypedDict
from typing import TypedDict, Dict, List, Any
import asyncio
import gradio as gr
import json
import openai
import os
import queue
import re
import shutil
import threading
import time


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Smart Product Buyer AI Agent

## Overview

This notebook details the **Smart Product Buyer AI Agent**, developed as a **Proof of Concept (PoC)** to assist users in making informed buying decisions. While the current implementation focuses on car purchasing, it is designed to be **easily extendable** to support additional websites and even other product categories. The project leverages **LangGraph** and **LLM-based intelligence** to provide an interactive, efficient, and adaptable user experience.

## Detailed Explanation

### Motivation
Modern consumers face challenges navigating the vast array of product options online. This agent streamlines the search and decision-making process by:
- Understanding user needs and preferences.
- Refining and applying complex filters across multiple platforms. For now, it only supports AutoTrader, but it can be extended to other platforms easily by adding a new scraper in the `scrapers` folder.
- Providing actionable insights and recommendations.

### Key Components
1. **User Input Processing**: Understands user requirements and preferences dynamically using LLM-powered interactions.
2. **Filter Refinement**: Tailors search filters to match user-defined parameters.
3. **Web Scraping and Integration**: Interfaces with platforms like AutoTrader to fetch and present relevant listings.
4. **Summarization and Insights**: Provides concise summaries and insights into listings, including general market reliability.

### Agent Architecture
The agent follows a structured workflow:
- **User Need Assessment**: Gathers and summarizes user preferences.
- **Filter Building**: Constructs and applies search filters.
- **Listing Retrieval**: Collects data from integrated platforms.
- **Insight Delivery**: Provides additional information and recommendations.

### Benefits
- **Efficiency**: Reduces the time spent searching and comparing products.
- **Clarity**: Summarizes complex data into actionable insights.
- **Flexibility**: Adaptable to various product categories beyond cars.

## Visual Representation

Below is the diagram of the agent's architecture:

![Smart Product Buyer Agent Architecture](../images/car_buyer_agent_langgraph.png)

---

## Code Setup

The following steps guide you through setting up the necessary environment and running the agent.

### Prerequisites
Ensure you have Python and Jupyter Notebook installed on your system.

The project can run on Google Colab or any local Jupyter Notebook environment, but for some reason scraping is very slow on Google Colab.
We recommend running the project on a local Jupyter Notebook environment, preferably on macOS or Linux. If you're using Windows, it's best to run it under WSL for optimal performance.

To start the Gradio interface, just run all the cells in the notebook, then connect to the Gradio interface by clicking the link provided.

You can set USE_GRADIO variable to False to run the project without Gradio interface. This makes it easier to debug and test the project.

Set up the .env file with the necessary API keys:
# - OPENAI_API_KEY (required)
- LANGCHAIN_API_KEY (not required if LangSmith is not used)

## About the Team

The **Smart Product Buyer AI Agent** was created by **Aurore Pistono**, **Clément Florval**, and **Louis Gauthier**, all members of the **[Digiwave](https://dgwave.net)** team. Together, we bring expertise in AI, innovative development strategies, and a passion for creating impactful technological solutions.

### Connect with the Team:
- [Aurore Pistono on LinkedIn](https://www.linkedin.com/in/aurore-pistono/)
- [Clément Florval on LinkedIn](https://www.linkedin.com/in/clement-florval/)
- [Louis Gauthier on LinkedIn](https://www.linkedin.com/in/louis-gthier/)

### Install Required Libraries

This cell installs all the necessary Python packages required for the project. Below is a description of each package:

1. **`langgraph`**:
   - Provides tools for building and managing state-based workflows, particularly useful for conversational agents.

2. **`langchain` and `langchain-openai`**:
   - Frameworks for developing applications powered by Language Models (LLMs). `langchain-openai` is specifically tailored for Ollama's APIs.

3. **`langchain-community`**:
   - A community-driven package offering additional tools and integrations for LangChain.

4. **`importnb`**:
   - Enables the import of Jupyter Notebooks as Python modules.

5. **`python-dotenv`**:
   - Manages environment variables stored in a `.env` file, providing secure access to sensitive data like API keys.

6. **`patchright`**:
   - Used for patching and managing updates for certain libraries or configurations.

7. **`lxml`**:
   - A powerful library for parsing and working with XML and HTML documents, often used in web scraping.

8. **`nest_asyncio`**:
   - Allows running asynchronous event loops within Jupyter Notebooks, resolving conflicts caused by its built-in event loop.

9. **`playwright`**:
   - A library for browser automation, used for scraping or testing web applications.

10. **`duckduckgo-search`**:
    - Provides programmatic access to DuckDuckGo search results for retrieving web-based information.

11. **`gradio`**:
    - A framework for building user-friendly web-based interfaces, commonly used for showcasing AI applications.
"""
logger.info("# Smart Product Buyer AI Agent")

# %pip install langgraph
# %pip install langchain
# %pip install langchain-openai
# %pip install langchain-community
# %pip install importnb
# %pip install python-dotenv
# %pip install patchright
# %pip install lxml
# %pip install nest_asyncio
# %pip install playwright
# %pip install duckduckgo-search
# %pip install gradio

"""
### Import Necessary Libraries

This cell imports the required libraries and modules for building and managing the workflow, integrating Ollama's APIs, handling asynchronous operations, and working with custom scrapers.
"""
logger.info("### Import Necessary Libraries")





# import nest_asyncio
# nest_asyncio.apply()

"""
### Web Scraping and Interface Definitions

This section defines essential functions and classes for interacting with websites, retrieving listings, and applying filters based on user requirements.

1. **`scroll_to_bottom`**:
   - Implements dynamic content loading by scrolling to the bottom of a webpage iteratively, ensuring all elements are loaded before scraping.

2. **`block_unnecessary_resources`**:
   - Improves scraping efficiency by blocking non-essential resources such as images during browser automation.

3. **`WebsiteInterface` Abstract Class**:
   - Serves as a base class for defining web scraper interfaces.
   - Provides structure for crawling websites and managing filters, ensuring consistency across multiple platforms.

4. **`AutotraderInterface`**:
   - A concrete implementation of the `WebsiteInterface` tailored for scraping car listings from AutoTrader.
   - Includes methods for:
     - Retrieving and processing car listings (`crawl`).
     - Fetching detailed information for a specific listing (`crawl_listing`).
     - Constructing filters and generating query URLs dynamically using LLM responses.

The modular design allows for easy addition of new platforms by extending the `WebsiteInterface` class and implementing the required methods.
"""
logger.info("### Web Scraping and Interface Definitions")

async def scroll_to_bottom(page, scroll_delay=0.1):
    """
    Scroll to the bottom of the page iteratively, with delays to ensure dynamic content is fully loaded.

    Args:
        page: The Playwright page instance.
        scroll_delay: Delay in seconds between scrolls to allow content loading.
    """

    logger.debug("Scrolling through the page...")

    scroll_size = 2160

    next_scroll = scroll_size
    for i in range(3):
        async def run_async_code_9ddac758():
            await page.evaluate(f"window.scrollTo(0, {next_scroll})")
            return 
         = asyncio.run(run_async_code_9ddac758())
        logger.success(format_json())

        next_scroll += scroll_size

        async def run_async_code_8d7aa160():
            await asyncio.sleep(scroll_delay)
            return 
         = asyncio.run(run_async_code_8d7aa160())
        logger.success(format_json())

    logger.debug("Finished scrolling through the page.")

async def block_unnecessary_resources(route):
    if route.request.resource_type in ["image"]:
        async def run_async_code_bcc496ee():
            await route.abort()
            return 
         = asyncio.run(run_async_code_bcc496ee())
        logger.success(format_json())
    else:
        async def run_async_code_79198b20():
            await route.continue_()
            return 
         = asyncio.run(run_async_code_79198b20())
        logger.success(format_json())

class WebsiteInterface(ABC):
    def __init__(self):
        self.base_url = ""

    @abstractmethod
    async def crawl(self) -> List[Dict[str, str]]:
        """
        Abstract method to crawl the website and extract listings.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_filters_info(self) -> str:
        """
        Abstract method to return a prompt for the LLM describing the filters and expected output format.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def set_filters_from_llm_response(self, llm_response: str):
        """
        Abstract method to process the LLM's response and set the URL with appropriate filters.
        Must be implemented by subclasses.
        """
        pass

class AutotraderInterface(WebsiteInterface):
    def __init__(self):
        self.base_url = "https://www.autotrader.com/cars-for-sale/all-cars"

    async def crawl(self) -> List[Dict[str, str]]:
        listings = []

        url = self.url

        async def run_async_code_d66fbb88():
            async def run_async_code_22449de9():
                playwright = await async_playwright().start()
                return playwright
            playwright = asyncio.run(run_async_code_22449de9())
            logger.success(format_json(playwright))
            return playwright
        playwright = asyncio.run(run_async_code_d66fbb88())
        logger.success(format_json(playwright))

        async def run_async_code_8ecc12ef():
            browser = await playwright.chromium.launch(headless=True,
            return browser
        browser = asyncio.run(run_async_code_8ecc12ef())
        logger.success(format_json(browser))
                                                    args=[
                                                            "--no-sandbox",
                                                            "--disable-setuid-sandbox",
                                                            "--disable-dev-shm-usage",
                                                            "--disable-extensions",
                                                            "--disable-gpu"
                                                    ]
                                                    )

        async def async_func_78():
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
                viewport={"width": 1920, "height": 1080},
                locale="en-US",
                timezone_id="America/New_York",
            )
            return context
        context = asyncio.run(async_func_78())
        logger.success(format_json(context))


        logger.debug("Opening browser page")

        async def run_async_code_94ce32fc():
            async def run_async_code_7cc17402():
                page = await context.new_page()
                return page
            page = asyncio.run(run_async_code_7cc17402())
            logger.success(format_json(page))
            return page
        page = asyncio.run(run_async_code_94ce32fc())
        logger.success(format_json(page))

        async def run_async_code_c46863ef():
            await page.route("**/*", block_unnecessary_resources)
            return 
         = asyncio.run(run_async_code_c46863ef())
        logger.success(format_json())

        logger.debug("Loading page")

        async def run_async_code_93a852d7():
            await page.goto(url, wait_until="domcontentloaded")
            return 
         = asyncio.run(run_async_code_93a852d7())
        logger.success(format_json())

        logger.debug("Page partially loaded. Starting to scroll.")

        async def run_async_code_9aa622c5():
            await scroll_to_bottom(page)
            return 
         = asyncio.run(run_async_code_9aa622c5())
        logger.success(format_json())

        async def run_async_code_88e08c31():
            async def run_async_code_0d9a82db():
                page_content = await page.content()
                return page_content
            page_content = asyncio.run(run_async_code_0d9a82db())
            logger.success(format_json(page_content))
            return page_content
        page_content = asyncio.run(run_async_code_88e08c31())
        logger.success(format_json(page_content))

        tree = html.fromstring(page_content)

        listings_elements = tree.xpath('//div[@data-cmp="inventoryListing"]')

        listings = []

        for listing in listings_elements:
            car_data = {}
            car_data['title'] = listing.xpath('.//h2[@data-cmp="subheading"]/text()')
            car_data['mileage'] = listing.xpath('.//div[@data-cmp="mileageSpecification"]/text()')
            car_data['price'] = listing.xpath('.//div[@data-cmp="firstPrice"]/text()')
            car_data['dealer'] = listing.xpath('.//div[@class="text-subdued"]/text()')
            car_data['phone'] = listing.xpath('.//span[@data-cmp="phoneNumber"]/text()')
            car_data['url'] = listing.xpath('.//a[@data-cmp="link"]/@href')
            car_data['image'] = listing.xpath('.//img[@data-cmp="inventoryImage"]/@src')

            car_data = {key: (val[0].strip() if val else None) for key, val in car_data.items()}

            car_data['url'] = car_data['url'].split('?')[0]

            car_data['url'] = re.sub(r'^(https?://[^/]+).*$', r'\1', self.base_url) + car_data['url']

            car_data = { "id": f"{self.__class__.__name__}_{car_data['url'].split('/')[-1]}" } | car_data

            listings.append(car_data)

        if __name__ == "__main__":
            logger.debug("Found the following car listings:")
            for car in listings:
                logger.debug(car)

        logger.debug("Found", len(listings), "listings")

        async def run_async_code_13376ced():
            await browser.close()
            return 
         = asyncio.run(run_async_code_13376ced())
        logger.success(format_json())

        return listings

    async def crawl_listing(self, listing_url) -> List[Dict[str, str]]:
        listing_info = ""

        url = listing_url

        async def run_async_code_d66fbb88():
            async def run_async_code_22449de9():
                playwright = await async_playwright().start()
                return playwright
            playwright = asyncio.run(run_async_code_22449de9())
            logger.success(format_json(playwright))
            return playwright
        playwright = asyncio.run(run_async_code_d66fbb88())
        logger.success(format_json(playwright))

        async def run_async_code_8ecc12ef():
            browser = await playwright.chromium.launch(headless=True,
            return browser
        browser = asyncio.run(run_async_code_8ecc12ef())
        logger.success(format_json(browser))
                                                    args=[
                                                            "--no-sandbox",
                                                            "--disable-setuid-sandbox",
                                                            "--disable-dev-shm-usage",
                                                            "--disable-extensions",
                                                            "--disable-gpu"
                                                    ]
                                                    )

        async def async_func_156():
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
                viewport={"width": 1920, "height": 1080},
                locale="en-US",
                timezone_id="America/New_York",
            )
            return context
        context = asyncio.run(async_func_156())
        logger.success(format_json(context))


        logger.debug("Opening browser page")

        async def run_async_code_94ce32fc():
            async def run_async_code_7cc17402():
                page = await context.new_page()
                return page
            page = asyncio.run(run_async_code_7cc17402())
            logger.success(format_json(page))
            return page
        page = asyncio.run(run_async_code_94ce32fc())
        logger.success(format_json(page))

        async def run_async_code_c46863ef():
            await page.route("**/*", block_unnecessary_resources)
            return 
         = asyncio.run(run_async_code_c46863ef())
        logger.success(format_json())

        logger.debug("Loading page")

        async def run_async_code_93a852d7():
            await page.goto(url, wait_until="domcontentloaded")
            return 
         = asyncio.run(run_async_code_93a852d7())
        logger.success(format_json())

        logger.debug("Page partially loaded. Starting to scroll.")

        async def run_async_code_9aa622c5():
            await scroll_to_bottom(page)
            return 
         = asyncio.run(run_async_code_9aa622c5())
        logger.success(format_json())

        async def run_async_code_88e08c31():
            async def run_async_code_0d9a82db():
                page_content = await page.content()
                return page_content
            page_content = asyncio.run(run_async_code_0d9a82db())
            logger.success(format_json(page_content))
            return page_content
        page_content = asyncio.run(run_async_code_88e08c31())
        logger.success(format_json(page_content))

        tree = html.fromstring(page_content)
        listing_info = tree.xpath("//div[contains(@class, 'container') and contains(@class, 'margin-top-5')]/div[contains(@class, 'row')]//text()")
        listing_info = "\t".join(listing_info).strip()


        if __name__ == "__main__":
            logger.debug("Found the following information:")
            logger.debug(listing_info)

        async def run_async_code_13376ced():
            await browser.close()
            return 
         = asyncio.run(run_async_code_13376ced())
        logger.success(format_json())

        return listing_info

    def get_filters_info(self) -> str:
        """
        Return a prompt for the LLM describing the filters and expected output format.
        """
        return f"""
        You are a helpful assistant that translates user requirements into a URL with query parameters.

        The base URL is: {self.base_url}
        Filters:
        - zip: User's zip code (integer).
        - searchRadius: Search radius in miles (integer, e.g., 75, 100, 200).
        - startYear: Minimum year of the car (integer).
        - endYear: Maximum year of the car (integer).
        - makeCode: Car manufacturer code (string, can appear multiple times, e.g., "BMW", "FORD").
        - listingType: Type of listing (one of "NEW", "USED", "CERTIFIED", "3P_CERT").
        - mileage: Maximum mileage of the car (integer).
        - driveGroup: Type of drive (one of "AWD4WD", "FWD", "RWD").
        - extColorSimple: External color of the car (e.g., "BLACK", "WHITE", "RED", "GRAY").
        - intColorSimple: Internal color of the car (e.g., "BEIGE", "BLACK", "BLUE").
        - mpgRange: Fuel efficiency in miles per gallon (e.g., "30-MPG").
        - fuelTypeGroup: Type of fuel (one of "GSL", "DSL", "HYB", "ELE", "PIH").
        - bodyStyleSubtypeCode: Type of body style (e.g., "FULLSIZE_CREW", "COMPACT_EXTEND").
        - truckBedLength: Truck bed length (e.g., "SHORT", "EXTRA SHORT", "UNSPECIFIED").
        - vehicleStyleCode: Vehicle style (e.g., "CONVERT", "WAGON", "HATCH", "SUVCROSS").
        - dealType: Type of deal (e.g., "goodprice", "greatprice").
        - doorCode: Number of doors (e.g., "2", "3", "4").
        - engineDisplacement: Engine size range in liters (e.g., "1.0-1.9", "2.0-2.9").
        - featureCode: Specific features of the car (e.g., "1062" for heated seats, "1327" for navigation).
        - transmissionCode: Transmission type (e.g., "AUT" for automatic, "MAN" for manual).
        - vehicleHistoryType: Vehicle history (e.g., "NO_ACCIDENTS", "ONE_OWNER", "CLEAN_TITLE").
        - newSearch: Boolean to indicate a new search (e.g., "true").
        - sortBy: Sorting option for the results (optional).
            Options:
            - "relevance" (default): Sort by relevance.
            - "derivedpriceASC": Sort by price, lowest to highest.
            - "derivedpriceDESC": Sort by price, highest to lowest.
            - "distanceASC": Sort by distance, closest to farthest.
            - "datelistedASC": Sort by date, oldest first.
            - "datelistedDESC": Sort by date, newest first.
            - "mileageASC": Sort by mileage, lowest to highest.
            - "mileageDESC": Sort by mileage, highest to lowest.
            - "yearASC": Sort by year, oldest to newest.
            - "yearDESC": Sort by year, newest to oldest.

        Special filters:
        - price: Price is embedded in the path of the URL, e.g., "/cars-over-45000" or "/cars-between-10000-and-20000".

        Example Output:
        A complete URL with query parameters, e.g.,:
        "{self.base_url}/cars-between-10000-and-20000?zip=10001&startYear=2010&endYear=2020&makeCode=BMW&makeCode=FORD&listingType=USED&mileage=50000&fuelTypeGroup=GSL&intColorSimple=BLACK&vehicleHistoryType=NO_ACCIDENTS"

        Based on the user's needs, format the response as only the complete URL (no extra explanations). The URL is an example, don't include filters if they are not needed by the user.
        """

    def set_filters_from_llm_response(self, llm_response: str):
        """
        Process the LLM's response and set the URL with the provided parameters.
        """
        if llm_response.startswith(self.base_url):
            self.url = llm_response.strip()
        else:
            raise ValueError("Invalid URL format provided by LLM response: " + llm_response)

"""
### Set Up Playwright Dependencies

This cell installs the necessary dependencies for **Playwright**, a library used for browser automation, and configures the Chromium browser. It performs the following tasks:

1. **Install Playwright dependencies**:
   - Ensures that system-level dependencies required by Playwright are installed.

2. **Install Playwright browsers**:
   - Downloads and sets up the necessary browsers for Playwright to work, including Chromium.

3. **Patch Chromium**:
   - Installs any required patches for the Chromium browser to ensure compatibility with Playwright.
"""
logger.info("### Set Up Playwright Dependencies")

# !playwright install-deps
# !playwright install
# !patchright install chromium

"""
### Load Environment Variables and Set Up API Keys

This cell configures the environment and initializes API keys required for the project:

1. **Load `.env` Variables**:
   - Uses `load_dotenv()` to load sensitive information (like API keys) from a `.env` file.

2. **Configure Ollama API Key**:
#    - Retrieves the `OPENAI_API_KEY` from the environment or Colab's `userdata` if running in Google Colab.

3. **Set LangChain Configuration**:
   - Disables tracing (`LANGCHAIN_TRACING_V2`) and configures the LangChain endpoint and project.

4. **Initialize Clients**:
   - Sets up `GPT` as the model (`llama3.2`) using `ChatOllama`.
   - Creates `langsmith_client` and `openai_client` for managing Ollama interactions.
"""
logger.info("### Load Environment Variables and Set Up API Keys")

load_dotenv()

try:
#   os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY', userdata.get('OPENAI_API_KEY'))
except:
#   os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "car_buyer_agent"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY', "")

GPT = ChatOllama(model="llama3.2")

langsmith_client = Client()
openai_client = wrap_openai(openai.Client())

"""
### Define the `State` Class

This cell defines the `State` class, which inherits from `MessagesState`, to represent the current state of the car-buying process. It includes:

1. **`user_needs`**: Stores the user's requirements for the car (e.g., budget, features).
2. **`web_interfaces`**: A list of web scraper interfaces (e.g., AutoTrader) to fetch car listings.
3. **`listings`**: A collection of car listings retrieved from the web platforms.
4. **`selected_listing`**: The specific car listing chosen by the user for further exploration.
5. **`additional_info`**: Additional information about the selected car (e.g., common issues, reliability).
6. **`next_node`**: The next action or state transition in the workflow.
"""
logger.info("### Define the `State` Class")

class State(MessagesState):
    """Represents the state of the car-buying process."""
    user_needs: str
    web_interfaces: List[WebsiteInterface]
    listings: List[Dict[str, str]]
    selected_listing: Dict[str, str]
    additional_info: Dict[str, str]
    next_node: str

"""
### Define Input and Output Helper Functions

1. **`get_user_input`**:
   - A utility function to capture user input during the interaction.
   - It wraps Python’s `input()` function, allowing for optional arguments (`*args`, `**kwargs`) for flexibility in prompt customization.

2. **`show_assistant_output`**:
   - Displays the assistant's output (e.g., LLM responses) to the user.
   - Uses Python's `logger.debug()` function, enabling formatted or contextual responses.
"""
logger.info("### Define Input and Output Helper Functions")

def get_user_input(*args, **kwargs):
    """Get user input."""
    return input(*args, **kwargs)

def show_assistant_output(*args, **kwargs):
    """Show the output of the LLM."""
    logger.debug(*args, **kwargs)

"""
### Define the `ask_user_needs` Function

This function initiates the process of gathering user requirements for the car-buying assistant. It uses LLM responses to guide the conversation and determine the next steps. Key components include:

1. **State Initialization**:
   - Retrieves previous messages and existing user needs from the `state` object.

2. **Conversation Starter**:
   - Constructs a system message to ask the user about their car requirements (e.g., budget, usage, preferences).

3. **Interaction Handling**:
   - Appends the assistant's and user's messages to the conversation flow using `SystemMessage`, `AIMessage`, and `HumanMessage`.

4. **Summarization**:
   - Summarizes user input into concise points and determines the next step in the workflow:
     - `ask_user_needs`: Collect more details.
     - `build_filters`: Proceed to filtering options.
     - `irrelevant`: Handle unrelated queries.

5. **LLM Integration**:
   - Uses `USER_NEEDS_GPT` (configured with a custom response format) to process user needs and suggest the next action.

6. **Output**:
   - Displays summarized needs and the determined next step to the user.
"""
logger.info("### Define the `ask_user_needs` Function")


class NextStep(Enum):
    ASK_USER_NEEDS = "ask_user_needs"
    BUILD_FILTERS = "build_filters"
    IRRELEVANT = "irrelevant"

class UserNeeds(BaseModel):
    user_needs: str
    next_step: NextStep

USER_NEEDS_GPT = ChatOllama(model="llama3.2")

def ask_user_needs(state: State) -> State:
    """Ask user initial questions to define their needs for the car."""
    messages = state.get("messages", [])
    if len(messages) == 0:
        system_message = "You are a car buying assistant. Your goal is to help the user find a car that meets their needs. Start by introducing yourself and asking about their requirements, such as intended usage (e.g., commuting, family trips), budget, size preferences, and any specific constraints or features they value. Use their responses to guide them toward the best options."
    else:
        system_message = "Ask the user for any additional information that can help narrow down the search. If he asked any questions before, answer them before asking for more information. When answering, make sure to provide clear and concise information, with relevant examples."

    existing_needs = state.get("user_needs", "")
    if existing_needs:
        system_message += f" Here's what we know about the needs of the user so far:\n\n{existing_needs}"

    messages.append(SystemMessage(content=system_message))

    response = GPT.invoke(messages).content
    messages += [AIMessage(response)]
    show_assistant_output(f"\033[92m{messages[-1].content}\033[0m", flush=True)

    messages += [HumanMessage(get_user_input(response))]
    logger.debug(f"\033[94m{messages[-1].content}\033[0m", flush=True)

    summarization_messages = messages.copy()

    summarization_messages += [
        SystemMessage(
            "Summarize the user's car-buying needs in clear and concise bullet points based on their input and any prior knowledge.\n"
            "Provide the next step, such as asking for more details or answer questions under ask_user_needs or going forward to build_filter:\n"
            "- Use 'ask_user_needs' if you need more information or if the user asked a question.\n"
            "- Use 'build_filters' if you have enough details to search for cars online.\n"
            "If the user's query is irrelevant to the matter at hand (buying a car), respond 'irrelevant'."
        )
    ]

    response = json.loads(USER_NEEDS_GPT.invoke(summarization_messages).content)

    state["user_needs"] = response["user_needs"]

    messages += [AIMessage("I have summarized your car-buying needs as follows:\n" + state["user_needs"])]

    show_assistant_output(f"\033[92m{messages[-1].content}\033[0m")

    state["next_node"] = response["next_step"]

    logger.debug(f"\nNext node: {state['next_node']}", flush=True)

    return state

"""
### Define the `build_filters` Function

This function constructs and refines search filters based on user-provided requirements. It interacts with web scraper interfaces to tailor the search for relevant car listings. Key elements include:

1. **Initialization**:
   - Displays a message indicating the filter-building process has started.

2. **Iterating Over Web Interfaces**:
   - Loops through each scraper in `state["web_interfaces"]` to gather and apply filter options.

3. **Filter Information**:
   - Retrieves filter details from each interface using `get_filters_info()` and incorporates them with the user's needs.

4. **LLM-Assisted Filter Application**:
   - Sends the filter details and user needs to the LLM (`GPT`) for processing.
   - Parses and applies the LLM's response to the interface using `set_filters_from_llm_response()`.

5. **Error Handling**:
   - Catches and displays errors (e.g., validation issues or unexpected exceptions) for each interface, ensuring robustness.

6. **Output**:
   - Provides success or failure messages for each interface and displays the updated search URL when filters are successfully applied.
"""
logger.info("### Define the `build_filters` Function")

def build_filters(state: State) -> State:
    """Build and refine search filters based on user needs."""

    show_assistant_output("Building filters based on user needs...")

    for interface in state["web_interfaces"]:
        filters_info = interface.get_filters_info()



        system_message = SystemMessage(filters_info + "\n\n" + "User needs:\n" + state["user_needs"])

        try:
            result = GPT.invoke([system_message])
            llm_response = result.content.strip()

            interface.set_filters_from_llm_response(llm_response)
            show_assistant_output(f"\nSuccessfully set filters for: {interface.__class__.__name__}")
            show_assistant_output(f"Updated URL: {interface.url}")
        except ValueError as e:
            show_assistant_output(f"Failed to set filters for {interface.base_url}: {e}")
        except Exception as e:
            show_assistant_output(f"An error occurred while processing filters for {interface.base_url}: {e}")

    return

"""
### Define the `fetch_listings_from_sources` Function

This asynchronous function retrieves car listings from various web interfaces based on the applied filters. Key aspects include:

1. **Purpose**:
   - Simulates the process of fetching car listings, tailored to the filters defined earlier.

2. **Input**:
   - `web_interfaces`: A list of web scraper interfaces (e.g., AutoTrader) that implement the `crawl` method for data retrieval.

3. **Operation**:
   - Iterates over each interface in `web_interfaces` and asynchronously collects listings using `await interface.crawl()`.

4. **Output**:
   - Returns a consolidated list of dictionaries, where each dictionary represents a car listing
"""
logger.info("### Define the `fetch_listings_from_sources` Function")

async def fetch_listings_from_sources(web_interfaces: List[WebsiteInterface]) -> List[Dict[str, str]]:
    """Simulate retrieval of car listings from Autotrader.com based on filters.

    Args:
        filters (dict): Dictionary containing search filters (e.g., budget, fuel type).

    Returns:
        list: A list of dictionaries, each representing a car listing.
    """
    listings = []
    for interface in web_interfaces:
        async def run_async_code_f999bacd():
            async def run_async_code_c1a020c0():
                listings += await interface.crawl()
                return listings +
            listings + = asyncio.run(run_async_code_c1a020c0())
            logger.success(format_json(listings +))
            return listings +
        listings + = asyncio.run(run_async_code_f999bacd())
        logger.success(format_json(listings +))

    return listings

"""
### Define the `search_listings` Function

This function searches for car listings based on the user's needs and displays the most relevant results. It incorporates user feedback to determine the next step in the workflow.

1. **Initial Setup**:
   - Adds a system message to indicate the start of the search.
   - Calls `fetch_listings_from_sources` asynchronously to retrieve listings from the web interfaces.

2. **Listing Retrieval**:
   - Uses `asyncio.run()` to fetch listings, stores them in `state["listings"]`, and outputs the total count retrieved.

3. **Display Listings**:
   - Constructs a user-friendly list of the top 5 results, including images, titles, and other details.
   - Prompts the user to select a listing, refine the search, or end the conversation.

4. **User Interaction**:
   - Captures the user's response via the `CLASSIFIER_GPT` model, which categorizes the action (`select_listing`, `refine_search`, or `end_conversation`).
   - Updates the `state["next_node"]` based on the user's choice:
     - `select_listing`: Prepares for detailed exploration of a specific listing.
     - `refine_search`: Returns to the user needs stage.
     - `end_conversation`: Terminates the workflow.
"""
logger.info("### Define the `search_listings` Function")


class UserResponse(BaseModel):
    action: Literal['select_listing', 'refine_search', 'end_conversation']
    listing_id: Optional[str]

CLASSIFIER_GPT = ChatOllama(model="llama3.2")

def search_listings(state: State) -> State:
    """Search for cars on LaCentrale and mobile.de based on filters."""
    """Display the first listings for the user to view."""
    """Synchronous wrapper for search_listings."""

    state["messages"] += [SystemMessage("Searching for listings based on user needs, this may take time...")]
    show_assistant_output(state["messages"][-1].content)

    async def _search_listings():
        async def run_async_code_17454142():
            return await fetch_listings_from_sources(state["web_interfaces"])
            return 
         = asyncio.run(run_async_code_17454142())
        logger.success(format_json())

    listings = asyncio.run(_search_listings())
    state["listings"] = listings

    show_assistant_output(f"Successfully fetched {len(listings)} listings from the sources.")

    AI_message = ""

    AI_message += "Here are recent listings that match your requirements:\n"
    for i, listing in enumerate(state["listings"][:5], 1):
        AI_message += f"{i}.\n"
        for key, value in listing.items():
            formatted_key = key.replace("_", " ").capitalize()
            if formatted_key == "Image" and value:
                AI_message += f"   {formatted_key}: ![Example Image]({value})\n"
            else:
                AI_message += f"   {formatted_key}: {value}\n"
        AI_message += "\n"  # Add an extra line for readability

    user_prompt = "Would you like to view more details about a specific listing, or refine your search (Write END to finish this conversation) ?"
    AI_message += user_prompt

    state["messages"].append(AIMessage(AI_message))
    show_assistant_output(f"\033[92m{state['messages'][-1].content}\033[0m")
    state["messages"].append(HumanMessage(get_user_input(user_prompt)))
    logger.debug(f"\033[94m{state['messages'][-1].content}\033[0m")

    response = json.loads(CLASSIFIER_GPT.invoke(state["messages"]).content)

    if response["action"] == "select_listing":
        state["next_node"] = "fetch_additional_info"
        selected_listing_id = response["listing_id"]
        for i, listing in enumerate(state["listings"][:5], 1):
            if selected_listing_id in listing["id"]:
                state["selected_listing"] = listing
                break
    elif response["action"] == "refine_search":
        state["next_node"] = "ask_user_needs"
    else:
        state["next_node"] = END

    return state

"""
### Define the `fetch_additional_info` Function

This function retrieves detailed information about a selected car listing, enhances it with insights from the web, and allows the user to decide the next steps.

1. **Crawl the Car Listing**:
   - Asynchronously fetches additional details about the selected car from its source URL using the appropriate scraper (`crawl_listing`).

2. **Summarize Car Details**:
   - Uses the LLM (`GPT`) to generate a clear and concise summary of the car's details, formatted for readability.

3. **Fetch Web-Based Insights**:
   - Queries DuckDuckGo for information about the car model (e.g., common issues, reliability) and formats the results.

4. **Enhance Context with LLM**:
   - Combines the fetched insights with user needs to generate a comprehensive summary of the car's specifications and general issues.

5. **User Interaction**:
   - Displays the additional information to the user and prompts them to either:
     - View details of another listing (`fetch_additional_info`).
     - Refine their search (`ask_user_needs`).
     - End the conversation.

6. **Workflow Updates**:
   - Updates `state["next_node"]` based on the user's action, ensuring a smooth transition to the next step.
"""
logger.info("### Define the `fetch_additional_info` Function")


duckduckgo_search = DuckDuckGoSearchResults(max_results=3)

def fetch_additional_info(state: State) -> State:
    """Fetch more details about the selected car listing."""
    listing = state["selected_listing"]


    async def _crawl_car_listing():
        for interface in state["web_interfaces"]:
            if listing["id"].split("_")[0].lower() in interface.__class__.__name__.lower():
                async def run_async_code_91d27dca():
                    return await interface.crawl_listing(listing["url"])
                    return 
                 = asyncio.run(run_async_code_91d27dca())
                logger.success(format_json())

    info_car_for_sale = asyncio.run(_crawl_car_listing())

    prompt = SystemMessage(
        f"Summarize all the relevant information about the selected car for sale into a paragraph: {listing['title']}\n\n"
        f"Here is the raw information about the car for sale:\n\n{info_car_for_sale}"
        f"Format the summary clearly and concisely, with line breaks between sections."
    )

    car_info_summary = GPT.invoke([prompt]).content

    show_assistant_output("\033[92mHere are more details about the car for sale:\n\033[0m", flush=True)

    show_assistant_output("\033[92m" + car_info_summary + "\n\n\033[0m", flush=True)

    state["messages"] += [prompt, AIMessage(car_info_summary)]

    car_name = listing["title"]

    queries = [f"{car_name} common issues", f"{car_name} problem", f"{car_name} reliability"]
    context = ""
    for query in queries:
        search_results = duckduckgo_search.invoke(query)
        formatted_results = f"QUERY: {query}\n\n{search_results}\n-------------------\n"
        context += formatted_results

    prompt = SystemMessage(
        f"Provide additional information about this car: {listing['title']}, "
        f"including engine specifications, common issues with this model, and market value."
        f"Here is additioanl context to help you provide the information:\n\n{context}"
        f"Here are the user needs, give some insights about the car based on the user needs:\n\n{state['user_needs']}"
    )

    result = GPT.invoke([prompt])

    listing["additional_info"] = result.content

    show_assistant_output(f"\033[92mHere is additional information about the model in general, coming from Internet:\n{listing['additional_info']}\n\033[0m")

    user_prompt = "Would you like to view more details about another listing, or refine your search (Write END to finish this conversation) ?"
    state["messages"] += [SystemMessage(user_prompt)]
    state["messages"] += [HumanMessage(get_user_input(user_prompt))]
    logger.debug(f"\033[94m{state['messages'][-1].content}\033[0m", flush=True)

    response = json.loads(CLASSIFIER_GPT.invoke(state["messages"]).content)

    if response["action"] == "select_listing":
        state["next_node"] = "fetch_additional_info"
        selected_listing_id = response["listing_id"]
        for i, listing in enumerate(state["listings"][:5], 1):
            if selected_listing_id in listing["id"]:
                state["selected_listing"] = listing
                break
    elif response["action"] == "refine_search":
        state["next_node"] = "ask_user_needs"
    else:
        state["next_node"] = END

    return state

"""
### Initialize and Define the Workflow Graph

This cell sets up the state-based workflow using `StateGraph` from `langgraph`. It defines the nodes, edges, and conditional logic for navigating through the car-buying assistant's process.

1. **Initialize the Workflow**:
   - The `StateGraph` object is created with the `State` class to manage the workflow state.

2. **Define Nodes**:
   - Each step in the workflow is represented as a node:
     - `ask_user_needs`: Gathers user requirements.
     - `build_filters`: Constructs search filters.
     - `search_listings`: Retrieves car listings.
     - `fetch_additional_info`: Provides detailed information about a selected car.
     - `irrelevant`: Handles unrelated queries.

3. **Set Workflow Edges**:
   - Nodes are connected based on the possible transitions:
     - Conditional transitions from `ask_user_needs` depend on the next step (`build_filters`, `ask_user_needs`, or `irrelevant`).
     - `build_filters` transitions directly to `search_listings`.
     - Conditional transitions from `search_listings` determine whether to fetch more details, return to user needs, or end the workflow.
     - The `irrelevant` node ends the workflow.

4. **Entry and Exit Points**:
   - The workflow begins at the `ask_user_needs` node.
   - Conditional edges from `fetch_additional_info` allow for revisiting user needs, exploring more details, or ending the workflow.

5. **Compile the Workflow**:
   - The workflow is compiled into an executable application (`app`), ready to process user queries.
"""
logger.info("### Initialize and Define the Workflow Graph")

workflow = StateGraph(State)

workflow.add_node("ask_user_needs", ask_user_needs)
workflow.add_node("build_filters", build_filters)
workflow.add_node("search_listings", search_listings)
workflow.add_node("fetch_additional_info", fetch_additional_info)
workflow.add_node("irrelevant", lambda state: state)

workflow.add_conditional_edges("ask_user_needs", lambda state: state["next_node"], ["build_filters", "ask_user_needs", "irrelevant"])
workflow.add_edge("build_filters", "search_listings")
workflow.add_conditional_edges("search_listings", lambda state: state["next_node"], ["fetch_additional_info", "ask_user_needs", END])
workflow.add_edge("irrelevant", END)

workflow.set_entry_point("ask_user_needs")
workflow.add_conditional_edges("fetch_additional_info", lambda state: state["next_node"], ["ask_user_needs", "fetch_additional_info", END])


app = workflow.compile()

"""
### Visualize the Workflow Graph

This cell generates and displays a visual representation of the workflow using Mermaid.js. The graph shows the nodes and their connections, providing a clear overview of the assistant's structure.

1. **Generate Graph**:
   - The `draw_mermaid_png` method from `MermaidDrawMethod.API` creates a PNG image of the workflow graph.

2. **Display Graph**:
   - The `Image` function renders the generated PNG, enabling visualization of the nodes (states) and edges (transitions).
"""
logger.info("### Visualize the Workflow Graph")

display(
    Image(
        app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
)

"""
### Define `run_car_buyer_agent` Function

This function initializes and executes the car-buying assistant workflow using the compiled LangGraph application.

1. **Initialization**:
   - Creates an empty `messages` list to store the conversation history.

2. **Set Initial State**:
   - Defines the `initial_state` object with the following:
     - `user_needs`: Empty, awaiting user input.
     - `web_interfaces`: A list containing the `AutotraderInterface` for scraping car listings.
     - `listings`, `selected_listing`, `additional_info`: Empty placeholders to be populated during the workflow.
     - `next_node`: Empty, to be updated dynamically.
     - `messages`: Tracks messages exchanged between the assistant and the user.

3. **Run Workflow**:
   - Invokes the workflow using `app.invoke()` with the initialized state.

4. **Output**:
   - Returns the final `result` after executing the workflow.
"""
logger.info("### Define `run_car_buyer_agent` Function")

def run_car_buyer_agent():
    """Run the car-buying assistant with LangGraph."""

    messages = []

    initial_state = State(
        user_needs={},
        web_interfaces=[AutotraderInterface()],
        listings=[],
        selected_listing={},
        additional_info={},
        next_node="",
        messages=messages
    )
    result = app.invoke(initial_state)
    return result

"""
### Conditional Execution Without Gradio

This cell provides a fallback mechanism to run the car-buying assistant in a command-line environment if the `USE_GRADIO` flag is set to `False`.

1. **Define Input/Output Functions**:
   - `get_user_input`: Captures user input via the terminal using `input()`.
   - `show_assistant_output`: Displays the assistant's output using `logger.debug()`.

2. **Execute the Agent**:
   - Calls the `run_car_buyer_agent()` function to initiate the workflow and stores the result in `car_buyer_result`.

3. **Debugging Output**:
   - Prints the raw result of the workflow execution for inspection.

4. **Display Final Recommendation**:
   - If a car listing is selected:
     - Outputs the title, price, and mileage of the recommended car.
     - Prints additional details retrieved during the workflow.
   - If no listing is selected, informs the user.
"""
logger.info("### Conditional Execution Without Gradio")

USE_GRADIO = True

if not USE_GRADIO:
    def get_user_input(*args, **kwargs):
        """Get user input."""
        return input(*args, **kwargs)

    def show_assistant_output(*args, **kwargs):
        """Show the output of the LLM."""
        logger.debug(*args, **kwargs)

    car_buyer_result = run_car_buyer_agent()

    logger.debug("Car Buyer Result:", car_buyer_result)

    if "selected_listing" in car_buyer_result:
        listing = car_buyer_result["selected_listing"]
        logger.debug(f"\nFinal Recommendation:\n{listing['title']} - {listing['price']} - {listing['mileage']} km")
        logger.debug("Additional Information:")
        for key, value in car_buyer_result["additional_info"].items():
            logger.debug(f"{key}: {value}")
    else:
        logger.debug("No car listing selected.")

"""
### Gradio Interface and Threaded Execution

This cell sets up a **Gradio-based chatbot interface** for the car-buying assistant, enabling user interaction via a web-based GUI.

#### Key Components:

1. **Input and Output Queues**:
   - `InputQueue`:
     - Simulates `stdin` behavior by queuing user inputs for asynchronous processing.
   - `output_queue`:
     - Stores the assistant's responses for incremental display in the UI.

2. **Input/Output Functions**:
   - `get_user_input`:
     - Waits for and retrieves user input from the `input_queue`.
   - `show_assistant_output`:
     - Formats and sends assistant responses to the `output_queue`.

3. **Agent Interaction**:
   - `interact_with_agent`:
     - Processes user messages, sends them to the agent, and retrieves responses incrementally for a seamless conversational flow.
   - `get_initial_message`:
     - Captures the initial response from the agent to display upon starting the interface.

4. **Threaded Execution**:
   - `run_langgraph_agent`:
     - Executes the LangGraph-based workflow in a separate thread to allow non-blocking interaction in the Gradio interface.

5. **Gradio Interface**:
   - `gr.ChatInterface`:
     - Creates a real-time chat interface with the following configurations:
       - `interact_with_agent`: Handles message exchanges.
       - `chatbot`: Configures the chat window's appearance and behavior.

6. **Execution Flow**:
   - If `USE_GRADIO` is `True`:
     - Starts the agent in a separate thread.
     - Initializes the Gradio interface with the assistant's initial message and launches it.
"""
logger.info("### Gradio Interface and Threaded Execution")


waiting_for_input = False # Flag to indicate if the agent is waiting for user input

class InputQueue:
    """A custom input queue that mimics stdin behavior."""
    def __init__(self):
        self.queue = queue.Queue()

    def readline(self):
        """Mimic the readline behavior of stdin."""
        try:
            r = self.queue.get(block=True)  # Wait until input is available
            return r
        except queue.Empty:
            return ""

    def write(self, message):
        """Handle writes if needed (for debugging)."""
        pass

    def flush(self):
        """No-op for compatibility."""
        pass

    def put(self, message):
        """Put a message into the queue."""
        self.queue.put(message)

output_queue = queue.Queue()
input_queue = queue.Queue()

def get_user_input(*args, **kwargs):
    """Get user input."""
    global waiting_for_input
    logger.debug("Waiting for user input...")
    waiting_for_input = True
    r = input_queue.get()
    waiting_for_input = False
    logger.debug(f"Received user input")
    return r

def show_assistant_output(*args, **kwargs):
    """Show the output of the LLM."""

    result = " ".join(args) + kwargs.get("end", "\n")

    result = re.sub(r'\033\[\d+m', '', result)

    output_queue.put(result)

def interact_with_agent(user_message, history, discard_user_input=False):
    """Send user message to the bot and handle the response."""

    global waiting_for_input

    if not discard_user_input:
        input_queue.put(user_message + "\n")  # Send user input to LangGraph

    partial_message = ""

    while True:
        try:
            message = output_queue.get(timeout=0.1)  # Wait for bot output
            if message:

                partial_message += message
                yield partial_message
        except queue.Empty:
            is_end = waiting_for_input
            if is_end:
                break
            time.sleep(0.1)

def get_initial_message():
    """Run the agent and capture the initial message."""
    initial_message = ""

    for message in interact_with_agent("", [], True):  # Consume the generator to get the full initial message
        initial_message = message  # Keep updating until the generator finishes

    return initial_message

initial_message_content = ""

def run_langgraph_agent():
    """Run the LangGraph agent and redirect its stdout."""
    global initial_message_content
    run_car_buyer_agent()  # Start the LangGraph workflow


if USE_GRADIO:
    agent_thread = threading.Thread(target=run_langgraph_agent, daemon=True)
    agent_thread.start()

    initial_message_content = get_initial_message()

    initial_messages = [{"role": "assistant", "content": initial_message_content}]

    chat = gr.ChatInterface(interact_with_agent,
                    chatbot=gr.Chatbot(label="Car Buyer Chatbot", autoscroll=True, scale=1, value=initial_messages, type="messages", height=200),
                    type="messages").launch()

logger.info("\n\n[DONE]", bright=True)