from IPython.display import Markdown
from dotenv import load_dotenv
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.file.utils import save_file
from jet.logger import logger
from langchain.schema import HumanMessage
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_tavily import TavilyCrawl
from langchain_tavily import TavilyExtract
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient
import datetime
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
![](https://europe-west1-atp-views-tracker.cloudfunctions.net/working-analytics?notebook=tutorials--agent-with-tavily-web-access--hybrid-agent-tutorial)

## 3. Combine Internal Data with Web Data 🌐🤝📊

Welcome to tutorial 3!


In this tutorial, you'll discover how to build a **hybrid agent** that combines real-time web data with your own internal knowledge sources. For enterprises, the true power of agents comes from integrating proprietary internal data with information from the public web.

By the end, you'll be able to:

- **Integrate** web search results and private documents into a single agent workflow
- **Enable** your agent to compare, contrast, and synthesize information from both public and internal sources
- **Build** more accurate, up-to-date, and context-aware applications for research, analysis, and decision-making


> **Why is this approach valuable?**  
> It empowers your AI to answer questions with the most current information available online, while also leveraging your proprietary data—just like a human expert would.  
>  
> You'll unlock new capabilities for market research, competitive intelligence, and any scenario where both internal and external knowledge matter.

---

## Getting Started

Follow these steps to set up:

1. **Sign up** for Tavily at [app.tavily.com](https://app.tavily.com/home/?utm_source=github&utm_medium=referral&utm_campaign=nir_diamant) to get your API key.

   *Refer to the screenshots linked below for step-by-step guidance:*

   - [Screenshot: Signup Page](assets/sign-up.png)
   - [Screenshot: Tavily API Keys Dashboard](assets/api-key.png)


2. **Sign up** for Ollama to get your API key. By default, we’ll use Ollama—but you can substitute any other LLM provider.
   

2. **Copy your API keys** from your Tavily and Ollama account dashboard.

3. **Paste your API keys** into the cell below and execute the cell.
"""
logger.info("## 3. Combine Internal Data with Web Data 🌐🤝📊")

# !echo "TAVILY_API_KEY=<your-tavily-api-key>" >> .env
# !echo "OPENAI_API_KEY=<your-ollama-api-key>" >> .env

"""
Install dependencies in the cell below.
"""
logger.info("Install dependencies in the cell below.")

# %pip install -U tavily-python langchain-chroma langchain-ollama langgraph langchain-tavily --quiet

"""
### Setting Up Your Tavily API Client

The code below will instantiate the Tavily client with your API key.
"""
logger.info("### Setting Up Your Tavily API Client")

# import getpass

load_dotenv()

# if not os.environ.get("TAVILY_API_KEY"):
#     os.environ["TAVILY_API_KEY"] = getpass.getpass("TAVILY_API_KEY:\n")

# if not os.environ.get("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for Ollama: ")

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

"""
## Vector Database Set Up

In this tutorial, we’ll use a pre-built vector database to simulate an internal knowledge store. The data is based on synthetically generated mock CRM documents, designed to support a CRM and sales insights use case.

To keep things simple, we've prepared a vector store using Chroma and Ollama embeddings, built from the mock CRM data. Feel free to explore the underlying documents in the [supplemental/docs](supplemental/docs) directory.

To get started, simply initialize the vector index by running the Python cell below.
"""
logger.info("## Vector Database Set Up")


embeddings = OllamaEmbeddings(model="mxbai-embed-large")

vector_store = Chroma(
    collection_name="crm",  # replace with "my_custom_index" for custom documents option
    embedding_function=embeddings,
    persist_directory="supplemental/db",
)

"""
However, if you'd prefer to integrate your own documents, we've made it easy to vectorize your files by following three steps:
1. Upload your files (must be PDF) to the [supplemental/docs](supplemental/docs) directory.

2. Follow the [`vectorize_tutorial.ipynb`](supplemental/vectorize_tutorial.ipynb) notebook to create your custom vector index.

3. Replace `collection_name="crm"` with `collection_name="my_custom_index"` in the Chroma instantiation code in the cell above and re-run it.

Note: the vector database pipeline we created is just for demonstration and not optimized — you’re encouraged to plug in your own optimized vector pipeline, vector store, and embeddings model to suit your data and needs.

Let's test out our vector index with a simple query.
"""
logger.info("However, if you'd prefer to integrate your own documents, we've made it easy to vectorize your files by following three steps:")

retriever = vector_store.as_retriever()

search_results = retriever.invoke("robotics use case")

"""
Let's view the results of the vector search.
"""
logger.info("Let's view the results of the vector search.")

for result in search_results:
    logger.debug(result.page_content)
    logger.debug("\n")

"""
Next, we'll set up our tools: both the web tools introduced in Tutorial 2 and our new vector search tool.
"""
logger.info("Next, we'll set up our tools: both the web tools introduced in Tutorial 2 and our new vector search tool.")


vector_search_tool = retriever.as_tool(
    name="vector_search",
    description="""
    Perform a vector search on our company's CRM data.
    """,
)

search = TavilySearch(max_results=10, topic="general")

extract = TavilyExtract(extract_depth="advanced")

crawl = TavilyCrawl()

"""
We'll set up several Ollama foundation models, such as o3-mini and the gpt-4.1 model. If you prefer a different LLM provider, you can easily plug in any LangChain Chat model.
"""
logger.info("We'll set up several Ollama foundation models, such as o3-mini and the gpt-4.1 model. If you prefer a different LLM provider, you can easily plug in any LangChain Chat model.")


# o3_mini = ChatOllama(model="llama3.2")

# gpt_4_1 = ChatOllama(model="llama3.2")
model_client = ChatOllama(model="qwen3:4b-q4_K_M")

"""
## Hybrid Agent

We'll now define our hybrid react agent using LangGraph, illustrated in the diagram below. 

<img src="./assets/hybrid.svg" alt="Agent" width="500"/>

The system prompt is a crucial component in guiding the agent's behavior, as it sets the context, boundaries, and instructions for how the agent should interact with available tools and data sources. A well-crafted system prompt ensures the agent understands what information it can access, how to use each tool effectively, and what standards to follow when generating responses.

The system prompt below explicitly outlines the agent's capabilities, including access to both public web tools (search, extract, crawl) and an internal vector search tool for proprietary CRM data. It details the purpose and best practices for each tool, emphasizes the importance of grounding answers in retrieved information, and prohibits the agent from making up facts or relying on prior knowledge. This comprehensive prompt helps the agent make informed decisions about when and how to use each tool, resulting in more accurate and reliable outputs. 

The prompt clearly describes what information is available in the vector store, so the agent knows how to 
use it alongside the web tools. Feel free to experiment with the system prompt and try different language models to adjust the 
agent's behavior and optimize results for your needs.
"""
logger.info("## Hybrid Agent")


today = datetime.datetime.today().strftime("%A, %B %d, %Y")

hybrid_agent = create_react_agent(
    model=model_client,
    tools=[search, crawl, extract, vector_search_tool],
    prompt=ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
        You are a ReAct-style research agent equipped with the following tools:

        * **Tavily Web Search**
        * **Tavily Web Extract**
        * **Tavily Web Crawl**
        * **Internal Vector Search** (for proprietary CRM data)

        Your mission is to conduct comprehensive, accurate, and up-to-date research, using a combination of real-time public web information and internal enterprise knowledge.  All answers must be grounded in retrieved information from the tools provided. You are not permitted to make up information or rely on prior knowledge.

        **Today's Date:** {today}

        **Available Tools:**

        1. **Tavily Web Search**

        * **Purpose:** Retrieve relevant web pages based on a query.
        * **Usage:** Provide a search query to receive up to 10 semantically ranked results, each containing the title, URL, and a content snippet.
        * **Best Practices:**
            * Use specific queries to narrow down results.
                * Example: "Ollama's latest product release" and "Ollama's headquarters location" rather than "Ollama's latest product release and headquarters location"
            * Optimize searches using parameters such as `search_depth`, `time_range`, `include_domains`, and `include_raw_content`.
            * Break down complex queries into specific, focused sub-queries.

        2. **Tavily Web Crawl**

        * **Purpose:** Explore a website's structure and gather content from linked pages for deep research and information discovery from a single source.
        * **Usage:** Input a base URL to crawl, specifying parameters such as `max_depth`, `max_breadth`, and `extract_depth`.
        * **Best Practices:**

            * Begin with shallow crawls and progressively increase depth.
            * Utilize `select_paths` or `exclude_paths` to focus the crawl.
            * Set `extract_depth` to "advanced" for comprehensive extraction.

        3. **Tavily Web Extract**

        * **Purpose:** Extract the full content from specific web pages.
        * **Usage:** Provide URLs to retrieve detailed content.
        * **Best Practices:**

            * Set `extract_depth` to "advanced" for detailed content, including tables and embedded media.
            * Enable `include_images` if image data is necessary.

        4. **Internal Vector Search**

        * **Purpose:** Search the proprietary CRM knowledge base.
        * **Usage:** Submit a natural language query to retrieve relevant context from the CRM vector store. Contains context about key accounts like Meta, Apple, Google, Amazon, Microsoft, and Tesla.
        * **Best Practices:**
            * When possible, refer to specific information, such as names, dates, product usage, or meetings.


        **Guidelines for Conducting Research:**

        * You may not answer questions based on prior knowledge or assumptions.
        * **Citations:** Always support findings with source URLs, clearly provided as in-text citations.
        * **Accuracy:** Rely solely on data obtained via provided tools (web or vector store) —never fabricate information.
        * **If none of the tools return useful information, respond:**
            * "I'm sorry, but none of the available tools provided sufficient information to answer this question."

        **Research Workflow:**

        * **Thought:** Consider necessary information and next steps.
        * **Action:** Select and execute appropriate tools.
        * **Observation:** Analyze obtained results.
        **IMPORTANT: Repeat Thought/Action/Observation cycles as needed. You MUST only respond to the user once you have gathered all the information you need.**
        * **Final Answer:** Synthesize and present findings with citations in markdown format.

        **Example Workflow:**

        **Workflow 4: Combine Web Tools and Vector Search**

        **Question:** What has Apple publicly shared about their AI strategy, and do we have any internal notes on recent meetings with them?

        * **Thought:** I’ll start by looking for Apple’s public statements on AI using a search.
        * **Action:** Tavily Web Search with the query "Apple AI strategy 2024" and `time_range` set to "month".
        * **Observation:** Found a recent article from Apple’s newsroom and a keynote transcript.
        * **Thought:** I want to read the full content of Apple’s keynote.
        * **Action:** Tavily Web Extract on the URL from the search result.
        * **Observation:** Extracted detailed content from Apple’s announcement.
        * **Thought:** Now, I’ll check our internal CRM data for recent meeting notes with Apple.
        * **Action:** Internal Vector Search with the query "recent Apple meetings AI strategy".
        * **Observation:** Retrieved notes from a Q2 strategy sync with Apple’s enterprise team.
        * **Final Answer:** Synthesized response combining public and internal data, with citations.
        ---

        You will now receive a research question from the user:
        """,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    ),
    name="hybrid_agent",
)


inputs = {
    "messages": [
        HumanMessage(
            content="Search for the latest news on google relevant to our current CRM data on them"
        )
    ]
}

for s in hybrid_agent.stream(inputs, stream_mode="values"):
    message = s["messages"][-1]
    if isinstance(message, tuple):
        logger.debug(message)
    else:
        logger.success(message)

"""
Check out the intermediary agent traces above to see how the agent combined the vector search and web search together.

Let's view the final output.
"""
logger.info("Check out the intermediary agent traces above to see how the agent combined the vector search and web search together.")

# Markdown(message.content)
prompt_response_md1 = f"""\
## Prompt

{inputs["messages"][-1].content}

## Response

{message.content}
"""
save_file(prompt_response_md1, f"{OUTPUT_DIR}/agent_chat_1.md")

"""
As shown, the final report integrates relevant web insights with our existing CRM data on Google Cloud ML Ops.

Let's try another example. We'll ask the agent: "Check for the Google deal size and find the latest earnings report for them to determine if they are currently on a spending spree."
"""
logger.info("As shown, the final report integrates relevant web insights with our existing CRM data on Google Cloud ML Ops.")

inputs = {
    "messages": [
        HumanMessage(
            content="Check for the Google Deal size and find the latest earnings report for them to validate if they are in spending spree"
        )
    ]
}

for s in hybrid_agent.stream(inputs, stream_mode="values"):
    message = s["messages"][-1]
    if isinstance(message, tuple):
        logger.debug(message)
    else:
        logger.success(message)

"""
Let's view the final output.
"""
logger.info("Let's view the final output.")

# Markdown(message.content)
prompt_response_md1 = f"""\
## Prompt

{inputs["messages"][-1].content}

## Response

{message.content}
"""
save_file(prompt_response_md1, f"{OUTPUT_DIR}/agent_chat_2.md")

"""
# Conclusion
 
Congratulations on completing this tutorial—and the entire series—on building intelligent agents with web access using the Tavily API!
 
Throughout these tutorials, you've learned how to:
 
- **Set up the Tavily API** for real-time web search, extraction, and crawling.
- **Design agent architectures** that combine both web and internal knowledge sources.
- **Integrate multiple tools**—including web search, extract, crawling, and vector search—into a unified, powerful agent system.
- **Apply these agents** to real-world scenarios, such as researching companies and analyzing news.
 
By leveraging Tavily's web API, your agents can now access up-to-date information beyond their training data, making them more effective and relevant for dynamic business.
 
Well done on reaching the end of this series! We hope these skills empower you to build even more advanced and insightful AI solutions. Happy coding!
"""
logger.info("# Conclusion")

logger.info("\n\n[DONE]", bright=True)