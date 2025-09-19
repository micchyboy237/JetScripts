from IPython.display import Markdown
from dotenv import load_dotenv
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.file.utils import save_file
from jet.logger import CustomLogger
from langchain.schema import HumanMessage
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
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
![](https://europe-west1-atp-views-tracker.cloudfunctions.net/working-analytics?notebook=tutorials--agent-with-tavily-web-access--web-agent-tutorial)

# 2. Build a Web Research Agent 🧙 

Welcome to tutorial 2!

In the first exercise, you learned the basics of the 
Tavily API. Now, let's take it a step further: we'll 
combine that knowledge with LLMs to unlock 
real value. In this tutorial, you'll learn how to build a web research agent that can search, extract, crawl, and reason over live web data.

You already explored the various parameter 
configurations available for each Tavily API endpoint. 
With the official [Tavily-LangChain integration](https://www.tavily.com/integrations/langchain) integration, our agent can 
automatically set these parameters—like the 
`time_range` for search or specific crawl 
`instructions`—based on the context and requirements of 
each task. This dynamic, LLM-powered configuration is 
powerful in agentic systems.

By the end of this lesson, you'll know how to:
- Seamlessly connect foundation models to the web for up-to-date research
- Build a react-style web agent 
- Dynamically configure search, extract, and crawl parameters the Tavily-LangChain integration.



---

## Getting Started

Follow these steps to set up:

1. **Sign up** for Tavily at [app.tavily.com](https://app.tavily.com/home/?utm_source=github&utm_medium=referral&utm_campaign=nir_diamant) to get your API key.

   *Refer to the screenshots linked below for step-by-step guidance:*

   - ![Screenshot: Signup Page](assets/sign-up.png)
   - ![Screenshot: Tavily API Keys Dashboard](assets/api-key.png)


2. **Sign up** for MLX to get your API key. By default, we’ll use MLX—but you can substitute any other LLM provider.
   

2. **Copy your API keys** from your Tavily and MLX account dashboard.

3. **Paste your API keys** into the cell below and execute the cell.
"""
logger.info("# 2. Build a Web Research Agent 🧙")

# !echo "TAVILY_API_KEY=<your-tavily-api-key>" >> .env
# !echo "OPENAI_API_KEY=<your-openai-api-key>" >> .env

"""
Install dependencies in the cell below.
"""
logger.info("Install dependencies in the cell below.")

# %pip install -U tavily-python langchain-openai langchain langchain-tavily langgraph --quiet

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
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for MLX: ")

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

"""
Let's define the following modular tools with the Tavily-LangChain integration:
1. **Search** the web for relevant information

2. **Extract** content from specific web pages

3. **Crawl** entire websites
"""
logger.info(
    "Let's define the following modular tools with the Tavily-LangChain integration:")


search = TavilySearch(max_results=10, topic="general")

extract = TavilyExtract(extract_depth="advanced")

crawl = TavilyCrawl()

"""
Now let's set up several MLX foundation models to power our agent, such as o3-mini and the gpt-4.1 model. If you prefer a different LLM provider, you can easily plug in any LangChain Chat Model.
"""
logger.info("Now let's set up several MLX foundation models to power our agent, such as o3-mini and the gpt-4.1 model. If you prefer a different LLM provider, you can easily plug in any LangChain Chat Model.")


# o3_mini = ChatMLX(model="o3-mini-2025-01-31", api_key=os.getenv("OPENAI_API_KEY"))

# gpt_4_1 = ChatMLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats", api_key=os.getenv("OPENAI_API_KEY"))
model_client = ChatOllama(model="qwen3:4b-q4_K_M")

"""
## Web Agent

Next, we'll build a Web Agent powered by Tavily, which consists of three main components: the language model, a set of web tools, and a system prompt. The language model (such as o3-mini or gpt-4.1) serves as the agent's "brain," while the web tools (Search, Extract, and Crawl) allow the agent to interact with and gather information from the internet. The system prompt guides the agent's behavior, explaining how and when to use each tool to accomplish its research goals.

This agent leverages a pre-built LangGraph reAct implementation, as illustrated in the diagram below. The reAct framework enables the agent to reason about which actions to take, use the web tools in sequence, and iterate as needed until it completes its research task. The system prompt is especially important—it instructs the agent on best practices for using the tools together, ensuring that the agent's responses are thorough, accurate, and well-sourced.

You are encouraged to experiment with the system prompt or try different language models (like swapping between gpt-4.1 and o3-mini) to change the agent's style, personality, or optimize its performance for specific use cases.

<img src="./assets/web-agent.svg" alt="Agent" width="500"/>
"""
logger.info("## Web Agent")


today = datetime.datetime.today().strftime("%A, %B %d, %Y")

web_agent = create_react_agent(
    model=model_client,
    tools=[search, extract, crawl],
    prompt=ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
        You are a research agent equipped with advanced web tools: Tavily Web Search, Web Crawl, and Web Extract. Your mission is to conduct comprehensive, accurate, and up-to-date research, grounding your findings in credible web sources.

        **Today's Date:** {today}

        **Available Tools:**

        1. **Tavily Web Search**

        * **Purpose:** Retrieve relevant web pages based on a query.
        * **Usage:** Provide a search query to receive semantically ranked results, each containing the title, URL, and a content snippet.
        * **Best Practices:**

            * Use specific queries to narrow down results.
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

        **Guidelines for Conducting Research:**

        * **Citations:** Always support findings with source URLs, clearly provided as in-text citations.
        * **Accuracy:** Rely solely on data obtained via provided tools—never fabricate information.
        * **Methodology:** Follow a structured approach:

        * **Thought:** Consider necessary information and next steps.
        * **Action:** Select and execute appropriate tools.
        * **Observation:** Analyze obtained results.
        * Repeat Thought/Action/Observation cycles as needed.
        * **Final Answer:** Synthesize and present findings with citations in markdown format.

        **Example Workflows:**

        **Workflow 1: Search Only**

        **Question:** What are recent news headlines about artificial intelligence?

        * **Thought:** I need quick, recent articles about AI.
        * **Action:** Use Tavily Web Search with the query "recent artificial intelligence news" and set `time_range` to "week".
        * **Observation:** Retrieved 10 relevant articles from reputable news sources.
        * **Final Answer:** Summarize key headlines with citations.

        **Workflow 2: Search and Extract**

        **Question:** Provide detailed insights into recent advancements in quantum computing.

        * **Thought:** I should find recent detailed articles first.
        * **Action:** Use Tavily Web Search with the query "recent advancements in quantum computing" and set `time_range` to "month".
        * **Observation:** Retrieved 10 relevant results.
        * **Thought:** I should extract content from the most comprehensive article.
        * **Action:** Use Tavily Web Extract on the most relevant URL from search results.
        * **Observation:** Extracted detailed information about quantum computing advancements.
        * **Final Answer:** Provide detailed insights summarized from extracted content with citations.

        **Workflow 3: Search and Crawl**

        **Question:** What are the latest advancements in renewable energy technologies?

        * **Thought:** I need recent articles about advancements in renewable energy.
        * **Action:** Use Tavily Web Search with the query "latest advancements in renewable energy technologies" and set `time_range` to "month".
        * **Observation:** Retrieved 10 articles discussing recent developments in solar panels, wind turbines, and energy storage.
        * **Thought:** To gain deeper insights, I'll crawl a relevant industry-leading renewable energy site.
        * **Action:** Use Tavily Web Crawl on the URL of a leading renewable energy industry website, setting `max_depth` to 2.
        * **Observation:** Gathered extensive content from multiple articles linked on the site, highlighting new technologies and innovations.
        * **Final Answer:** Provide a synthesized summary of findings with citations.

        ---

        You will now receive a research question from the user:

        """,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    ),
    name="web_agent",
)

"""
### Test Your Tavily Web Agent

Now we'll run the agent and see how it uses the different web tools.
"""
logger.info("### Test Your Tavily Web Agent")


inputs = {
    "messages": [
        HumanMessage(
            content="find all the iphone models currently available on apple.com and their prices"
        )
    ]
}

for s in web_agent.stream(inputs, stream_mode="values"):
    message = s["messages"][-1]
    if isinstance(message, tuple):
        logger.debug(message)
    else:
        logger.success(message)

"""
Examine the agent's intermediate steps printed above, including how it chooses and configures different tool parameters. Then, display the agent's final answer in markdown format.
"""
logger.info("Examine the agent's intermediate steps printed above, including how it chooses and configures different tool parameters. Then, display the agent's final answer in markdown format.")

# Markdown(message.content)
prompt_response_md1 = f"""\
## Prompt

{inputs["messages"][-1].content}

## Response

{message.content}
"""
save_file(prompt_response_md1, f"{OUTPUT_DIR}/agent_chat_1.md")

"""
Let's try a different example.
"""
logger.info("Let's try a different example.")

inputs = {
    "messages": [
        HumanMessage(
            content="return 5 job postings for a software engineer in the bay area on linkedin"
        )
    ]
}

for s in web_agent.stream(inputs, stream_mode="values"):
    message = s["messages"][-1]
    if isinstance(message, tuple):
        logger.debug(message)
    else:
        logger.success(message)

"""
Print the agent's final response as markdown.
"""
logger.info("Print the agent's final response as markdown.")

prompt_response_md2 = f"""\
## Prompt

{inputs["messages"][-1].content}

## Response

{message.content}
"""

# Markdown(message.content)
save_file(prompt_response_md2, f"{OUTPUT_DIR}/agent_chat_2.md")

"""
Notice how the agent cleverly combines Tavily’s tools—search, crawl, and extract—to complete the task end-to-end.

## Conclusion & Next Steps
 
In this tutorial, you learned how to:
- Set up Tavily web tools (search, extract, crawl) with LangChain
- Build an intelligent web research agent using LangGraph's `create_react_agent`
- Design effective system prompts for autonomous web research
- Test your agent with real-world tasks like product research and job searching
 
You now have a fully functional web research agent that autonomously combines search, extraction, and crawling to complete complex research objectives.
 
**Ready for more advanced capabilities?**  
Continue to **Tutorial #3: Building a Hybrid Agent** to learn how to integrate web research with internal vector search. You'll unlock new capabilities for any scenario where both internal and external knowledge matter.
 
[👉 **Continue to Tutorial #3**](./hybrid-agent-tutorial.ipynb)
"""
logger.info("## Conclusion & Next Steps")

logger.info("\n\n[DONE]", bright=True)
