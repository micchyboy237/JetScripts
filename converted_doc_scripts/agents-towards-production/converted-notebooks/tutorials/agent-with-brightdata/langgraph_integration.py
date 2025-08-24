from dotenv import load_dotenv
from jet.logger import CustomLogger
from langchain_brightdata import BrightDataSERP
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
![](https://europe-west1-atp-views-tracker.cloudfunctions.net/working-analytics?notebook=tutorials--agent-with-brightdata--langgraph-integration)

# Building an Intelligent Web Scraping Agent with LangGraph and Bright Data

## Overview

This comprehensive tutorial demonstrates how to construct an intelligent web scraping agent using LangGraph's ReAct framework integrated with Bright Data's native LangChain tools. The approach combines reasoning capabilities with powerful web data extraction to create a system that can autonomously search, analyze, and extract information from the web.

Modern data collection workflows require more than simple scraping tools. They need intelligent agents that can understand context, make decisions about search strategies, and adapt to different types of queries. This tutorial addresses that need by building a ReAct (Reasoning and Acting) agent that combines the decision-making capabilities of large language models with the robust data collection infrastructure provided by Bright Data.

## Learning Objectives

By completing this tutorial, you will understand how to:

- Implement LangGraph ReAct agents with Bright Data's LangChain integration
- Configure and utilize Bright Data's SERP (Search Engine Results Page) tool for intelligent web searches
- Design agents that can reason through complex web scraping tasks automatically
- Handle real-time data extraction with appropriate error management
- Construct scalable web scraping workflows for production environments

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   LangGraph      │───▶│   Bright Data   │
│                 │    │   ReAct Agent    │    │   SERP Tool     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                          │
                              ▼                          ▼
                       ┌─────────────┐            ┌─────────────┐
                       │   Google    │            │   Search    │
                       │   Gemini    │            │   Results   │
                       │    LLM      │            │   (JSON)    │
                       └─────────────┘            └─────────────┘
                              │                          │
                              └────────┬─────────────────┘
                                       ▼
                               ┌───────────────┐
                               │   Structured  │
                               │   Response    │
                               └───────────────┘
```

The system operates through a coordinated workflow where the ReAct agent receives user queries, determines the appropriate search strategy using the language model, executes searches through Bright Data's infrastructure, and synthesizes the results into structured responses.

## Getting Started

Follow these steps to set up your development environment:

1. **Sign up** for Bright Data at [This link](https://brightdata.com/?hs_signup=1&utm_source=brand&utm_campaign=brnd-mkt_github_nirdiamant_logo) (currently offering 5k unlocker requests for free - every month!).
    
    _Refer to the screenshots linked below for step-by-step guidance:_
    
    - ![Screenshot: Signup Page](assets/Signup.png)
2. **Copy your API key** from [your account settings](https://brightdata.com/cp/setting):
    
    - ![Screenshot: Settings Page](assets/Settings.png)
3. **LLM API key** - We'll use Google's Gemini through their API. Sign up at [Google AI Studio](https://aistudio.google.com/) to get your API key.
    
4. **Install required dependencies** and set up your environment variables.

## Installation and Dependency Management

The implementation relies on several key libraries that provide different aspects of the agent functionality. Understanding these dependencies helps in troubleshooting and extending the system later.

### Core Package Installation
"""
logger.info("# Building an Intelligent Web Scraping Agent with LangGraph and Bright Data")

# %pip install langchain-brightdata langchain-google-genai langgraph python-dotenv --quiet
# %load_ext autoreload
# %autoreload 2

"""
### Environment Variable Configuration

Proper API key management is crucial for both security and functionality. The following cell demonstrates how to securely store your credentials in environment variables.
"""
logger.info("### Environment Variable Configuration")

# !echo "BRIGHT_DATA_API_TOKEN=<your-brightdata-api-key>" >> .env
# !echo "GOOGLE_API_KEY=<your-google-api-key>" >> .env

"""
### Library Import and Environment Validation

This section imports all necessary libraries and validates that the environment is properly configured with the required API keys.
"""
logger.info("### Library Import and Environment Validation")


load_dotenv()

logger.debug("Environment setup complete!")
logger.debug(f"Bright Data API Key loaded: {'Yes' if os.getenv('BRIGHT_DATA_API_TOKEN') else 'No'}")
logger.debug(f"Google API Key loaded: {'Yes' if os.getenv('GOOGLE_API_KEY') else 'No'}")

"""
## Core Components and Architecture

Understanding the individual components and their interactions is essential for effectively implementing and customizing the web scraping agent.

### LangGraph ReAct Agent Framework

The ReAct (Reasoning and Acting) paradigm represents a significant advancement in agent design. Traditional approaches either focus purely on reasoning or purely on action execution. ReAct agents integrate both capabilities, allowing them to reason about problems systematically while taking concrete actions to gather information or manipulate their environment.

In the context of web scraping, this integration is particularly valuable. The agent can analyze a user's query to determine what type of information is needed, formulate appropriate search strategies, execute those searches, analyze the results, and potentially refine its approach based on what it discovers. This creates a much more intelligent and adaptive scraping system compared to static, rule-based approaches.

### Bright Data SERP Integration

Bright Data's Search Engine Results Page (SERP) tool provides enterprise-grade web search capabilities through their global proxy network. This infrastructure offers several advantages over direct search engine APIs, including geographic diversity, rate limiting management, and consistent data formatting.

The tool supports multiple search engines including Google and Bing, with configurable parameters for geographic location, language preferences, and result quantities. The built-in parsing functionality converts raw search results into structured data that can be easily processed by the language model.

### Google Gemini Language Model

Google's Gemini model serves as the reasoning engine for the agent. Its multimodal capabilities and strong performance on reasoning tasks make it well-suited for analyzing search results and determining appropriate next actions. The model's ability to understand context and generate coherent responses is crucial for synthesizing information from multiple sources into useful insights.

## Implementation: Building the Web Scraping Agent

The implementation process involves configuring each component and integrating them into a cohesive system. We'll build the agent step by step, explaining the rationale behind each configuration decision.

### Language Model Initialization

The first step involves establishing a connection to the Google Gemini API and configuring the model parameters for optimal performance in web scraping scenarios.
"""
logger.info("## Core Components and Architecture")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # Fast and capable model
    temperature=0.1  # Low temperature for consistent, focused responses
)

logger.debug("Language model initialized successfully!")

"""
The configuration uses `gemini-2.0-flash` for its balance of speed and capability, particularly important for interactive agent applications. The low temperature setting (0.1) ensures consistent and focused responses, which is crucial for reliable agent behavior in production environments. Higher temperatures might introduce unnecessary variability in the agent's decision-making process.

### Bright Data SERP Tool Configuration

The SERP tool configuration determines how the agent will interact with search engines and what type of results it will receive.
"""
logger.info("### Bright Data SERP Tool Configuration")

serp_tool = BrightDataSERP(
    search_engine="google",  # Use Google as the search engine
    country="us",           # Search from US perspective
    language="en",          # English language results
    results_count=10,       # Get top 10 results
    parse_results=True,     # Automatically parse and structure results
)

logger.debug("Bright Data SERP tool configured successfully!")
logger.debug(f"   Search Engine: {serp_tool.search_engine}")
logger.debug(f"   Country: {serp_tool.country}")
logger.debug(f"   Language: {serp_tool.language}")
logger.debug(f"   Results Count: {serp_tool.results_count}")

"""
The configuration parameters significantly impact the type and quality of search results. Google is selected as the primary search engine due to its comprehensive index and result quality. The US country setting and English language preference optimize for content relevant to English-speaking audiences, though these can be adjusted for different markets or languages.

The results count of 10 provides a good balance between comprehensive coverage and processing efficiency. More results offer greater information breadth but require more processing time and API resources. The automatic parsing feature ensures that raw HTML responses are converted into structured data that the language model can easily understand and process.

### ReAct Agent Assembly

The final step in the basic implementation combines the language model and search tool into a functional ReAct agent.
"""
logger.info("### ReAct Agent Assembly")

agent = create_react_agent(
    model=llm,           # The language model we initialized
    tools=[serp_tool],   # List of tools available to the agent
    prompt="""You are a web researcher agnet with access to SERP tool, you will HAVE to use the user query, if no specific, country, language, search engine or specific vertical, choose what's best fit users questions"""
)

logger.debug("ReAct Web Scraper Agent created successfully!")
logger.debug("Agent is ready to search and analyze web content!")

"""
The `create_react_agent()` function integrates the language model with the available tools, creating an agent that can autonomously decide when and how to use the search capabilities. The prompt provides guidance to the agent about its role and decision-making process, ensuring it understands when to utilize the search tool and how to adapt its strategy based on user queries.

## Agent Testing and Validation

Testing the agent with various query types helps validate its functionality and demonstrates its reasoning capabilities.

### Basic Search Functionality Test

This test demonstrates how the agent processes a straightforward information request and utilizes its search capabilities.
"""
logger.info("## Agent Testing and Validation")

logger.debug("Testing Basic Search...")
logger.debug("="*50)

user_query = "What are the latest developments and news in AI technology in the US?"

for step in agent.stream(
    {"messages": [("human", user_query)]},
    stream_mode="values",
):
    step["messages"][-1].pretty_logger.debug()

"""
The streaming output allows you to observe the agent's reasoning process in real-time. You'll see how the agent analyzes the query, decides to use the search tool, formulates appropriate search terms, executes the search, and synthesizes the results into a coherent response. This transparency is valuable for understanding agent behavior and debugging issues.

## Agent Behavior Analysis

Understanding how the agent processes different types of queries provides insights into its decision-making process and helps optimize its performance.

### Query Processing Pipeline

The agent follows a systematic approach when processing user queries:

**Initial Analysis Phase:** The agent examines the user's question to determine the type of information required, whether current or real-time data is necessary, and how specific or broad the search scope should be.

**Tool Selection Logic:** Based on the analysis, the agent decides whether to use the search tool. This decision depends on factors such as whether the query requires current information beyond the model's training data, if the question involves specific factual claims that need verification, and how detailed the information request is.

**Search Strategy Formation:** When search is determined to be necessary, the agent formulates appropriate search terms and parameters. This involves extracting key concepts from the user query, determining optimal search term combinations, and selecting appropriate search parameters.

**Result Synthesis:** After receiving search results, the agent analyzes information from multiple sources, synthesizes findings into coherent insights, provides structured and relevant answers, and cites sources when appropriate.

### Advanced Configuration Patterns

The basic agent can be extended and customized for various specialized use cases through configuration modifications.

**Multi-Language Research Configuration:**
```python
spanish_serp = BrightDataSERP(
    search_engine="google",
    country="es",      # Spain
    language="es",     # Spanish
    results_count=15,
    parse_results=True,
)

spanish_agent = create_react_agent(llm, [spanish_serp])
```

**Alternative Search Engine Integration:**
```python
bing_serp = BrightDataSERP(
    search_engine="bing",  # Use Bing instead of Google
    country="us",
    language="en",
    results_count=10,
    parse_results=True,
)

bing_agent = create_react_agent(llm, [bing_serp])
```

**High-Volume Research Configuration:**
```python
research_serp = BrightDataSERP(
    search_engine="google",
    country="us",
    language="en",
    results_count=20,  # More results for comprehensive research
    parse_results=True,
)

research_agent = create_react_agent(llm, [research_serp])
```

## Advanced Implementation: Research Assistant

Building upon the basic agent, we can create more sophisticated research assistants that handle complex, multi-faceted queries with enhanced capabilities.

### Configurable Research Assistant Factory

This implementation demonstrates how to create reusable agent configurations for different research scenarios.
"""
logger.info("## Agent Behavior Analysis")

def create_research_assistant(search_engine="google", country="us", language="en"):
    """
    Create a specialized research assistant agent

    Args:
        search_engine (str): Search engine to use ("google" or "bing")
        country (str): Country code for localized results
        language (str): Language code for results

    Returns:
        Agent configured for research tasks
    """

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)

    serp_tool = BrightDataSERP(
        search_engine=search_engine,
        country=country,
        language=language,
        results_count=15,  # More results for thorough research
        parse_results=True,
    )

    agent = create_react_agent(llm, [serp_tool])

    logger.debug(f"Research Assistant created!")
    logger.debug(f"   Engine: {search_engine.title()}")
    logger.debug(f"   Location: {country.upper()}")
    logger.debug(f"   Language: {language.upper()}")

    return agent

research_assistant = create_research_assistant()

"""
### Comprehensive Research Query Testing

The research assistant can handle complex, multi-part queries that require comprehensive information gathering and analysis.
"""
logger.info("### Comprehensive Research Query Testing")

research_query = """
Please research the renewable energy market trends for 2024-2025.
I need information about:
1. Market growth predictions
2. Leading companies and their strategies
3. Recent technological breakthroughs
4. Government policies affecting the sector
"""

logger.debug("Starting comprehensive research...")
logger.debug("="*60)

for step in research_assistant.stream(
    {"messages": [("human", research_query)]},
    stream_mode="values",
):
    step["messages"][-1].pretty_logger.debug()

"""
## Conclusion and Future Directions

This tutorial has demonstrated the construction of an intelligent web scraping agent that combines the reasoning capabilities of large language models with robust web data collection infrastructure. The resulting system can autonomously search the web, process real-time information, conduct comprehensive research across multiple topics, handle complex queries with multi-step reasoning, and provide structured, actionable insights from web data.

### Key Accomplishments

The implementation covers essential aspects of modern agent development including complete web scraping environment setup, intelligent ReAct agent creation, integration with Bright Data's powerful search capabilities, error handling and optimization strategies, and development of specialized research assistants.

### Extension Opportunities

The foundational system presented here can be extended in several directions to create more sophisticated and specialized applications.

**Tool Integration Expansion:** Additional Bright Data tools can be integrated to expand the agent's capabilities beyond search, including web scraping, data parsing, and content extraction tools.

**User Interface Development:** A web-based interface can be built to make the agent accessible to non-technical users, with features for query history, result export, and research project management.

**Data Persistence Systems:** Implementing data storage solutions allows for research result archiving, trend analysis over time, and building institutional knowledge bases.

**Automation and Scheduling:** The agent can be enhanced with scheduling capabilities for automated research tasks, monitoring specific topics, and generating regular reports.

**Domain Specialization:** Specialized agents can be developed for specific industries or use cases, with customized prompts, specialized tools, and domain-specific knowledge integration.

### Production Considerations

When deploying agents in production environments, several additional considerations become important including rate limiting and API usage management, error handling and recovery mechanisms, result caching and optimization, security and access control, and monitoring and logging systems.

### Application Domains

The techniques demonstrated in this tutorial are applicable to a wide range of real-world scenarios including competitive intelligence and market monitoring, academic research and literature discovery, content curation and trend analysis, price monitoring and market research, and news aggregation and analysis.

The combination of intelligent reasoning with robust web data collection creates powerful possibilities for automated information gathering and analysis. As these technologies continue to evolve, the potential applications will expand, making intelligent web scraping agents an increasingly valuable tool for organizations and researchers across various domains.
"""
logger.info("## Conclusion and Future Directions")

logger.info("\n\n[DONE]", bright=True)