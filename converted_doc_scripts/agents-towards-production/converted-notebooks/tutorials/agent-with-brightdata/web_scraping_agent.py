from jet.transformers.formatters import format_json
from dotenv import load_dotenv
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langgraph.prebuilt import create_react_agent
from mcp_use.adapters.langchain_adapter import LangChainAdapter
from mcp_use.client import MCPClient
import asyncio
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
![](https://europe-west1-atp-views-tracker.cloudfunctions.net/working-analytics?notebook=tutorials--agent-with-brightdata--web-scraping-agent)

# Building an Intelligent Web Scraping Agent with LangGraph and Bright Data + MCP

## Overview

This comprehensive tutorial demonstrates how to build a production-ready web scraping agent that combines LangGraph's ReAct (Reasoning and Acting) framework with Bright Data's advanced scraping infrastructure. The resulting system can intelligently navigate websites, extract structured data, and conduct complex research workflows with minimal human intervention.

The agent you'll build represents a significant advancement over traditional web scraping approaches. Instead of writing custom scrapers for each website, you'll create an intelligent system that can reason about different scraping strategies, select appropriate tools, and adapt to various web structures automatically.

## Learning Objectives

By completing this tutorial, you will understand how to:

- Implement LangGraph ReAct agents with external tool integration for autonomous decision-making
- Configure Bright Data's Model Context Protocol (MCP) server for enterprise-grade web scraping
- Design intelligent agents that dynamically select optimal scraping strategies based on target websites
- Extract structured data from major platforms including e-commerce sites, social media, and news sources
- Implement browser automation workflows for complex user interactions
- Build comprehensive research pipelines that synthesize information from multiple sources

## Architecture Overview

The system architecture consists of three primary components working in harmony:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   User Query    │───▶│   ReAct Agent    │───▶│   Bright Data MCP   │
│                 │    │   (LangGraph)    │    │      Server         │
└─────────────────┘    │                  │    │                     │
                       │ ┌──────────────┐ │    │ ┌─────────────────┐ │
                       │ │ Reasoning    │ │    │ │ Search Engines  │ │
                       │ │ Engine       │ │    │ │ Web Scrapers    │ │
                       │ └──────────────┘ │    │ │ Platform APIs   │ │
                       │ ┌──────────────┐ │    │ │ Browser Tools   │ │
                       │ │ Tool         │ │    │ └─────────────────┘ │
                       │ │ Selection    │ │    └─────────────────────┘
                       │ └──────────────┘ │
                       └──────────────────┘
```

The ReAct agent serves as the intelligent coordinator, analyzing user requests and selecting appropriate tools from Bright Data's comprehensive suite of web scraping capabilities. This design ensures both flexibility and reliability in handling diverse web scraping scenarios.

## Prerequisites and Environment Setup

Before beginning the implementation, you'll need to establish accounts and configure your development environment. This section guides you through the essential setup steps.

### Required Accounts and API Keys

**Bright Data Account Setup**

1. Create a Bright Data account at [This link](https://brightdata.com/?hs_signup=1&utm_source=brand&utm_campaign=brnd-mkt_github_nirdiamant_logo). New accounts receive 5,000 unlocker requests monthly at no cost, providing substantial resources for development and testing.
    
    ![Screenshot: Signup Page](assets/Signup.png)

2. Navigate to your [account settings](https://brightdata.com/cp/setting) and locate your API key. This credential will authenticate your agent with Bright Data's infrastructure.

    ![Screenshot: Settings Page](assets/Settings.png)

**Language Model Access**

3. Register for an OpenRouter account to obtain API access for language models. While this tutorial uses Gemini through OpenRouter for optimal performance and cost efficiency, the architecture supports any compatible language model.

**Development Environment**

4. Ensure your development environment includes Python 3.8 or higher with pip package management capabilities.

## Dependency Installation and Configuration

The following section installs the required Python packages and establishes the foundational imports for our web scraping agent. Each dependency serves a specific role in the overall architecture.

### API Key Configuration

This cell creates environment variables for your API credentials. Replace the placeholder values with your actual API keys before execution.
"""
logger.info("# Building an Intelligent Web Scraping Agent with LangGraph and Bright Data + MCP")

# !echo "BRIGHT_DATA_API_TOKEN=<your-brightdataa-api-key>" >> .env
# !echo "OPENROUTER_API_KEY=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" >> .env

"""
### Package Installation

This cell installs the core dependencies required for the web scraping agent. The packages include LangGraph for agent orchestration, Ollama client for language model interaction, MCP client for Bright Data integration, and supporting utilities.
"""
logger.info("### Package Installation")

# %pip install langgraph langchain-ollama mcp-use python-dotenv asyncio --quiet
# %load_ext autoreload
# %autoreload 2

"""
### Core Imports and Environment Loading

This cell imports the essential libraries and loads environment variables from the configuration file. The imports include async handling capabilities, language model clients, agent frameworks, and MCP integration components.
"""
logger.info("### Core Imports and Environment Loading")


load_dotenv()

"""
### Environment Verification

Before proceeding with agent configuration, it's essential to verify that all required environment variables are properly loaded. This verification step prevents runtime errors and ensures smooth operation.
"""
logger.info("### Environment Verification")

logger.debug("✅ Environment setup complete!")
logger.debug(f"OpenRouter API Key loaded: {'Yes' if os.getenv('OPENROUTER_API_KEY') else 'No'}")

"""
## Understanding the ReAct Agent Framework

The ReAct (Reasoning and Acting) framework represents a significant advancement in autonomous agent design. Unlike traditional rule-based systems or simple chain-of-thought approaches, ReAct agents integrate reasoning capabilities with action execution in a unified framework.

### Core Principles of ReAct Architecture

ReAct agents operate on a fundamental principle that combines deliberative reasoning with interactive tool usage. This approach enables agents to:

**Dynamic Reasoning**: The agent analyzes each situation and develops a reasoning strategy tailored to the specific task requirements. Rather than following predetermined scripts, it evaluates available information and determines the most effective approach.

**Tool Selection**: Based on its reasoning, the agent selects appropriate tools from its available repertoire. This selection process considers factors such as data requirements, website characteristics, and desired output formats.

**Iterative Refinement**: The agent can execute multiple actions in sequence, using the results of previous actions to inform subsequent decisions. This capability is particularly valuable for complex web scraping scenarios that require multi-step workflows.

### Application to Web Scraping

In the context of web scraping, ReAct agents excel because they can adapt their approach based on the target website's characteristics. For instance, when encountering an e-commerce site, the agent might recognize the need for structured product data extraction. Conversely, when analyzing news articles, it might prioritize content extraction and summarization techniques.

This adaptability eliminates the need for maintaining separate scraping scripts for different websites, significantly reducing development overhead while improving reliability and maintainability.

## Bright Data MCP Integration

The Model Context Protocol (MCP) represents a standardized approach for integrating external services with language model applications. Bright Data's MCP server provides access to a comprehensive suite of web scraping tools through a unified interface.

### MCP Architecture Benefits

The MCP integration offers several advantages over direct API integration:

**Standardized Interface**: All Bright Data tools are accessible through consistent function signatures, simplifying agent development and reducing integration complexity.

**Automatic Tool Discovery**: The MCP server dynamically exposes available tools, allowing the agent to discover and utilize new capabilities without code modifications.

**Error Handling**: Built-in error handling and retry mechanisms improve reliability when dealing with network issues or temporary service unavailability.

**Performance Optimization**: The MCP server includes caching and request optimization features that enhance overall system performance.

### Bright Data MCP Server Configuration

This function establishes the connection to Bright Data's MCP server and converts the available tools into LangChain-compatible format. The configuration process involves setting up the server connection parameters and authenticating with your API credentials.
"""
logger.info("## Understanding the ReAct Agent Framework")

async def setup_bright_data_tools():
    """
    Configure Bright Data MCP client and create LangChain-compatible tools
    """

    bright_data_config = {
        "mcpServers": {
                    "Bright Data": {
            "command": "npx",
            "args": ["@brightdata/mcp"],
            "env": {
                "API_TOKEN": os.getenv("BRIGHT_DATA_API_TOKEN"),
            }
            }
        }
    }

    client = MCPClient.from_dict(bright_data_config)
    adapter = LangChainAdapter()

    tools = await adapter.create_tools(client)
    logger.success(format_json(tools))

    logger.debug(f"✅ Connected to Bright Data MCP server")
    logger.debug(f"📊 Available tools: {len(tools)}")

    return tools

tools = await setup_bright_data_tools()
logger.success(format_json(tools))

"""
### Available Tool Categories

The Bright Data MCP server provides access to several categories of web scraping tools, each optimized for specific use cases:

**Search Engine Integration**: Comprehensive access to major search engines including Google, Bing, and Yandex, enabling the agent to discover relevant web content based on user queries.

**Universal Web Scraping**: General-purpose scraping capabilities that can extract content from any website in multiple formats including Markdown and HTML, with built-in bot detection bypass mechanisms.

**Platform-Specific Extractors**: Specialized tools for major platforms such as Amazon, LinkedIn, Instagram, Facebook, X (Twitter), TikTok, YouTube, Reddit, and Zillow. These extractors understand the specific data structures of each platform and can extract information more efficiently than general-purpose scrapers.

**Browser Automation**: Advanced capabilities for simulating user interactions including navigation, clicking, typing, and screenshot capture. These tools are essential for websites that require complex user interactions or JavaScript execution.

The diversity of available tools ensures that the agent can handle virtually any web scraping scenario with optimal efficiency and reliability.

## Language Model and Agent Configuration

The next phase involves configuring the language model and creating the ReAct agent with a comprehensive system prompt. The system prompt plays a crucial role in defining the agent's behavior, reasoning patterns, and tool selection strategies.

### ReAct Agent Initialization

This function creates the complete web scraping agent by combining the language model with the Bright Data tools. The system prompt is carefully crafted to guide the agent's decision-making process and ensure optimal tool utilization.
"""
logger.info("### Available Tool Categories")


async def create_web_scraper_agent():
    """
    Create a ReAct agent configured for intelligent web scraping
    """

    tools = await setup_bright_data_tools()
    logger.success(format_json(tools))

    current_date = datetime.datetime.now().strftime("%B %d, %Y")

    llm = ChatOllama(
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        model_name="google/gemini-2.5-flash-lite-preview-06-17",  # Fast and capable model for reasoning
        temperature=0.1  # Low temperature for consistent, focused responses
    )

    system_prompt = f"""You are a web data extraction agent. Today's date is {current_date}.

You have {len(tools)} specialized tools for web scraping and data extraction. When users request web data or current information, you MUST use these tools - do not rely on your training data.

Available capabilities:
- Search engines (Google/Bing/Yandex)
- Universal web scraping (any website)
- Platform extractors (Amazon, LinkedIn, Instagram, Facebook, X, TikTok, YouTube, Reddit, etc.)
- Browser automation

Process:
1. Identify data need
2. Select appropriate tool
3. Execute extraction
4. Return structured results

Always use tools for current/live data requests."""

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt
    )

    logger.debug("🤖 ReAct Web Scraper Agent created successfully!")
    return agent

agent = await create_web_scraper_agent()
logger.success(format_json(agent))

"""
### System Prompt Design Principles

The system prompt serves as the foundation for the agent's behavior and incorporates several key design principles:

**Clear Role Definition**: The prompt establishes the agent's identity as a web data extraction specialist, providing clear context for its primary function.

**Tool Awareness**: By explicitly stating the number and types of available tools, the prompt ensures the agent understands its capabilities and prioritizes tool usage over relying on training data.

**Process Framework**: The four-step process outlined in the prompt provides a structured approach to handling user requests, ensuring consistent and methodical responses.

**Temporal Context**: Including the current date helps the agent understand the temporal context of requests and prioritize recent information when relevant.

**Mandatory Tool Usage**: The instruction to always use tools for current data requests prevents the agent from providing outdated information from its training data.

## Basic Search Functionality Testing

With the agent configured, we can now test its fundamental capabilities. The first test demonstrates the agent's ability to search for current information and synthesize results from multiple sources.

### Understanding Agent Decision-Making

When presented with a search query, the agent follows a systematic decision-making process. It first analyzes the query to understand the information requirements, then selects the most appropriate search tool based on the query characteristics. The agent then executes the search, analyzes the results, and presents a structured summary.

### Basic Search Query Execution

This test demonstrates the agent's ability to search for current information and process multiple search results into a coherent summary. The agent will automatically select appropriate search engines and present the findings in a structured format.
"""
logger.info("### System Prompt Design Principles")

async def test_basic_search():
    """
    Test the agent's ability to search for current information
    """

    logger.debug("Testing Basic Search Functionality...")
    logger.debug("="*50)

    search_result = await agent.ainvoke({
            "messages": [("human", "Give me the latest AI news from this week, Include full URLs to source.")],
        })
    logger.success(format_json(search_result))

    logger.debug("\n🔍 Search Results:")
    logger.debug(search_result["messages"][-1].content)

    return search_result

basic_search_result = await test_basic_search()
logger.success(format_json(basic_search_result))

"""
### Search Result Analysis

The search functionality demonstrates several key capabilities of the ReAct agent:

**Query Understanding**: The agent correctly interprets the request for recent AI news and understands the temporal requirement ("this week").

**Tool Selection**: Based on the query type, the agent selects the appropriate search engine tool from its available options.

**Result Processing**: The agent aggregates results from multiple sources and presents them in a structured, readable format with source URLs.

**Content Prioritization**: The results show the agent's ability to identify and prioritize relevant, current information from authoritative sources.

## Advanced Platform-Specific Data Extraction

Beyond general web searching, the agent's true power lies in its ability to extract structured data from specific platforms. Each platform presents unique challenges in terms of data structure, access methods, and content organization.

### E-commerce Platform Analysis

E-commerce platforms like Amazon contain rich structured data including product specifications, pricing, reviews, and availability information. The agent's platform-specific tools can navigate these complex data structures and extract relevant information efficiently.

When analyzing e-commerce data, the agent considers multiple factors including product features, pricing trends, customer feedback, and comparative analysis across similar products. This comprehensive approach provides users with actionable insights for purchase decisions or market research.

### E-commerce Data Extraction and Analysis

This test demonstrates the agent's ability to research and compare products on e-commerce platforms. The agent will search for products, extract structured data, and provide comparative analysis with pricing and feature information.
"""
logger.info("### Search Result Analysis")

async def test_ecommerce_scraping():
    """
    Test structured data extraction from e-commerce platforms
    """

    logger.debug("Testing E-commerce Data Extraction...")
    logger.debug("="*50)

    ecommerce_result = await agent.ainvoke({
            "messages": [("human", "Find information about the top-rated wireless headphones on Amazon and compare their features and prices")]
        })
    logger.success(format_json(ecommerce_result))

    logger.debug("\n🛒 E-commerce Analysis:")
    logger.debug(ecommerce_result["messages"][-1].content)

    return ecommerce_result

ecommerce_result = await test_ecommerce_scraping()
logger.success(format_json(ecommerce_result))

"""
### E-commerce Analysis Capabilities

The e-commerce analysis demonstrates several sophisticated capabilities:

**Product Discovery**: The agent can search for products within specific categories and identify top-rated options based on customer reviews and ratings.

**Feature Extraction**: Detailed product specifications, features, and technical details are systematically extracted and organized for easy comparison.

**Price Analysis**: Current pricing information is gathered and presented alongside historical pricing trends when available.

**Comparative Framework**: The agent structures its analysis to facilitate decision-making by highlighting key differentiators between products.

**Actionable Recommendations**: Rather than simply presenting data, the agent provides guidance based on different use cases and budget considerations.

## Social Media Content Analysis

Social media platforms represent a unique challenge for data extraction due to their dynamic nature, complex user interfaces, and diverse content formats. The agent's social media capabilities enable analysis of discussions, sentiment, and trending topics across various platforms.

### Reddit Discussion Analysis

Reddit's structure of communities (subreddits) and threaded discussions provides rich opportunities for understanding public opinion and expert insights on specific topics. The agent can navigate Reddit's interface, extract relevant discussions, and analyze the content for key themes and insights.

### Reddit Content Extraction and Analysis

This test demonstrates the agent's ability to search for and analyze social media content. The agent will search for relevant Reddit discussions and extract key conversation topics and community insights.
"""
logger.info("### E-commerce Analysis Capabilities")

async def test_social_media_simple():
    """
    Test Reddit extraction with a specific approach
    """
    logger.debug("Testing Reddit Extraction...")
    logger.debug("="*50)

    result = await agent.ainvoke({
            "messages": [("human", "Search for 'electric vehicles reddit' and then scrape one of the Reddit discussion pages you find. Show me what people are discussing.")]
        })
    logger.success(format_json(result))

    logger.debug("\n📱 Reddit Analysis:")
    logger.debug(result["messages"][-1].content)
    return result

social_simple = await test_social_media_simple()
logger.success(format_json(social_simple))

"""
### Social Media Analysis Insights

The social media analysis showcases several key capabilities:

**Platform-Specific Navigation**: The agent understands Reddit's structure and can navigate to specific subreddits and discussion threads.

**Content Aggregation**: Multiple posts and comments are analyzed to identify common themes and discussion topics.

**Sentiment Analysis**: The agent can interpret the tone and sentiment of discussions, identifying positive, negative, or neutral attitudes toward topics.

**Trend Identification**: By analyzing multiple discussions, the agent can identify emerging trends and popular topics within communities.

**Context Understanding**: The agent maintains context about the specific community culture and discussion norms when interpreting content.

## Complex Multi-Step Research Workflows

The agent's most sophisticated capability lies in conducting complex research that requires multiple tools, reasoning steps, and information synthesis. These workflows demonstrate the full potential of the ReAct framework in handling real-world research scenarios.

### Research Workflow Architecture

Complex research workflows involve several interconnected phases:

**Task Decomposition**: The agent analyzes complex queries and breaks them down into manageable sub-tasks that can be addressed individually.

**Tool Orchestration**: Multiple tools are employed in sequence, with each tool's output informing the selection and configuration of subsequent tools.

**Information Synthesis**: Data from various sources is analyzed, compared, and synthesized into coherent insights that address the original research question.

**Quality Validation**: The agent evaluates the quality and relevance of gathered information, identifying gaps that require additional research.

### Multi-Step Research Execution

This test challenges the agent with a complex research task requiring multiple information sources and analysis steps. The agent must demonstrate its ability to plan research steps, execute them systematically, and synthesize findings into actionable insights.
"""
logger.info("### Social Media Analysis Insights")

async def test_complex_research():
    """
    Test the agent's ability to conduct multi-step research
    """

    logger.debug("Testing Complex Multi-Step Research...")
    logger.debug("="*50)

    research_result = await agent.ainvoke({
            "messages": [("human", """
            I need to research the current state of the renewable energy market. Please:
            1. Find recent news about renewable energy developments
            2. Look up major renewable energy companies and their stock performance
            3. Analyze social media sentiment about renewable energy
            4. Provide a comprehensive market overview with key insights
            """)]
        })
    logger.success(format_json(research_result))

    logger.debug("\n🔬 Complex Research Results:")
    logger.debug(research_result["messages"][-1].content)

    return research_result

research_result = await test_complex_research()
logger.success(format_json(research_result))

"""
### Research Methodology and Planning

The agent's response to complex research requests demonstrates sophisticated planning capabilities:

**Requirement Analysis**: The agent evaluates the research request and identifies areas where additional specification would improve research quality.

**Scope Definition**: Rather than proceeding with assumptions, the agent seeks clarification to ensure research efforts are focused and relevant.

**Tool Mapping**: The agent understands which tools are appropriate for different types of information gathering and outlines its approach.

**Quality Assurance**: By requesting specific parameters, the agent ensures that the final research output will meet professional standards and user expectations.

This planning approach reflects best practices in professional research methodology and demonstrates the agent's capability to handle enterprise-level research requirements.

## Comprehensive Research Assistant Implementation

The final component of our tutorial involves creating a comprehensive research assistant function that demonstrates the full integration of all the agent's capabilities. This function serves as a template for building production-ready research workflows.

### Research Assistant Architecture

The research assistant function incorporates several design principles that make it suitable for production use:

**Parameterized Configuration**: Users can specify research scope, source limits, and other parameters to control the research process.

**Structured Output**: Research results are organized in a consistent format that facilitates further analysis or reporting.

**Progress Tracking**: The function provides feedback on research progress, helping users understand the complexity and scope of the work being performed.

**Quality Controls**: Built-in mechanisms ensure that research focuses on high-quality, relevant sources rather than simply maximizing the quantity of information gathered.

### Research Assistant Function Implementation

This comprehensive function demonstrates how to structure complex research queries and configure the agent for optimal performance. The function includes parameterization for research scope and provides structured output formatting.
"""
logger.info("### Research Methodology and Planning")

async def research_assistant(query: str, max_sources: int = 5):
    """
    A comprehensive research assistant using our web scraping agent

    Args:
        query (str): The research question or topic
        max_sources (int): Maximum number of sources to analyze
    """

    logger.debug(f"🔍 Starting research on: {query}")
    logger.debug("="*60)

    research_prompt = f"""
    Please conduct comprehensive research on: "{query}"

    Your research should include:
    1. Current news and developments (last 30 days)
    2. Expert opinions and analysis
    3. Statistical data and trends
    4. Social media sentiment (if relevant)
    5. Key players and companies involved

    Use multiple sources and provide a well-structured summary with:
    - Executive summary
    - Key findings
    - Supporting data
    - Sources used

    Limit your research to {max_sources} high-quality sources.
    """

    result = await agent.ainvoke({
            "messages": [("human", research_prompt)]
        })
    logger.success(format_json(result))

    logger.debug(f"\n📊 Research Complete!")
    logger.debug(result["messages"][-1].content)

    return result

research_result = await research_assistant("Impact of artificial intelligence on job markets in 2025",5)
logger.success(format_json(research_result))

"""
### Research Assistant Benefits and Applications

The comprehensive research assistant function demonstrates several key benefits:

**Scalable Research Framework**: The parameterized approach allows the same function to handle research projects of varying scope and complexity.

**Consistent Output Structure**: The structured prompt ensures that research results follow a predictable format, facilitating integration with other systems or processes.

**Quality Focus**: By limiting the number of sources and emphasizing quality, the function prioritizes depth over breadth in research coverage.

**Professional Standards**: The research methodology incorporates best practices from professional research organizations, ensuring outputs meet enterprise requirements.

**Customization Capability**: The function can be easily modified to address specific industry requirements or research methodologies.
"""
logger.info("### Research Assistant Benefits and Applications")

logger.info("\n\n[DONE]", bright=True)