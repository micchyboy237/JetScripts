from datetime import datetime
from dotenv import load_dotenv
from jet.llm.ollama.base_langchain import ChatOllama
from jet.logger import CustomLogger
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from pydantic import BaseModel
from tavily import TavilyClient
from typing import Dict, List, Any, TypedDict, Optional
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# AInsight: AI/ML Weekly News Reporter

## 📚 Overview
This notebook demonstrates the implementation of an intelligent news aggregation and summarization system using a multi-agent architecture. AInsight automatically collects, processes, and summarizes AI/ML news for general audiences.

### Motivation
The rapid evolution of AI/ML technology creates several challenges:
- Information overload from multiple news sources
- Technical complexity making news inaccessible to general audiences
- Time-consuming manual curation and summarization
- Inconsistent reporting formats and quality

AInsight addresses these challenges through:
- Automated news collection and filtering
- Intelligent summarization for non-technical readers
- Consistent, well-structured reporting
- Scalable, maintainable architecture
- Saving time and effort!

## 🏗️ Multi-Agent System Architecture

AInsight processes news through three specialized agents:

1. **NewsSearcher Agent**
   - Primary news collection engine
   - Interfaces with Tavily API
   - Filters for relevance and recency
   - Handles source diversity

2. **Summarizer Agent**
   - Processes technical content
   - Uses llama3.2 for natural language generation (LLM can be configured per user preference, used Ollama in this tutorial for accessibility)
   - Handles technical term simplification

3. **Publisher Agent**
   - Takes list of summaries as input
   - Formats them into a structured prompt
   - Makes single llama3.2 call to generate complete report with:
     * Introduction section
     * Organized summaries
     * Further reading links
   - Saves final report as markdown file

<div style="text-align: center;">

<img src="../images/ainsight_langgraph.svg" alt="ainsight by langgraph" style="width:20%; height:50%;">
</div>

### 🎯 Learning Objectives
1. Understand multi-agent system architecture
2. Implement state management with LangGraph
3. Work with external APIs (Tavily, Ollama)
4. Create modular, maintainable Python code

### 🔧 Technical Requirements
- Python 3.11+
- Ollama API key
- Tavily API key
- Required packages (see setup section)

---

## 🚀 Setup and Configuration

First, let's install the required packages:
"""
logger.info("# AInsight: AI/ML Weekly News Reporter")

# !pip install langchain langchain-openai langgraph tavily-python python-dotenv

"""
### Environment Configuration

Create a `.env` file in your project directory with the following:

```plaintext
# OPENAI_API_KEY=your-openai-api-key
TAVILY_API_KEY=your-tavily-api-key
```
"""
logger.info("### Environment Configuration")


load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
llm = ChatOllama(
    model="llama3.2",
    temperature=0.1,
    max_tokens=600
)

"""
## 📊 Data Models and State Management

Think of state as a "memory" that will flow through your workflow (graph) later.

We use Pydantic and TypedDict to define our data structures:
"""
logger.info("## 📊 Data Models and State Management")

class Article(BaseModel):
    """
    Represents a single news article

    Attributes:
        title (str): Article headline
        url (str): Source URL
        content (str): Article content
    """
    title: str
    url: str
    content: str

class Summary(TypedDict):
    """
    Represents a processed article summary

    Attributes:
        title (str): Original article title
        summary (str): Generated summary
        url (str): Source URL for reference
    """
    title: str
    summary: str
    url: str

class GraphState(TypedDict):
    """
    Maintains workflow state between agents

    Attributes:
        articles (Optional[List[Article]]): Found articles
        summaries (Optional[List[Summary]]): Generated summaries
        report (Optional[str]): Final compiled report
    """
    articles: Optional[List[Article]]
    summaries: Optional[List[Summary]]
    report: Optional[str]

"""
## 🤖 Agent Implementation

### 1. NewsSearcher Agent
"""
logger.info("## 🤖 Agent Implementation")

class NewsSearcher:
    """
    Agent responsible for finding relevant AI/ML news articles
    using the Tavily search API
    """

    def search(self) -> List[Article]:
        """
        Performs news search with configured parameters

        Returns:
            List[Article]: Collection of found articles
        """
        response = tavily.search(
            query="artificial intelligence and machine learning news",
            topic="news",
            time_period="1w",
            search_depth="advanced",
            max_results=5
        )

        articles = []
        for result in response['results']:
            articles.append(Article(
                title=result['title'],
                url=result['url'],
                content=result['content']
            ))

        return articles

"""
### 2. Summarizer Agent
"""
logger.info("### 2. Summarizer Agent")

class Summarizer:
    """
    Agent that processes articles and generates accessible summaries
    using llama3.2
    """

    def __init__(self):
        self.system_prompt = """
        You are an AI expert who makes complex topics accessible
        to general audiences. Summarize this article in 2-3 sentences, focusing on the key points
        and explaining any technical terms simply.
        """

    def summarize(self, article: Article) -> str:
        """
        Generates an accessible summary of a single article

        Args:
            article (Article): Article to summarize

        Returns:
            str: Generated summary
        """
        response = llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Title: {article.title}\n\nContent: {article.content}")
        ])
        return response.content

"""
### 3. Publisher Agent
"""
logger.info("### 3. Publisher Agent")

class Publisher:
    """
    Agent that compiles summaries into a formatted report
    and saves it to disk
    """

    def create_report(self, summaries: List[Dict]) -> str:
        """
        Creates and saves a formatted markdown report

        Args:
            summaries (List[Dict]): Collection of article summaries

        Returns:
            str: Generated report content
        """
        prompt = """
        Create a weekly AI/ML news report for the general public.
        Format it with:
        1. A brief introduction
        2. The main news items with their summaries
        3. Links for further reading

        Make it engaging and accessible to non-technical readers.
        """

        summaries_text = "\n\n".join([
            f"Title: {item['title']}\nSummary: {item['summary']}\nSource: {item['url']}"
            for item in summaries
        ])

        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=summaries_text)
        ])

        current_date = datetime.now().strftime("%Y-%m-%d")
        markdown_content = f"""
        Generated on: {current_date}

        {response.content}
        """

        filename = f"ai_news_report_{current_date}.md"
        with open(filename, 'w') as f:
            f.write(markdown_content)

        return response.content

"""
## 🔄 Workflow Implementation

### State Management Nodes

You can think of nodes as the "workers" (aka agents) in your workflow. Each node:

1. Takes the current state
2. Processes it
3. Returns updated state

For example the node of NewsSearcher agent:  

1. Takes current state (empty at first)
2. Searches for articles
3. Updates state with found articles
"""
logger.info("## 🔄 Workflow Implementation")

def search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node for article search

    Args:
        state (Dict[str, Any]): Current workflow state

    Returns:
        Dict[str, Any]: Updated state with found articles
    """
    searcher = NewsSearcher()
    state['articles'] = searcher.search()
    return state

def summarize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node for article summarization

    Args:
        state (Dict[str, Any]): Current workflow state

    Returns:
        Dict[str, Any]: Updated state with summaries
    """
    summarizer = Summarizer()
    state['summaries'] = []

    for article in state['articles']: # Uses articles from previous node
        summary = summarizer.summarize(article)
        state['summaries'].append({
            'title': article.title,
            'summary': summary,
            'url': article.url
        })
    return state

def publish_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node for report generation

    Args:
        state (Dict[str, Any]): Current workflow state

    Returns:
        Dict[str, Any]: Updated state with final report
    """
    publisher = Publisher()
    report_content = publisher.create_report(state['summaries'])
    state['report'] = report_content
    return state

"""
### Workflow Graph Creation
"""
logger.info("### Workflow Graph Creation")

def create_workflow() -> StateGraph:
    """
    Constructs and configures the workflow graph
    search -> summarize -> publish

    Returns:
        StateGraph: Compiled workflow ready for execution
    """

    workflow = StateGraph(state_schema=GraphState)

    workflow.add_node("search", search_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("publish", publish_node)

    workflow.add_edge("search", "summarize") # search results flow to summarizer
    workflow.add_edge("summarize", "publish") # summaries flow to publisher

    workflow.set_entry_point("search")

    return workflow.compile()

"""
## 🎬 Usage Example
"""
logger.info("## 🎬 Usage Example")

if __name__ == "__main__":
    workflow = create_workflow()
    final_state = workflow.invoke({
        "articles": None,
        "summaries": None,
        "report": None
    })

    logger.debug("\n=== AI/ML Weekly News Report ===\n")
    logger.debug(final_state['report'])

"""
## 📝 Customization Options

1. Modify search parameters in `NewsSearcher`:
   - `search_depth`: "basic" or "advanced"
   - `max_results`: Number of articles to fetch
   - `time_period`: "1d", "1w", "1m", etc.

2. Adjust summarization in `Summarizer`:
   - Update `system_prompt` for different summary styles
   - Modify GPT model parameters (temperature, max_tokens)

3. Customize report format in `Publisher`:
   - Edit the report prompt for different layouts
   - Modify markdown template

## 🤔 Additional Considerations

### Current Limitations

1. **Content Access**
   - Limited to publicly available news
   - Dependency on Tavily API coverage
   - No access to paywalled content

2. **Language Support**
   - Primary focus on English content

3. **Technical Constraints**
   - API rate limits

### Potential Improvements

1. **Enhanced News Collection**
   - Specify domains to search on depending on user preference

2. **Improved Summarization**
   - Add multi-language support
   - Implement fact-checking

3. **Advanced Features**
   - Topic classification
   - Trend detection

### Specific Use Cases

1. **Research Organizations**
   - Track technology developments
   - Monitor competition
   - Identify collaboration opportunities

2. **Educational Institutions**
   - Create teaching materials
   - Support student research
   - Track field developments

3. **Tech Companies**
   - Market intelligence
   - Innovation tracking
   - Strategic planning

4. **Media Organizations**
   - Content curation
   - Story research
   - Trend analysis

## 🔍 Troubleshooting

1. API Key Issues:
   - Ensure `.env` file exists and contains valid keys
   - Check API key permissions and quotas

2. Package Dependencies:
   - Run `pip list` to verify installations
   - Check package versions for compatibility

3. Rate Limits:
   - Monitor API usage
   - Implement retry logic if needed

## References:
- Tavily search API doc: https://docs.tavily.com/docs/rest-api/api-reference
- LangGraph Conceptual Guides: https://langchain-ai.github.io/langgraph/concepts/low_level/
"""
logger.info("## 📝 Customization Options")

logger.info("\n\n[DONE]", bright=True)