import asyncio
from jet.transformers.formatters import format_json
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OllamaChatCompletionClient
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from jet.logger import CustomLogger
import arxiv
import os
import requests
import shutil
import time


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Literature Review

A common task while exploring a new topic is to conduct a literature review. In this example we will explore how a multi-agent team can be configured to conduct a _simple_ literature review.

- **Arxiv Search Agent**: Use the Arxiv API to search for papers related to a given topic and return results.
- **Google Search Agent**: Use the Google Search api to find papers related to a given topic and return results.
- **Report Agent**: Generate a report based on the information collected by the arxviv search and Google search agents.


First, let us import the necessary modules.
"""
logger.info("# Literature Review")


"""
## Defining Tools 

Next, we will define the tools that the agents will use to perform their tasks. In this case we will define a simple function `search_arxiv` that will use the `arxiv` library to search for papers related to a given topic.  

Finally, we will wrap the functions into a `FunctionTool` class that will allow us to use it as a tool in the agents. 

Note: You will need to set the appropriate environment variables for tools as needed.

Also install required libraries: 

```bash
!pip install arxiv
```
"""
logger.info("## Defining Tools")

def google_search(query: str, num_results: int = 2, max_chars: int = 500) -> list:  # type: ignore[type-arg]


    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    if not api_key or not search_engine_id:
        raise ValueError("API key or Search Engine ID not found in environment variables")

    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": search_engine_id, "q": query, "num": num_results}

    response = requests.get(url, params=params)  # type: ignore[arg-type]

    if response.status_code != 200:
        logger.debug(response.json())
        raise Exception(f"Error in API request: {response.status_code}")

    results = response.json().get("items", [])

    def get_page_content(url: str) -> str:
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            words = text.split()
            content = ""
            for word in words:
                if len(content) + len(word) + 1 > max_chars:
                    break
                content += " " + word
            return content.strip()
        except Exception as e:
            logger.debug(f"Error fetching {url}: {str(e)}")
            return ""

    enriched_results = []
    for item in results:
        body = get_page_content(item["link"])
        enriched_results.append(
            {"title": item["title"], "link": item["link"], "snippet": item["snippet"], "body": body}
        )
        time.sleep(1)  # Be respectful to the servers

    return enriched_results


def arxiv_search(query: str, max_results: int = 2) -> list:  # type: ignore[type-arg]
    """
    Search Arxiv for papers and return the results including abstracts.
    """

    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)

    results = []
    for paper in client.results(search):
        results.append(
            {
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "published": paper.published.strftime("%Y-%m-%d"),
                "abstract": paper.summary,
                "pdf_url": paper.pdf_url,
            }
        )


    return results

google_search_tool = FunctionTool(
    google_search, description="Search Google for information, returns results with a snippet and body content"
)
arxiv_search_tool = FunctionTool(
    arxiv_search, description="Search Arxiv for papers related to a given topic, including abstracts"
)

"""
## Defining Agents 

Next, we will define the agents that will perform the tasks.
"""
logger.info("## Defining Agents")

model_client = OllamaChatCompletionClient(model="llama3.1")

google_search_agent = AssistantAgent(
    name="Google_Search_Agent",
    tools=[google_search_tool],
    model_client=model_client,
    description="An agent that can search Google for information, returns results with a snippet and body content",
    system_message="You are a helpful AI assistant. Solve tasks using your tools.",
)

arxiv_search_agent = AssistantAgent(
    name="Arxiv_Search_Agent",
    tools=[arxiv_search_tool],
    model_client=model_client,
    description="An agent that can search Arxiv for papers related to a given topic, including abstracts",
    system_message="You are a helpful AI assistant. Solve tasks using your tools. Specifically, you can take into consideration the user's request and craft a search query that is most likely to return relevant academi papers.",
)


report_agent = AssistantAgent(
    name="Report_Agent",
    model_client=model_client,
    description="Generate a report based on a given topic",
    system_message="You are a helpful assistant. Your task is to synthesize data extracted into a high quality literature review including CORRECT references. You MUST write a final report that is formatted as a literature review with CORRECT references.  Your response should end with the word 'TERMINATE'",
)

"""
## Creating the Team 

Finally, we will create a team of agents and configure them to perform the tasks.
"""
logger.info("## Creating the Team")

termination = TextMentionTermination("TERMINATE")
team = RoundRobinGroupChat(
    participants=[google_search_agent, arxiv_search_agent, report_agent], termination_condition=termination
)

await Console(
    team.run_stream(
        task="Write a literature review on no code tools for building multi agent ai systems",
    )
)

async def run_async_code_0349fda4():
    await model_client.close()
    return 
 = asyncio.run(run_async_code_0349fda4())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)