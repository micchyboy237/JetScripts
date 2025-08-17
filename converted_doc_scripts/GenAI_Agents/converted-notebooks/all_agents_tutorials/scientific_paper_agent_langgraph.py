import asyncio
from jet.transformers.formatters import format_json
from IPython.display import display, Markdown
from dotenv import load_dotenv
from jet.llm.ollama.base_langchain import ChatMLX
from jet.logger import CustomLogger
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from typing import Annotated, ClassVar, Sequence, TypedDict, Optional
import io
import json
import os
import pdfplumber
import shutil
import time
import urllib3


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Scientific paper agent using LangGraph

## Overview



This project implements an intelligent research assistant that helps users navigate, understand, and analyze scientific literature using LangGraph and advanced language models. By combining various academic API with sophisticated paper processing techniques, it creates a seamless experience for researchers, students, and professionals working with academic papers.

> NOTE: The presented workflow is not domain specific: each step in the graph can be adapted to a different domain by simply changing the prompts.

## Motivation

Research literature review represents a significant time investment in R&D, with studies showing that researchers spend 30-50% of their time reading, analyzing, and synthesizing academic papers. This challenge is universal across the research community. While thorough literature review is crucial for advancing science and technology, the current process remains inefficient and time-consuming.

Key challenges include:
- Extensive time commitment (30-50% of R&D hours) dedicated to reading and processing papers
- Inefficient search processes across fragmented database ecosystems
- Complex task of synthesizing and connecting findings across multiple papers
- Resource-intensive maintenance of comprehensive literature reviews
- Ongoing effort required to stay current with new publications

## Key components

 1. State-Driven Workflow Engine 
    - StateGraph Architecture: Five-node system for orchestrated research 
    - Decision Making Node: Query intent analysis and routing 
    - Planning Node: Research strategy formulation
    - Tool Execution Node: Paper retrieval and processing 
    - Judge Node: Quality validation and improvement cycles 

2. Paper Processing Integration 
    - Source Integration, CORE API for comprehensive paper access 
    - Document Processing, PDF content extraction, Text structure preservation 

3. Analysis Workflow 
    - State-aware processing pipeline 
    - Multi-step validation gates 
    - Quality-focused improvement cycles 
    - Human-in-the-loop validation options

An overview of the workflow is shown below:

![image](https://i.ibb.co/0BBzkcb/mermaid-diagram-2024-11-17-195744.png)]

## Method details

1. The system requires 
    - MLX API key to access GPT 4o. This model was chosen after comparing its performance with other, open-source alternatives (in particular Llama 3). However, any other LLM with tool calling capabilities can be used.
    - CORE API key for paper retrieval. CORE is one of the larges online repositories for scientific papers, counting over 136 million papers, and offers a free API for personal use. A key can be requested [here](https://core.ac.uk/services/api#form).

2. Technical Architecture: 
    - LangGraph for state orchestration.
    - PDFplumber for document processing.
    - Pydantic for structured data handling.

> Acknowledgment: Special thanks to CORE API for enabling academic paper access.

---

## Setup

This cell installs the required dependencies.
"""
logger.info("# Scientific paper agent using LangGraph")

# ! pip install --upgrade --quiet langchain==0.2.16 langchain-community==0.2.16 langchain-openai==0.1.23 langgraph==0.2.18 langsmith==0.1.114 pdfplumber python-dotenv

"""
This cell imports the required libraries and sets the environment variables.
"""
logger.info("This cell imports the required libraries and sets the environment variables.")



urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

# os.environ["OPENAI_API_KEY"] = "sk-proj-..."
os.environ["CORE_API_KEY"] = "..."

"""
## Prompts

This cell contains the prompts used in the workflow.

The `agent_prompt` contains a section explaining how to use complex queries with the CORE API, enabling the agent to solve more complex tasks.
"""
logger.info("## Prompts")

decision_making_prompt = """
You are an experienced scientific researcher.
Your goal is to help the user with their scientific research.

Based on the user query, decide if you need to perform a research or if you can answer the question directly.
- You should perform a research if the user query requires any supporting evidence or information.
- You should answer the question directly only for simple conversational questions, like "how are you?".
"""

planning_prompt = """

You are an experienced scientific researcher.
Your goal is to make a new step by step plan to help the user with their scientific research .

Subtasks should not rely on any assumptions or guesses, but only rely on the information provided in the context or look up for any additional information.

If any feedback is provided about a previous answer, incorportate it in your new planning.



For each subtask, indicate the external tool required to complete the subtask.
Tools can be one of the following:
{tools}
"""

agent_prompt = """

You are an experienced scientific researcher.
Your goal is to help the user with their scientific research. You have access to a set of external tools to complete your tasks.
Follow the plan you wrote to successfully complete the task.

Add extensive inline citations to support any claim made in the answer.




The CORE API has a specific query language that allows you to explore a vast papers collection and perform complex queries. See the following table for a list of available operators:

| Operator       | Accepted symbols         | Meaning                                                                                      |
|---------------|-------------------------|----------------------------------------------------------------------------------------------|
| And           | AND, +, space          | Logical binary and.                                                                           |
| Or            | OR                     | Logical binary or.                                                                            |
| Grouping      | (...)                  | Used to prioritise and group elements of the query.                                           |
| Field lookup  | field_name:value       | Used to support lookup of specific fields.                                                    |
| Range queries | fieldName(>, <,>=, <=) | For numeric and date fields, it allows to specify a range of valid values to return.         |
| Exists queries| _exists_:fieldName     | Allows for complex queries, it returns all the items where the field specified by fieldName is not empty. |

Use this table to formulate more complex queries filtering for specific papers, for example publication date/year.
Here are the relevant fields of a paper object you can use to filter the results:
{
  "authors": [{"name": "Last Name, First Name"}],
  "documentType": "presentation" or "research" or "thesis",
  "publishedDate": "2019-08-24T14:15:22Z",
  "title": "Title of the paper",
  "yearPublished": "2019"
}

Example queries:
- "machine learning AND yearPublished:2023"
- "maritime biology AND yearPublished>=2023 AND yearPublished<=2024"
- "cancer research AND authors:Vaswani, Ashish AND authors:Bello, Irwan"
- "title:Attention is all you need"
- "mathematics AND _exists_:abstract"
"""

judge_prompt = """
You are an expert scientific researcher.
Your goal is to review the final answer you provided for a specific user query.

Look at the conversation history between you and the user. Based on it, you need to decide if the final answer is satisfactory or not.

A good final answer should:
- Directly answer the user query. For example, it does not answer a question about a different paper or area of research.
- Answer extensively the request from the user.
- Take into account any feedback given through the conversation.
- Provide inline sources to support any claim made in the answer.

In case the answer is not good enough, provide clear and concise feedback on what needs to be improved to pass the evaluation.
"""

"""
## Utility classes and functions

This cell contains the utility classes and functions used in the workflow. It includes a wrapper around the CORE API, the Pydantic models for the input and output of the nodes, and a few general-purpose functions.

The `CoreAPIWrapper` class includes a retry mechanism to handle transient errors and make the workflow more robust.
"""
logger.info("## Utility classes and functions")

class CoreAPIWrapper(BaseModel):
    """Simple wrapper around the CORE API."""
    base_url: ClassVar[str] = "https://api.core.ac.uk/v3"
    api_key: ClassVar[str] = os.environ["CORE_API_KEY"]

    top_k_results: int = Field(description = "Top k results obtained by running a query on Core", default = 1)

    def _get_search_response(self, query: str) -> dict:
        http = urllib3.PoolManager()

        max_retries = 5
        for attempt in range(max_retries):
            response = http.request(
                'GET',
                f"{self.base_url}/search/outputs",
                headers={"Authorization": f"Bearer {self.api_key}"},
                fields={"q": query, "limit": self.top_k_results}
            )
            if 200 <= response.status < 300:
                return response.json()
            elif attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 2))
            else:
                raise Exception(f"Got non 2xx response from CORE API: {response.status} {response.data}")

    def search(self, query: str) -> str:
        response = self._get_search_response(query)
        results = response.get("results", [])
        if not results:
            return "No relevant results were found"

        docs = []
        for result in results:
            published_date_str = result.get('publishedDate') or result.get('yearPublished', '')
            authors_str = ' and '.join([item['name'] for item in result.get('authors', [])])
            docs.append((
                f"* ID: {result.get('id', '')},\n"
                f"* Title: {result.get('title', '')},\n"
                f"* Published Date: {published_date_str},\n"
                f"* Authors: {authors_str},\n"
                f"* Abstract: {result.get('abstract', '')},\n"
                f"* Paper URLs: {result.get('sourceFulltextUrls') or result.get('downloadUrl', '')}"
            ))
        return "\n-----\n".join(docs)

class SearchPapersInput(BaseModel):
    """Input object to search papers with the CORE API."""
    query: str = Field(description="The query to search for on the selected archive.")
    max_papers: int = Field(description="The maximum number of papers to return. It's default to 1, but you can increase it up to 10 in case you need to perform a more comprehensive search.", default=1, ge=1, le=10)

class DecisionMakingOutput(BaseModel):
    """Output object of the decision making node."""
    requires_research: bool = Field(description="Whether the user query requires research or not.")
    answer: Optional[str] = Field(default=None, description="The answer to the user query. It should be None if the user query requires research, otherwise it should be a direct answer to the user query.")

class JudgeOutput(BaseModel):
    """Output object of the judge node."""
    is_good_answer: bool = Field(description="Whether the answer is good or not.")
    feedback: Optional[str] = Field(default=None, description="Detailed feedback about why the answer is not good. It should be None if the answer is good.")

def format_tools_description(tools: list[BaseTool]) -> str:
    return "\n\n".join([f"- {tool.name}: {tool.description}\n Input arguments: {tool.args}" for tool in tools])

async def print_stream(app: CompiledStateGraph, input: str) -> Optional[BaseMessage]:
    display(Markdown("## New research running"))
    display(Markdown(f"### Input:\n\n{input}\n\n"))
    display(Markdown("### Stream:\n\n"))

    all_messages = []
    for chunk in app.stream({"messages": [input]}, stream_mode="updates"):
        for updates in chunk.values():
            if messages := updates.get("messages"):
                all_messages.extend(messages)
                for message in messages:
                    message.pretty_logger.debug()
                    logger.debug("\n\n")

    if not all_messages:
        return None
    return all_messages[-1]

"""
## Agent state

This cell defines the agent state, which contains the following information:
- `requires_research`: Whether the user query requires research or not.
- `num_feedback_requests`: The number of times the LLM asked for feedback.
- `is_good_answer`: Whether the LLM's final answer is good or not.
- `messages`: The conversation history between the user and the LLM.
"""
logger.info("## Agent state")

class AgentState(TypedDict):
    """The state of the agent during the paper research process."""
    requires_research: bool = False
    num_feedback_requests: int = 0
    is_good_answer: bool = False
    messages: Annotated[Sequence[BaseMessage], add_messages]

"""
## Agent tools

This cell defines the tools available to the agent. The toolkit contains a tool to search for scientific papers using the CORE API, a tool to download a scientific paper from a given URL, and a tool to ask for human feedback.

To make the paper download more robust, the tool includes a retry mechanism, similar to the one used for the CORE API, as well as a mock browser header to avoid 403 errors.
"""
logger.info("## Agent tools")

@tool("search-papers", args_schema=SearchPapersInput)
def search_papers(query: str, max_papers: int = 1) -> str:
    """Search for scientific papers using the CORE API.

    Example:
    {"query": "Attention is all you need", "max_papers": 1}

    Returns:
        A list of the relevant papers found with the corresponding relevant information.
    """
    try:
        return CoreAPIWrapper(top_k_results=max_papers).search(query)
    except Exception as e:
        return f"Error performing paper search: {e}"

@tool("download-paper")
def download_paper(url: str) -> str:
    """Download a specific scientific paper from a given URL.

    Example:
    {"url": "https://sample.pdf"}

    Returns:
        The paper content.
    """
    try:
        http = urllib3.PoolManager(
            cert_reqs='CERT_NONE',
        )

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }
        max_retries = 5
        for attempt in range(max_retries):
            response = http.request('GET', url, headers=headers)
            if 200 <= response.status < 300:
                pdf_file = io.BytesIO(response.data)
                with pdfplumber.open(pdf_file) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() + "\n"
                return text
            elif attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 2))
            else:
                raise Exception(f"Got non 2xx when downloading paper: {response.status_code} {response.text}")
    except Exception as e:
        return f"Error downloading paper: {e}"

@tool("ask-human-feedback")
def ask_human_feedback(question: str) -> str:
    """Ask for human feedback. You should call this tool when encountering unexpected errors."""
    return input(question)

tools = [search_papers, download_paper, ask_human_feedback]
tools_dict = {tool.name: tool for tool in tools}

"""
## Workflow nodes

This cell defines the nodes of the workflow. Note how the `judge_node` is configured to end the execution if the LLM failed to provide a good answer twice to keep latency acceptable.
"""
logger.info("## Workflow nodes")

base_llm = ChatMLX(model="llama-3.2-3b-instruct", log_dir=f"{OUTPUT_DIR}/chats", temperature=0.0)
decision_making_llm = base_llm.with_structured_output(DecisionMakingOutput)
agent_llm = base_llm.bind_tools(tools)
judge_llm = base_llm.with_structured_output(JudgeOutput)

def decision_making_node(state: AgentState):
    """Entry point of the workflow. Based on the user query, the model can either respond directly or perform a full research, routing the workflow to the planning node"""
    system_prompt = SystemMessage(content=decision_making_prompt)
    response: DecisionMakingOutput = decision_making_llm.invoke([system_prompt] + state["messages"])
    output = {"requires_research": response.requires_research}
    if response.answer:
        output["messages"] = [AIMessage(content=response.answer)]
    return output

def router(state: AgentState):
    """Router directing the user query to the appropriate branch of the workflow."""
    if state["requires_research"]:
        return "planning"
    else:
        return "end"

def planning_node(state: AgentState):
    """Planning node that creates a step by step plan to answer the user query."""
    system_prompt = SystemMessage(content=planning_prompt.format(tools=format_tools_description(tools)))
    response = base_llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def tools_node(state: AgentState):
    """Tool call node that executes the tools based on the plan."""
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_dict[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}

def agent_node(state: AgentState):
    """Agent call node that uses the LLM with tools to answer the user query."""
    system_prompt = SystemMessage(content=agent_prompt)
    response = agent_llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    """Check if the agent should continue or end."""
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "continue"
    else:
        return "end"

def judge_node(state: AgentState):
    """Node to let the LLM judge the quality of its own final answer."""
    num_feedback_requests = state.get("num_feedback_requests", 0)
    if num_feedback_requests >= 2:
        return {"is_good_answer": True}

    system_prompt = SystemMessage(content=judge_prompt)
    response: JudgeOutput = judge_llm.invoke([system_prompt] + state["messages"])
    output = {
        "is_good_answer": response.is_good_answer,
        "num_feedback_requests": num_feedback_requests + 1
    }
    if response.feedback:
        output["messages"] = [AIMessage(content=response.feedback)]
    return output

def final_answer_router(state: AgentState):
    """Router to end the workflow or improve the answer."""
    if state["is_good_answer"]:
        return "end"
    else:
        return "planning"

"""
## Workflow definition

This cell defines the workflow using LangGraph.
"""
logger.info("## Workflow definition")

workflow = StateGraph(AgentState)

workflow.add_node("decision_making", decision_making_node)
workflow.add_node("planning", planning_node)
workflow.add_node("tools", tools_node)
workflow.add_node("agent", agent_node)
workflow.add_node("judge", judge_node)

workflow.set_entry_point("decision_making")

workflow.add_conditional_edges(
    "decision_making",
    router,
    {
        "planning": "planning",
        "end": END,
    }
)
workflow.add_edge("planning", "agent")
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": "judge",
    },
)
workflow.add_conditional_edges(
    "judge",
    final_answer_router,
    {
        "planning": "planning",
        "end": END,
    }
)

app = workflow.compile()

"""
## Example usecase for PhD academic research

This cell tests the workflow with several example queries. These queries are designed to evaluate the agent on the following aspects:
- Completing tasks that are representative of the work a PhD researcher might need to perform.
- Addressing more specific tasks that require researching papers within a defined timeframe.
- Tackling tasks across multiple areas of research.
- Critically evaluating its own responses by sourcing specific information from the papers.
"""
logger.info("## Example usecase for PhD academic research")

test_inputs = [
    "Download and summarize the findings of this paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11379842/pdf/11671_2024_Article_4070.pdf",

    "Can you find 8 papers on quantum machine learning?",

    """Find recent papers (2023-2024) about CRISPR applications in treating genetic disorders,
    focusing on clinical trials and safety protocols""",

    """Find and analyze papers from 2023-2024 about the application of transformer architectures in protein folding prediction,
    specifically looking for novel architectural modifications with experimental validation."""
]

outputs = []
for test_input in test_inputs:
    async def run_async_code_d236c8f3():
        async def run_async_code_a57dbaf3():
            final_answer = await print_stream(app, test_input)
            return final_answer
        final_answer = asyncio.run(run_async_code_a57dbaf3())
        logger.success(format_json(final_answer))
        return final_answer
    final_answer = asyncio.run(run_async_code_d236c8f3())
    logger.success(format_json(final_answer))
    outputs.append(final_answer.content)

"""
# Display results

This cell displays the results of the test queries for a more compact visualisation of the results.
"""
logger.info("# Display results")

for input, output in zip(test_inputs, outputs):
    display(Markdown(f"## Input:\n\n{input}\n\n"))
    display(Markdown(f"## Output:\n\n{output}\n\n"))

"""
---

## Comparative Analysis

In this comprehensive analysis, we evaluated our Scientific Paper Agent against two leading AI knowledge co pilots : Microsoft Copilot and Perplexity AI. Using a standardized query - "Find 8 papers on quantum machine learning" - we conducted a detailed comparison across multiple dimensions to understand the strengths, limitations, and optimal use cases for each system.


#### Test Case Implementation

We implemented a controlled test using the same research query across all three platforms:
- Query: "Find 8 papers on quantum machine learning"
- Sample Size: Multiple test runs to ensure consistency
- Evaluation Time: Early 2024
- Metrics Tracked: Response time, metadata quality, and result structure

#### Key Findings

While our agent demonstrated superior academic rigor and metadata completeness, taking approximately 30 seconds per query, competitors like Microsoft Copilot (2 seconds) and Perplexity AI (4-5 seconds) showed advantages in response speed. This tradeoff between speed and depth reflects different design philosophies and target use cases.

The comparative analysis reveals a clear differentiation in approaches:
- Our Agent: Optimized for thorough academic research with comprehensive validation
- Microsoft Copilot: Focused on rapid information retrieval and general overview
- Perplexity AI: Balanced approach with emphasis on source verification

### Microsoft  copilot results 

![image](https://i.ibb.co/y4Zf4Pc/Screenshot-2024-11-17-at-21-40-21.png)]

### Perplexity AI results
![image](https://i.ibb.co/n1rr7kW/Screenshot-2024-11-17-at-21-40-42.png)]

### Metrics Comparsion
![image](https://i.ibb.co/5KbTmFq/Screenshot-2024-11-17-at-22-03-43.png)

Here we present a comprehensive comparison between our research assistant agent and leading platforms (Microsoft Copilot and Perplexity AI). Using a standardized query - "Find 8 papers on quantum machine learning" - we evaluated performance across key metrics including response time, metadata quality, and academic value. Our analysis reveals distinct trade-offs: while our agent takes longer to process (30s vs. 2-5s), it provides significantly more detailed metadata, validated sources, and structured academic output. The comparison table above breaks down these differences across multiple dimensions, helping users choose the right tool for their specific research needs - whether it's quick exploration (where Copilot excels) or deep academic research (where our agent shows its strengths).

---

##  Limitations

1. Technical Limitations
    - API rate limits for paper access
    - Handle time for large PDFs
    - Limited to publicly accessible papers
  
2. Functional Limitations
    - No support for image analysis in papers
    - Limited context window for very long papers
    - Cannot perform mathematical derivations
    - Language constraints for non-English papers


## Potential Improvements:

1. Technical Improvements
    - Implement parallel processing for multiple papers
    - Add caching system for frequently accessed papers
    - Integrate multiple academic APIs for broader coverage
    - Implement batch processing for large datasets

2. Functional Improvements
    - Add support for figure and table extraction
    - Implement cross-referencing between papers
    - Add citation network analysis
    - Include domain-specific validation rules
        
3. User Experience
    - Add interactive feedback mechanisms
    - Implement progress tracking
    - Add customizable validation criteria
    - Include export options for research summaries
        
   
## Specific Use Cases:

1. Academic Research, Literature review and paper analysis.
    - Comprehensive search
    - Citation tracking
    - Cross-reference validation

2. Industry Research, Technical documentation and patent analysis.
    - Focused search
    - Technical specification extraction
    - Competitive analysis

3. Educational, Student research assistance.
    - Simplified explanations
    - Learning resource identification
    - Guided research process

---

## Conclusion

This implementation demonstrates how state-driven architectures can transform academic paper analysis. By combining LangGraph's orchestration capabilities with robust API integrations, we've created a system that maintains research rigor while automating key aspects of paper processing. The workflow's emphasis on validation and quality control ensures reliable research outputs while significantly streamlining the paper analysis process.
"""
logger.info("## Comparative Analysis")

logger.info("\n\n[DONE]", bright=True)