from requests.exceptions import HTTPError
from IPython.display import Image, display, Markdown
from bs4 import BeautifulSoup
from dotenv import load_dotenv
# from google.colab import userdata
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.file.utils import save_file
from jet.logger import CustomLogger
from jet.visualization.langchain.mermaid_graph import render_mermaid_graph
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage, ChatMessage
from langchain_core.tools import BaseTool
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.memory import MemorySaver
# from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph, END
from psycopg import Connection
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from typing import TypedDict, Annotated, List, Dict, Any, Type
from uuid import uuid4
import ast
import ollama
import operator
import os
import pypdf
import requests
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

# New configurable fallback (easy to extend/test)
FALLBACK_URLS: Dict[str, str] = {
    "https://www.sciencedirect.com/science/article/pii/B9780128139324000055": "https://arxiv.org/pdf/2305.10431.pdf",
    "https://ieeexplore.ieee.org/document/9474391": "https://arxiv.org/pdf/2402.09353.pdf",
    # Add more as needed, e.g., "bonus": "https://arxiv.org/pdf/2501.00001.pdf"
}

# !pip install gradio grandalf huggingface-hub langchain langchain-community langchain-core langchain-ollama langgraph langgraph-checkpoint langgraph-checkpoint-postgres langgraph-checkpoint-sqlite langsmith ollama psycopg pydantic pydantic_core tiktoken langchain-huggingface pypdf

"""
# Systematic Review Automation System

A tool for automated academic literature review and synthesis

## Introduction
This system automates the process of creating systematic reviews of academic papers through a structured workflow. It handles everything from initial paper search to final draft generation using a directed graph architecture.

## Use Cases

**Primary Applications**
- Conducting systematic literature reviews
- Analyzing research trends across papers
- Synthesizing findings from multiple studies
- Creating comprehensive research summaries

**Key Features**
- Automated paper search and selection
- PDF download and analysis
- Section-by-section writing
- Revision and critique cycles

## Process Flow

1. **Research Phase**
- Topic planning and scoping
- Automated paper search via Semantic Scholar
- Smart paper selection (up to 3 papers - can be changed)
- Automatic PDF retrieval

2. **Analysis Phase**
- PDF text extraction
- Section-by-section analysis
- Key finding identification
- Cross-paper comparison

3. **Writing Phase**
- Automated section generation
- Abstract (100-word limit)
- Methods comparison
- Results synthesis
- APA reference formatting

4. **Review Phase**
- Quality assessment
- Revision suggestions
- Additional research triggers
- Final draft preparation

The system uses Ollama's GPT models for text processing and maintains state through a graph-based workflow, ensuring systematic and thorough review generation.

## I. Flowchart Components Description

### Nodes (Process Steps)

1. **Initial Stages**
- `_start_`: Beginning point of the process
- `process_input`: Initial data processing stage
- `planner`: Strategy development phase
- `researcher`: Research coordination phase

2. **Article Management**
- `search_articles`: Article search and identification
- `article_decisions`: Evaluation and selection of articles
- `download_articles`: Retrieval of selected articles
- `paper_analyzer`: In-depth analysis of papers

3. **Writing Components**
- `write_abstract`: Abstract composition
- `write_conclusion`: Conclusion development
- `write_introduction`: Introduction creation
- `write_methods`: Methodology documentation
- `write_references`: Reference compilation
- `write_results`: Results documentation

4. **Final Stages**
- `aggregate_paper`: Combining all sections
- `critique_paper`: Critical review phase
- `revise_paper`: Revision process
- `final_draft`: Final document preparation
- `_end_`: Process completion

## Edges (Connections)

1. **Main Flow**
- Solid arrows indicate direct progression between steps
- Sequential flow from start through research phases
- Parallel paths from paper_analyzer to writing components

2. **Special Connections**
- Dotted line with "True" label: Feedback loop to search_articles
- "revise" connection: Loop between critique_paper and revise_paper
- Multiple converging arrows into aggregate_paper from all writing components

3. **Decision Points**
- Branching at paper_analyzer to multiple writing tasks
- Convergence at aggregate_paper from all writing components
- Split path at critique_paper leading to either revision or final draft

## II. Imports
- if you have postgres set up you can use that
- we will use the MemorySaver() to store memory state
"""
logger.info("# Systematic Review Automation System")

# os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')


_ = load_dotenv()

"""
## III. Academic Search Tool

This tool helps researchers and students search for academic papers efficiently. It connects to the Semantic Scholar API and returns structured paper information.

### Main Components

**Input Parameters**
- `topic`: Your research subject
- `max_results`: Number of papers to retrieve (default: 20)

**Output Format**
Each paper result includes:
- Title
- Abstract
- Author list
- Publication year
- PDF link (if openly accessible)

## Key Features

**Search Capabilities**
- Connects to Semantic Scholar's database
- Filters for open access papers
- Returns structured, easy-to-process results

## Notes
- Only returns open access papers
- Async operations not currently supported
- Requires valid API connection
- Results are paginated for efficiency

This tool simplifies academic research by providing structured access to scholarly papers while handling common search and retrieval challenges automatically.
"""
logger.info("## III. Academic Search Tool")


class AcademicPaperSearchInput(BaseModel):
    topic: str = Field(...,
                       description="The topic to search for academic papers on")
    max_results: int = Field(
        20, description="Maximum number of results to return")


class AcademicPaperSearchTool(BaseTool):
    args_schema: type = AcademicPaperSearchInput  # Explicit type annotation
    name: str = Field("academic_paper_search_tool",
                      description="Tool for searching academic papers")
    description: str = Field(
        "Queries an academic papers API to retrieve relevant articles based on a topic")

    def __init__(self, name: str = "academic_paper_search_tool",
                 description: str = "Queries an academic paper API to retrieve relevant articles based on a topic"):
        super().__init__()
        self.name = name
        self.description = description

    def _run(self, topic: str, max_results: int) -> List[Dict[str, Any]]:
        search_results = self.query_academic_api(topic, max_results)

        return search_results

    async def _arun(self, topic: str, max_results: int) -> List[Dict[str, Any]]:
        raise NotImplementedError("Async version not implemented")

    def query_academic_api(self, topic: str, max_results: int) -> List[Dict[str, Any]]:
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": topic,
            "limit": max_results,  # max_results
            "fields": "title,abstract,authors,year,openAccessPdf",
            "openAccessPdf": True
        }
        try:
            while True:
                try:
                    response = requests.get(base_url, params=params)
                    logger.debug(response)

                    if response.status_code == 200:
                        papers = response.json().get("data", [])
                        formatted_results = [
                            {
                                "title": paper.get("title"),
                                "abstract": paper.get("abstract"),
                                "authors": [author.get("name") for author in paper.get("authors", [])],
                                "year": paper.get("year"),
                                "pdf": paper.get("openAccessPdf"),
                            }
                            for paper in papers
                        ]

                        return formatted_results
                except:
                    logger.debug(
                        (f"Failed to fetch papers: {response.status_code} - {response.text}. Trying Again..."))
        except KeyboardInterrupt:
            logger.debug("\nOperation cancelled by user")
            sys.exit(0)  # Clean exit


"""
## IV. Prompts

### Planning Phase Prompts

**Planner Prompt**
- Acts as initial architect of the review
- Sets up structure based on standard academic components
- Creates outline without conducting actual research
- Focuses on organization and methodology planning

**Research Prompt**
- Generates 5 targeted search queries
- Uses project plan to guide search strategy
- Interfaces with academic paper search tool
- Ensures comprehensive literature coverage

**Decision Prompt**
- Evaluates search results against project plan
- Selects top 3 most relevant papers - this number can be changed as you see fit!
- Returns only PDF URLs in JSON format
- Streamlines paper selection process

## Analysis Phase Prompts

**Analyze Paper Prompt**
- Breaks down papers into key sections
- Provides section-specific analysis:
  - Abstract: Key points
  - Introduction: Research motivation
  - Methods: Technical details and mathematical analysis
  - Results: Statistical findings
  - Conclusions: Analysis and counterarguments
- Includes metadata (title, year, authors, URL)

## Writing Phase Prompts

**Section-Specific Prompts:**
- Abstract (100-word limit, overview)
- Introduction (comprehensive background)
- Methods (comparative analysis of approaches)
- Results (cross-paper comparison)
- Conclusions (synthesis and future directions)
- References (APA formatting)

## Review Phase Prompts

**Critique Draft Prompt**
- Evaluates publication readiness
- Provides specific revision recommendations
- Assesses need for additional research
- Makes go/no-go publication decisions

**Revise Draft Prompt**
- Implements recommended changes
- Refines paper based on critique
- Ensures all feedback is addressed
- Produces final manuscript version

Each prompt works sequentially to build a comprehensive systematic review, from initial planning to final publication-ready manuscript.
"""
logger.info("## IV. Prompts")

planner_prompt = '''You are an academic researcher that is planning to write a systematic review of Academic and Scientific Research Papers.

A systematic review article typically includes the following components:
Title: The title should accurately reflect the topic being reviewed, and usually includes the words "a systematic review".
Abstract: A structured abstract with a short paragraph for each of the following: background, methods, results, and conclusion.
Introduction: Summarizes the topic, explains why the review was conducted, and states the review's purpose and aims.
Methods: Describes the methods used in the review.
Results: Presents the results of the review.
Discussion: Discusses the results of the review.
References: Lists the references used in the review.

Other important components of a systematic review include:
Scoping: A "trial run" of the review that helps shape the review's method and protocol.
Meta-analysis: An optional component that uses statistical methods to combine and summarize the results of multiple studies.
Data extraction: A central component where data is collected and organized for analysis.
Assessing the risk of bias: Helps establish transparency of evidence synthesis results.
Interpreting results: Involves considering factors such as limitations, strength of evidence, biases, and implications for future practice or research.
Literature identification: An important component that sets the data to be analyzed.

With this in mind, only create an outline plan based on the topic. Don't search anything, just set up the planning.
'''

research_prompt = '''You are an academic researcher that is searching Academic and Scientific Research Papers.

You will be given a project plan. Based on the project plan, generate 5 queries that you will use to search the papers.

Send the queries to the academic_paper_search_tool as a tool call.
'''

decision_prompt = '''You are an academic researcher that is searching Academic and Scientific Research Papers.

You will be given a project plan and a list of articles.

Based on the project plan and articles provided, you must choose a maximum of 3 to investigate that are most relevant to that plan.

IMPORTANT: You must return ONLY a JSON array of the PDF URLs with no additional text or explanation. Your entire response should be in this exact format:

[
    "url1",
    "url2",
    "url3",
    ...
]

Do not include any other text, explanations, or formatting.'''

analyze_paper_prompt = '''You are an academic researcher trying to understand the details of scientific and academic research papers.

You must look through the text provided and get the details from the Abstract, Introduction, Methods, Results, and Conclusions.
If you are in an Abstract section, just give me the condensed thoughts.
If you are in an Introduction section, give me a concise reason on why the research was done.
If you are in a Methods section, give me low-level details of the approach. Analyze the math and tell me what it means.
If you are in a Results section, give me low-level relevant objective statistics. Tie it in with the methods
If you are in a Conclusions section, give me the fellow researcher's thoughts, but also come up with a counter-argument if none are given.

Remember to attach the other information to the top:
    Title : <title>
    Year : <year>
    Authors : <author1, author2, etc.>
    URL : <pdf url>
    TLDR Analysis:
        <your analysis>
'''

abstract_prompt = '''You are an academic researcher that is writing a systematic review of Academic and Scientific Research Papers.
You are tasked with writing the Abstract section of the paper based on the systematic outline and the analyses given.
Make the abstract no more than 100 words.
'''

introduction_prompt = '''You are an academic researcher that is writing a systematic review of Academic and Scientific Research Papers.
You are tasked with writing the Introduction section of the paper based on the systematic outline and the analyses given.
Make sure it is thorough and covers information in all the papers.
'''

methods_prompt = '''You are an academic researcher that is writing a systematic review of Academic and Scientific Research Papers.
You are tasked with writing the Methods section of the paper based on the systematic outline and the analyses given.
Make sure it is thorough and covers information in all the papers. Draw on the differences and similarities in approaches in each paper.
'''

results_prompt = '''You are an academic researcher that is writing a systematic review of Academic and Scientific Research Papers.
You are tasked with writing the Results section of the paper based on the systematic outline and the analyses given.
Make sure it is thorough and covers information in all the papers. If there are results to compare among papers, please do so.
'''

conclusions_prompt = '''You are an academic researcher that is writing a systematic review of Academic and Scientific Research Papers.
You are tasked with writing the Conclusions section of the paper based on the systematic outline and the analyses given.
Make sure it is thorough and covers information in all the papers.
Draw on the conclusions from other papers, and what you might think the future of the research holds.
'''

references_prompt = '''You are an academic researcher that is writing a systematic review of Academic and Scientific Research Papers.
You are tasked with writing the References section of the paper based on the systematic outline and the analyses given.
Construct an APA style references list
'''
critique_draft_prompt = """You are an academic researcher deciding whether or not a systematic review should be published.
Generate a critique and recommendations for the author's submission or generate a query to get more papers.

If you think just a revision needs to be made, provide detailed recommendations, including requests for length, depth, style.
If you think the paper is good as is, just end with the draft unchanged.
"""


revise_draft_prompt = """You are an academic researcher that is revising a systematic review that is about to be published.
Given the paper below, revise it following the recommendations given.

Return the revised paper with the implemented recommended changes.
"""

"""
## V. Understanding Agent State

### Core Components

**Message Management**
- `messages`: Tracks conversation history
- `last_human_index`: Keeps track of user interaction points
- `systematic_review_outline`: Stores the overall review structure

**Paper Processing**
- `papers`: List of downloaded papers for review
- `analyses`: Collection of individual paper analyses
- `combined_analysis`: Synthesized findings from all papers

## Document Sections
Each section is stored separately for flexible editing:
- `title`: Paper's main title
- `abstract`: Brief summary
- `introduction`: Background and context
- `methods`: Research methodology
- `results`: Research findings
- `conclusion`: Final interpretations
- `references`: Citation list

## Revision Control
- `draft`: Current version of the paper
- `revision_num`: Tracks revision iterations
- `max_revisions`: Limits revision cycles

## Special Features

**Type Annotations**
- `Annotated[List[str], operator.add]`: Allows list concatenation
- `Annotated[list[AnyMessage], reduce_messages]`: Manages message history
- `TypedDict`: Ensures type safety for all fields

## Usage Notes
- Each field maintains its own state
- Sections can be updated independently
- Revision tracking prevents infinite loops
- Message history helps maintain context
- Lists can be combined using operator.add

This state management system helps track all components of a systematic review from initial research through final revision.
"""
logger.info("## V. Understanding Agent State")


def reduce_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    for message in right:
        if not message.id:
            message.id = str(uuid4())
    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            if existing.id == message.id:
                merged[i] = message
                break
        else:
            merged.append(message)
    return merged


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_messages]
    systematic_review_outline: str
    last_human_index: int
    papers: Annotated[List[str], operator.add]  # papers downloaded
    analyses: Annotated[List[Dict], operator.add]  # Store analysis results
    combined_analysis: str  # Final combined analysis

    title: str
    abstract: str
    introduction: str
    methods: str
    results: str
    conclusion: str
    references: str

    draft: str
    revision_num: int
    max_revisions: int


"""
## VI. Creating Graph Components

### A. Message Processing Functions Explained

#### Process Input Function
```python
def process_input(state: AgentState)
```
Sets up initial conversation state:
- Sets revision limit to 2
- Finds last human message in chat history
- Returns initial settings (last_human_index, max_revisions, revision_num)

## Message Filter Function
```python
def get_relevant_messages(state: AgentState)
```
Cleans up conversation history by:
- Keeping non-empty human messages
- Keeping completed AI responses
- Removing system and tool messages
- Preserving conversation flow from last human input

Both functions help maintain clean conversation state and prepare messages for the systematic review process.
"""
logger.info("## VI. Creating Graph Components")


def process_input(state: AgentState):
    max_revision = 2
    messages = state.get('messages', [])

    last_human_index = len(messages) - 1
    for i in reversed(range(len(messages))):
        if isinstance(messages[i], HumanMessage):
            last_human_index = i
            break

    return {"last_human_index": last_human_index, "max_revisions": max_revision, "revision_num": 1}


def get_relevant_messages(state: AgentState) -> List[AnyMessage]:
    '''
    Don't get tool call messages for AI from history.
    Get state from everything up to the most recent human message
    '''
    messages = state['messages']
    filtered_history = []
    for message in messages:
        if isinstance(message, HumanMessage) and message.content != "":
            filtered_history.append(message)
        elif isinstance(message, AIMessage) and message.content != "":
            # Check if response_metadata exists and has finish_reason
            metadata = getattr(message, 'response_metadata', {})
            logger.debug(f"AIMessage metadata: {metadata}")
            if not metadata or 'finish_reason' not in metadata or metadata['finish_reason'] == "stop":
                filtered_history.append(message)
    last_human_index = state['last_human_index']
    return filtered_history[:-1] + messages[last_human_index:]


"""
### B. Planning and Research Node Functions

#### Plan Node
```python
def plan_node(state: AgentState)
```
Creates initial review outline:
- Gets filtered conversation history
- Uses planner prompt with system message
- Generates systematic review structure
- Returns outline in state dictionary

## Research Node
```python
def research_node(state: AgentState)
```
Develops research strategy:
- Takes review outline from state
- Applies research prompt
- Generates search queries
- Updates message history

**Common Elements**
- Both use temperature parameter for response variation
- Print progress to console
- Return updated state components
- Use model.invoke for AI responses

These nodes represent the initial planning and research strategy phases in the systematic review flowchart, setting up the foundation for article searching and analysis.
"""
logger.info("### B. Planning and Research Node Functions")


def plan_node(state: AgentState):
    logger.debug("PLANNER")
    relevant_messages = get_relevant_messages(state)
    messages = [SystemMessage(content=planner_prompt)] + relevant_messages
    response = model.invoke(messages, temperature=temperature)
    logger.debug(response)
    logger.newline()
    return {"systematic_review_outline": [response]}


def research_node(state: AgentState):
    logger.debug("RESEARCHER")
    review_plan = state['systematic_review_outline']
    messages = [SystemMessage(content=research_prompt)] + review_plan
    response = model.invoke(messages, temperature=temperature)
    logger.debug(response)
    logger.newline()
    return {"messages": [response]}


"""
### C. Search Node Functions

#### Take Action Node
```python
def take_action(state: AgentState)
```
Handles tool execution:
- Gets last message from state
- Checks for tool calls
- Executes requested tools
- Returns results as tool messages
- Handles invalid tool requests

## Decision Node
```python
def decision_node(state: AgentState)
```
Makes paper selection:
- Uses review plan and message history
- Applies decision prompt
- Evaluates paper relevance
- Returns selection decisions

## Article Download Node
```python
def article_download(state: AgentState)
```
Manages paper downloads:
- Takes URLs from decisions
- Creates 'papers' directory
- Downloads PDFs
- Handles download errors
- Returns file information

**Common Features**
- Error handling throughout
- Progress logging
- State management
- Structured returns
- Message formatting

These nodes represent the paper selection and acquisition phase of the systematic review process, bridging planning and analysis stages.
"""
logger.info("### C. Search Node Functions")


def take_action(state: AgentState):
    ''' Get last message from agent state.
    If we get to this state, the language model wanted to use a tool.
    The tool calls attribute will be attached to message in the Agent State. Can be a list of tool calls.
    Find relevant tool and invoke it, passing in the arguments
    '''
    logger.debug("GET SEARCH RESULTS")
    last_message = state["messages"][-1]
    tool_log_file = os.path.join(LOG_DIR, "tool_calls.log")
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        logger.debug("No tool calls found in last message",
                     log_file=tool_log_file)
        return {"messages": state['messages']}
    results = []
    for t in last_message.tool_calls:
        logger.pretty(
            {"tool_call_request": {"name": t['name'], "args": t['args']}},
            log_file=tool_log_file
        )
        if not t['name'] in tools:
            logger.debug(f"Bad tool name: {t['name']}", log_file=tool_log_file)
            result = "bad tool name, retry"
        else:
            result = tools[t['name']].invoke(t['args'])
            logger.pretty(
                {"tool_call_response": {"name": t['name'], "result": result}},
                log_file=tool_log_file
            )
        results.append(ToolMessage(
            tool_call_id=t['id'], name=t['name'], content=str(result)))
    return {"messages": results}


def decision_node(state: AgentState):
    logger.debug("DECISION-MAKER")
    review_plan = state['systematic_review_outline']
    relevant_messages = get_relevant_messages(state)
    messages = [SystemMessage(content=decision_prompt)] + \
        review_plan + relevant_messages
    response = model.invoke(messages, temperature=temperature)
    logger.debug(response)
    logger.newline()
    return {"messages": [response]}


def article_download(state: AgentState):
    logger.debug("DOWNLOAD PAPERS")
    last_message = state["messages"][-1]
    try:
        if isinstance(last_message.content, str):
            urls = ast.literal_eval(last_message.content)
        else:
            urls = last_message.content
        filenames = []
        data_dir = os.path.join(OUTPUT_DIR, "data")
        os.makedirs(data_dir, exist_ok=True)
        for url in urls:
            downloaded = False
            # Try original URL first
            for attempt_url in [url] + [FALLBACK_URLS.get(url, "")]:
                if not attempt_url:
                    break
                try:
                    # Use headers to mimic a browser request
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
                    }
                    response = requests.get(
                        attempt_url, headers=headers, timeout=10)
                    response.raise_for_status()

                    # Validate content is likely a PDF
                    content_type = response.headers.get(
                        "Content-Type", "").lower()
                    if "pdf" not in content_type:
                        logger.debug(
                            f"Error downloading {attempt_url}: Content-Type is {content_type}, not a PDF")
                        continue

                    filename = os.path.join(
                        data_dir, os.path.basename(attempt_url))
                    if not filename.endswith(".pdf"):
                        filename += ".pdf"

                    # Save file temporarily to validate
                    temp_filename = filename + ".tmp"
                    with open(temp_filename, "wb") as f:
                        f.write(response.content)

                    # Validate PDF integrity
                    try:
                        with open(temp_filename, "rb") as f:
                            pdf_reader = pypdf.PdfReader(f)
                            pdf_reader.pages  # Attempt to access pages to ensure valid PDF
                        os.rename(temp_filename, filename)
                        filenames.append(
                            {"paper": filename, "source_url": attempt_url})
                        logger.debug(
                            f"Successfully downloaded and validated: {filename} (from {attempt_url})")
                        downloaded = True
                        break  # Success, move to next URL
                    except Exception as e:
                        logger.debug(
                            f"Error validating PDF {attempt_url}: {str(e)}")
                        os.remove(temp_filename) if os.path.exists(
                            temp_filename) else None
                        continue
                except requests.exceptions.RequestException as e:
                    logger.debug(f"Error downloading {attempt_url}: {str(e)}")
                    continue
            if not downloaded:
                logger.debug(f"All attempts failed for {url}")
                continue
        if not filenames:
            return {
                "messages": [
                    AIMessage(
                        content="No valid PDFs could be downloaded, even with fallbacks.",
                        response_metadata={"finish_reason": "error"}
                    )
                ]
            }
        return {
            "papers": [
                AIMessage(
                    content=filenames,
                    response_metadata={"finish_reason": "stop"}
                )
            ]
        }
    except Exception as e:
        return {
            "messages": [
                AIMessage(
                    content=f"Error processing downloads: {str(e)}",
                    response_metadata={"finish_reason": "error"}
                )
            ]
        }


"""
### D. Paper Analyzer Function Explained

#### Function Overview
```python
def paper_analyzer(state: AgentState)
```

**Purpose**: Analyzes downloaded academic papers and extracts key information.

## Key Operations

1. **Paper Processing**
- Iterates through downloaded papers
- Converts PDFs to markdown using pymupdf4llm
- Processes each paper individually

2. **Analysis Setup**
- Creates system message with analysis prompt
- Adds paper content as human message
- Uses GPT-4 model for analysis
- Sets low temperature (0.1) for consistent results

3. **Output Handling**
- Accumulates analyses for all papers
- Returns combined analysis in state format
- Maintains analysis history
"""
logger.info("### D. Paper Analyzer Function Explained")


def paper_analyzer(state: AgentState):
    logger.debug("ANALYZE PAPERS")
    analyses = ""
    for paper in state['papers'][-1].content:
        # Open PDF file and extract text using pypdf
        with open(paper['paper'], 'rb') as file:  # Use paper['paper'] directly
            pdf_reader = pypdf.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        messages = [
            SystemMessage(content=analyze_paper_prompt),
            HumanMessage(content=text)
        ]

        model = ChatOllama(model="llama3.2")
        response = model.invoke(messages, temperature=0.1)
        logger.debug(response)
        analyses += response.content
    return {
        "analyses": [analyses]
    }


"""
### E. Paper Writing Functions Explained

#### API Call Handler
```python
def _make_api_call(model, messages, temperature=0.1)
```
- Manages API calls with retry logic
- Handles rate limiting
- Uses exponential backoff
- Maximum 5 retry attempts

## Section Writing Functions
All section writers follow similar pattern:

**Common Structure**
- Takes state with review plan and analyses
- Uses section-specific prompt
- Uses GPT-4 mini model
- Returns section content
- Handles API calls safely

**Individual Functions**
1. `write_abstract`
   - Creates concise summary
   - Uses abstract prompt

2. `write_introduction`
   - Sets research context
   - Uses introduction prompt

3. `write_methods`
   - Details methodology
   - Uses methods prompt

4. `write_results`
   - Presents findings
   - Uses results prompt

5. `write_conclusion`
   - Summarizes implications
   - Uses conclusions prompt

6. `write_references`
   - Formats citations
   - Uses references prompt

Each function:
- Prints progress
- Uses temperature 0.1
- Returns section in state format
- Handles API communication safely
"""
logger.info("### E. Paper Writing Functions Explained")


def _make_api_call(model, messages, temperature=0.1):
    @retry(
        retry=retry_if_exception_type(HTTPError),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5)
    )
    def _call():
        try:
            response = model.invoke(messages, temperature=temperature)
            return response
        except HTTPError as e:
            if e.response.status_code == 429:
                logger.debug(
                    f"Rate limit reached. Waiting before retry... ({e})")
                raise
            else:
                logger.debug(f"Non-rate-limit HTTP error: {str(e)}")
                raise
        except Exception as e:
            logger.debug(f"Unexpected error in API call: {str(e)}")
            raise
    return _call()


def write_abstract(state: AgentState):
    logger.debug("WRITE ABSTRACT")
    review_plan = state['systematic_review_outline']
    analyses = state['analyses']
    messages = [SystemMessage(content=abstract_prompt)
                ] + review_plan + analyses
    model = ChatOllama(model="llama3.2")
    response = _make_api_call(model, messages)
    logger.debug(response)
    logger.newline()
    return {"abstract": [response]}


def write_introduction(state: AgentState):
    logger.debug("WRITE INTRODUCTION")
    review_plan = state['systematic_review_outline']
    analyses = state['analyses']
    messages = [SystemMessage(content=introduction_prompt)
                ] + review_plan + analyses
    model = ChatOllama(model="llama3.2")
    response = _make_api_call(model, messages)
    logger.debug(response)
    logger.newline()
    return {"introduction": [response]}


def write_methods(state: AgentState):
    logger.debug("WRITE METHODS")
    review_plan = state['systematic_review_outline']
    analyses = state['analyses']
    messages = [SystemMessage(content=methods_prompt)] + review_plan + analyses
    model = ChatOllama(model="llama3.2")
    response = _make_api_call(model, messages)
    logger.debug(response)
    logger.newline()
    return {"methods": [response]}


def write_results(state: AgentState):
    logger.debug("WRITE RESULTS")
    review_plan = state['systematic_review_outline']
    analyses = state['analyses']
    messages = [SystemMessage(content=results_prompt)] + review_plan + analyses
    model = ChatOllama(model="llama3.2")
    response = _make_api_call(model, messages)
    logger.debug(response)
    logger.newline()
    return {"results": [response]}


def write_conclusion(state: AgentState):
    logger.debug("WRITE CONCLUSION")
    review_plan = state['systematic_review_outline']
    analyses = state['analyses']
    messages = [SystemMessage(content=conclusions_prompt)
                ] + review_plan + analyses
    model = ChatOllama(model="llama3.2")
    response = _make_api_call(model, messages)
    logger.debug(response)
    logger.newline()
    return {"conclusion": [response]}


def write_references(state: AgentState):
    logger.debug("WRITE REFERENCES")
    review_plan = state['systematic_review_outline']
    analyses = state['analyses']
    messages = [SystemMessage(content=references_prompt)
                ] + review_plan + analyses
    model = ChatOllama(model="llama3.2")
    response = _make_api_call(model, messages)
    logger.debug(response)
    logger.newline()
    return {"references": [response]}


"""
### F. Final Stage Functions

#### Aggregator
```python
def aggregator(state: AgentState)
```
Combines all paper sections:
- Takes latest version of each section
- Maintains proper section order
- Adds spacing between sections
- Returns complete draft

## Critique Function
```python
def critique(state: AgentState)
```
Reviews complete draft:
- Uses review plan as reference
- Generates critique
- Increments revision counter
- Returns critique and revision number

## Paper Reviser
```python
def paper_reviser(state: AgentState)
```
Implements critique feedback:
- Takes latest critique and draft
- Uses revision prompt
- Generates revised version
- Returns updated draft

## Decision Function
```python
def exists_action(state: AgentState)
```
Controls workflow direction:
- Checks revision count limit
- Evaluates need for more research
- Returns decision:
  - "final_draft": If max revisions reached
  - True: If more research needed
  - "revise": For continued revision

## Final Draft Handler
```python
def final_draft(state: AgentState)
```
Completes review process:
- Returns final version of draft
- Marks end of revision cycle

These functions represent the final stages of the systematic review process, handling paper compilation, revision, and completion.
"""
logger.info("### F. Final Stage Functions")


def aggregator(state: AgentState):
    logger.debug("AGGREGATE")
    abstract = state['abstract'][-1].content
    introduction = state['introduction'][-1].content
    methods = state['methods'][-1].content
    results = state['results'][-1].content
    conclusion = state['conclusion'][-1].content
    references = state['references'][-1].content

    messages = [
        SystemMessage(
            content="Make a title for this systematic review based on the abstract. Write it in markdown."),
        HumanMessage(content=abstract)
    ]
    title = model.invoke(messages, temperature=0.1).content

    draft = title + "\n\n" + abstract + "\n\n" + introduction + "\n\n" + \
        methods + "\n\n" + results + "\n\n" + conclusion + "\n\n" + references

    return {"draft": [draft]}


def critique(state: AgentState):
    logger.debug("CRITIQUE")
    draft = state["draft"]
    review_plan = state['systematic_review_outline']

    messages = [SystemMessage(
        content=critique_draft_prompt)] + review_plan + draft
    response = model.invoke(messages, temperature=temperature)
    logger.debug(response)

    return {'messages': [response], "revision_num": state.get("revision_num", 1) + 1}


def paper_reviser(state: AgentState):
    logger.debug("REVISE PAPER")
    critique = state["messages"][-1].content
    draft = state["draft"]

    messages = [SystemMessage(content=revise_draft_prompt)
                ] + [critique] + draft
    response = model.invoke(messages, temperature=temperature)
    logger.debug(response)

    return {'draft': [response]}


def exists_action(state: AgentState):
    '''
    Determines whether to continue revising, end, or search for more articles
    based on the critique and revision count
    '''
    logger.debug("DECIDING WHETHER TO REVISE, END, or SEARCH AGAIN")

    if state["revision_num"] > state["max_revisions"]:
        return "final_draft"

    critique = state['messages'][-1]
    logger.debug(critique)

    if hasattr(critique, 'tool_calls') and critique.tool_calls:
        return True
    else:
        return "revise"


def final_draft(state: AgentState):
    logger.debug("FINAL DRAFT")
    return {"draft": state['draft']}


"""
## VII. Create Graph

### Graph Initialization
```python
graph = StateGraph(AgentState)
```
Creates a directed graph to manage the systematic review workflow using AgentState for data management.

## Node Addition
The graph adds nodes in logical groups:

**Initial Processing**
- process_input: Entry point
- planner: Creates review strategy
- researcher: Develops search approach
- search_articles: Finds papers
- article_decisions: Selects papers
- download_articles: Gets PDFs
- paper_analyzer: Analyzes content

**Writing Sections**
- write_abstract
- write_introduction
- write_methods
- write_results
- write_conclusion
- write_references

**Final Processing**
- aggregate_paper: Combines sections
- critique_paper: Reviews draft
- revise_paper: Makes changes
- final_draft: Completes review

## Edge Connections

**Main Flow**
- Linear flow from input through paper analysis
- Parallel paths from analyzer to writing sections
- All writing sections converge at aggregator

**Review Cycle**
Conditional branching after critique:
- To final_draft: If complete
- To revise_paper: If needs changes
- To search_articles: If needs more research

The graph creates a complete workflow for systematic review generation, with built-in revision cycles and quality control.
"""
logger.info("## VII. Create Graph")

graph = StateGraph(AgentState)
graph.add_node("process_input", process_input)
graph.add_node("planner", plan_node)
graph.add_node("researcher", research_node)
graph.add_node("search_articles", take_action)
graph.add_node("article_decisions", decision_node)
graph.add_node("download_articles", article_download)
graph.add_node("paper_analyzer", paper_analyzer)

graph.add_node("write_abstract", write_abstract)
graph.add_node("write_introduction", write_introduction)
graph.add_node("write_methods", write_methods)
graph.add_node("write_results", write_results)
graph.add_node("write_conclusion", write_conclusion)
graph.add_node("write_references", write_references)

graph.add_node("aggregate_paper", aggregator)
graph.add_node("critique_paper", critique)
graph.add_node("revise_paper", paper_reviser)
graph.add_node("final_draft", final_draft)

graph.add_edge("process_input", "planner")
graph.add_edge("planner", "researcher")
graph.add_edge("researcher", "search_articles")
graph.add_edge("search_articles", "article_decisions")
graph.add_edge("article_decisions", "download_articles")
graph.add_edge("download_articles", 'paper_analyzer')

graph.add_edge("paper_analyzer", "write_abstract")
graph.add_edge("paper_analyzer", "write_introduction")
graph.add_edge("paper_analyzer", "write_methods")
graph.add_edge("paper_analyzer", "write_results")
graph.add_edge("paper_analyzer", "write_conclusion")
graph.add_edge("paper_analyzer", "write_references")

graph.add_edge("write_abstract", "aggregate_paper")
graph.add_edge("write_introduction", "aggregate_paper")
graph.add_edge("write_methods", "aggregate_paper")
graph.add_edge("write_results", "aggregate_paper")
graph.add_edge("write_conclusion", "aggregate_paper")
graph.add_edge("write_references", "aggregate_paper")

graph.add_edge("aggregate_paper", 'critique_paper')

graph.add_conditional_edges(
    "critique_paper",
    exists_action,
    {"final_draft": "final_draft",
     "revise": "revise_paper",
     True: "search_articles"}
)

graph.add_edge("revise_paper", "critique_paper")
graph.add_edge("final_draft", END)

graph.set_entry_point("process_input")  # "llm"

"""
## VIII. Compile and Run Graph
"""
logger.info("## VIII. Compile and Run Graph")

checkpointer = MemorySaver()
graph = graph.compile(checkpointer=checkpointer)

render_mermaid_graph(graph, output_filename=f"{OUTPUT_DIR}/graph_output.png")

topic = "diffusion models for music generation"
thread_id = "test18"
temperature = 0.1
papers_tool = AcademicPaperSearchTool()
tooling = [papers_tool]
model = ChatOllama(model="llama3.2")  # llama3.2
tools = {t.name: t for t in tooling} if tooling else {}
model = model.bind_tools(tooling) if tools else model

agent_input = {"messages": [HumanMessage(content=topic)]}
thread_config = {"configurable": {"thread_id": thread_id}}
result = graph.invoke(agent_input, thread_config)

save_file(result, f"{OUTPUT_DIR}/workflow_results.json", result)

final_paper = result['draft'][-1].content
save_file(Markdown(final_paper), f"{OUTPUT_DIR}/final_paper.json")

logger.info("\n\n[DONE]", bright=True)
