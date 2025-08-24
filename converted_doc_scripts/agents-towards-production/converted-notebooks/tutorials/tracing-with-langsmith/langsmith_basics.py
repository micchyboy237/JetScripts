from jet.llm.mlx.base_langchain import ChatMLX
from jet.logger import CustomLogger
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import json
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
![](https://europe-west1-atp-views-tracker.cloudfunctions.net/working-analytics?notebook=tutorials--tracing-with-langsmith--langsmith-basics)

# LangSmith Tutorial: Adding Observability to AI Systems with LangGraph

## Overview

This tutorial demonstrates how to add observability to AI systems using LangSmith and LangGraph. While building AI applications has become more accessible, understanding how they make decisions and monitoring their behavior in real-world usage remains challenging.

### Why Observability Matters

Most AI applications work well in demonstrations but become difficult to debug and optimize when deployed. Without visibility into the decision-making process, teams struggle with fundamental questions: Why did the AI choose this particular response? Which parts of the system are slow or expensive? How can we systematically improve performance based on real usage patterns? What trajectories of tools are most effective? Which of these trajectories are the most effective, repetitive and cost effective?

Think of observability as adding a "flight recorder" to your AI system. Just as airlines use black boxes to understand what happened during flights, LangSmith captures every decision, timing, and data flow in your AI system. This transforms AI development from guesswork into engineering.

### What You'll Learn

By following this tutorial, you will understand how to instrument AI systems for transparency and observability. 

We'll build a simple research assistant using LangGraph that demonstrates key observability patterns. The assistant will analyze questions, decide whether to use tools, execute those tools when needed, and provide helpful responses. Throughout this process, LangSmith will capture detailed traces that show every decision point, timing data, and the flow of information through the system.

## Prerequisites and Initial Setup

Before we begin building, we need to set up our development environment. This involves installing the necessary packages and configuring API keys for both MLX(which powers our AI) and LangSmith (which provides observability).

Understanding this setup is crucial because LangSmith works by automatically intercepting and logging all LangGraph operations. Once configured, every LLM call, tool execution, and workflow step will be captured without requiring additional code changes. 

Requirements:
- Python 3.9+ 
- MLX API key ([get one here](https://platform.openai.com/api-keys))
- LangSmith account ([free signup](https://smith.langchain.com)) - This provides the observability dashboard where you'll see all the insights
"""
logger.info("# LangSmith Tutorial: Adding Observability to AI Systems with LangGraph")

# !pip install -U langchain-core langchain-openai langgraph langsmith requests

"""
## API Configuration

Now we'll configure the API keys and enable LangSmith tracing. The key insight here is that setting `LANGCHAIN_TRACING_V2=true` automatically enables comprehensive logging of all operations. Think of this as installing a "flight recorder" for your AI system. From this point forward, every decision and operation will be captured and made visible in your LangSmith dashboard.

This is fundamentally different from traditional logging because LangSmith understands the structure of AI workflows. Instead of just capturing text logs, it builds a complete picture of how information flows through your system.
"""
logger.info("## API Configuration")


# os.environ['OPENAI_API_KEY'] = ''
os.environ['LANGCHAIN_API_KEY'] = ''
os.environ['LANGCHAIN_TRACING_V2'] = 'true'  # This triggers observability
os.environ['LANGCHAIN_PROJECT'] = 'langsmith-tutorial-demo' #This is the project name where the traces will be stored

# required_vars = ['OPENAI_API_KEY', 'LANGCHAIN_API_KEY']
for var in required_vars:
    if not os.getenv(var) or 'your_' in os.getenv(var, ''):
        logger.debug(f"Warning: {var} needs your actual key")
    else:
        logger.debug(f"✓ {var} configured")

logger.debug(f"\nLangSmith Project: {os.getenv('LANGCHAIN_PROJECT')}")
logger.debug("\nTracing is now active - all AI operations will be logged for analysis")
logger.debug("Visit https://smith.langchain.com to see your traces")

"""
## Building a Simple Observable Agent

We'll create a minimal system that still demonstrates the power of LangSmith observability. Our agent will have just two capabilities: answering questions directly from its training data, and using a simple search tool when it needs current information.

Let's start by setting up our basic components:
"""
logger.info("## Building a Simple Observable Agent")


llm = ChatMLX(model="qwen3-1.7b-4bit-mini", temperature=0)

logger.debug("Language model initialized with temperature=0 for consistent behavior")
logger.debug("All LLM calls will be automatically traced in LangSmith")

"""
## Defining Our Simple Agent State

The agent's state represents the information that flows through our workflow. For our simple agent, we need just enough state to demonstrate meaningful observability. Think of this as the "memory" that LangSmith will track as it moves through each step of the process.

Each field in this state serves a specific purpose for observability. The user_question field lets us correlate behavior with input types. The needs_search field shows us the agent's decision-making. The search_result field captures tool execution results. And the reasoning field provides explicit explanations that help us understand why the agent made specific choices.
"""
logger.info("## Defining Our Simple Agent State")

class AgentState(TypedDict):
    """Simple state that flows through our agent workflow."""
    user_question: str        # The original question from the user
    needs_search: bool        # Whether we determined search is needed
    search_result: str        # Result from our search tool (if used)
    final_answer: str         # The response we'll give to the user
    reasoning: str            # Why we made our decisions (great for observability)

logger.debug("Agent state schema defined")
logger.debug("This structured state enables LangSmith to track information flow")

"""
## Creating Our Search Tool

We'll implement a search function that can fetch current information from Wikipedia. This tool demonstrates how LangSmith captures tool execution details, including timing, success/failure status, and returned data.

The implementation uses Wikipedia's search API properly. Unlike a page summary endpoint that requires exact page titles, the search API can handle general queries and return relevant results. Notice how we include comprehensive error handling and logging statements. These will appear in your LangSmith traces, helping you understand what happens during execution.

Understanding the difference between summary and search APIs is important: summary APIs expect exact page titles (like "Artificial_intelligence") while search APIs can handle natural language queries (like "what is AI").
"""
logger.info("## Creating Our Search Tool")

@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for current information about a topic."""
    try:
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 3  # Get top 3 results
        }

        response = requests.get(search_url, params=search_params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            search_results = data.get('query', {}).get('search', [])

            if search_results:
                top_result = search_results[0]
                page_title = top_result['title']

                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title.replace(' ', '_')}"
                summary_response = requests.get(summary_url, timeout=10)

                if summary_response.status_code == 200:
                    summary_data = summary_response.json()
                    extract = summary_data.get('extract', 'No summary available')
                    return f"Found information about '{page_title}': {extract[:400]}..."
                else:
                    return f"Found '{page_title}' but couldn't retrieve summary"
            else:
                return f"No Wikipedia articles found for '{query}'"
        else:
            return f"Wikipedia search failed with status {response.status_code}"

    except Exception as e:
        return f"Search error: {str(e)}"

logger.debug("Search tool created with proper Wikipedia search API integration")
logger.debug("Tool execution timing and results will be captured automatically")

"""
## Building the Decision-Making Workflow

Now we'll create the core logic of our agent. Even though this is a simple system, we'll structure it as separate functions to demonstrate how LangSmith traces multi-step workflows. Each function represents a clear decision point that will be visible in your observability dashboard.

This modular approach serves two purposes: it makes the code easier to understand and test, and it creates natural breakpoints that LangSmith can capture and analyze. 

### Step 1: Deciding Whether to Search

The first step analyzes the user's question and decides whether we need to search for current information, or if we can answer directly from the model's training data.
"""
logger.info("## Building the Decision-Making Workflow")

def decide_search_need(state: AgentState) -> AgentState:
    """Analyze the question and decide if we need to search for current information."""
    user_question = state["user_question"]

    decision_prompt = f"""
    Analyze this question and decide if it requires current/recent information that might not be in your training data:

    Question: "{user_question}"

    Consider:
    - Does this ask about recent events, current prices, or breaking news?
    - Does this ask about people, companies, or topics that change frequently?
    - Can you answer this well using your existing knowledge?

    Respond with exactly "SEARCH" if you need current information, or "DIRECT" if you can answer directly.
    Then on a new line, briefly explain your reasoning.
    """

    response = llm.invoke([SystemMessage(content=decision_prompt)])
    decision_text = response.content.strip()

    lines = decision_text.split('\n')
    decision = lines[0].strip()
    reasoning = lines[1] if len(lines) > 1 else "No reasoning provided"

    state["needs_search"] = decision == "SEARCH"
    state["reasoning"] = f"Decision: {decision}. Reasoning: {reasoning}"

    logger.debug(f"Decision: {'SEARCH' if state['needs_search'] else 'DIRECT'} - {reasoning}")

    return state

"""
### Step 2: Executing Search When Needed

If the previous step determined that search is needed, this function executes our search tool. LangSmith shows not just whether the search happened, but exactly what query was sent, how long it took, and what results came back.

This conditional execution pattern is common in AI systems, and LangSmith handles it by showing you which path was taken and why.
"""
logger.info("### Step 2: Executing Search When Needed")

def execute_search(state: AgentState) -> AgentState:
    """Execute search if needed, otherwise skip this step."""
    if not state["needs_search"]:
        logger.debug("Skipping search - not needed for this question")
        state["search_result"] = "No search performed"
        return state

    logger.debug(f"Executing search for: {state['user_question']}")

    search_result = wikipedia_search.invoke({"query": state["user_question"]})
    state["search_result"] = search_result

    logger.debug(f"Search completed: {len(search_result)} characters returned")

    return state

"""
### Step 3: Generating the Final Response

The final step synthesizes all available information into a helpful response. This is where we combine the model's built-in knowledge with any search results we gathered. The synthesis process is often the most complex part of an AI system, and LangSmith's observability helps you understand how well this process works.
"""
logger.info("### Step 3: Generating the Final Response")

def generate_response(state: AgentState) -> AgentState:
    """Generate the final response using all available information."""
    user_question = state["user_question"]
    search_result = state.get("search_result", "")
    used_search = state["needs_search"]

    if used_search and "Search error" not in search_result:
        context = f"Question: {user_question}\n\nSearch Results: {search_result}"
        response_prompt = f"""
        Answer the user's question using both your knowledge and the search results provided.

        {context}

        Provide a helpful, accurate response that synthesizes the information.
        """
    else:
        response_prompt = f"""
        Answer this question using your existing knowledge:

        {user_question}

        Provide a helpful, accurate response.
        """

    response = llm.invoke([SystemMessage(content=response_prompt)])
    state["final_answer"] = response.content

    logger.debug(f"Response generated: {len(response.content)} characters")

    return state

"""
## Assembling the Workflow

Now we'll connect our functions into a complete workflow using LangGraph. This creates an explicit graph structure that LangSmith can visualize, showing you the exact path your AI takes through the decision-making process.

The graph structure is particularly valuable for observability because it makes the control flow explicit. Instead of having conditional logic buried in function calls, LangGraph creates a visual representation that shows exactly which steps were executed and in what order.

Think of this as creating a roadmap that LangSmith can follow to show you the journey your data took through the system.
"""
logger.info("## Assembling the Workflow")

workflow = StateGraph(AgentState)

workflow.add_node("decide", decide_search_need)
workflow.add_node("search", execute_search)
workflow.add_node("respond", generate_response)

workflow.set_entry_point("decide")
workflow.add_edge("decide", "search")     # Always go to search step (it will skip if not needed)
workflow.add_edge("search", "respond")    # Then generate response
workflow.add_edge("respond", END)         # Finish

simple_agent = workflow.compile()

logger.debug("Workflow compiled successfully")
logger.debug("Flow: decide → search → generate_response")
logger.debug("Ready to demonstrate LangSmith observability")

"""
## The Workflow Visualization

 Below is a visualization of our agent's workflow graph. This diagram shows the structure that LangSmith will trace during execution:
 
#  <img src="assets/wiki_agent_td.png" alt="Agent Workflow Graph" width="700">
 
 This visualization helps us understand the flow between the different components of our agent: decision-making, search execution, and response generation.

## Testing with Full Observability

We'll run our agent on different types of questions and see how LangSmith captures every detail of the execution. Each test will generate a complete trace that shows you the decision-making process, timing information, and all intermediate results.

This test runner includes timing measurements and metadata tagging, which helps organize your traces in LangSmith for easier analysis. The metadata and tags serve as filters that let you group and compare similar types of executions.
"""
logger.info("## The Workflow Visualization")

def run_test_with_observability(question: str, test_type: str) -> dict:
    """Run a test and capture comprehensive observability data."""
    logger.debug(f"\n{'='*60}")
    logger.debug(f"Testing: {question}")
    logger.debug(f"Type: {test_type}")
    logger.debug(f"{'='*60}")

    start_time = time.time()

    initial_state = {
        "user_question": question,
        "needs_search": False,
        "search_result": "",
        "final_answer": "",
        "reasoning": ""
    }

    try:
        config = {
            "metadata": {
                "test_type": test_type,
                "tutorial": "langsmith-observability"
            },
            "tags": ["tutorial", "demo", test_type]
        }

        final_state = simple_agent.invoke(initial_state, config=config)

        end_time = time.time()
        total_time = end_time - start_time

        logger.debug(f"\nResults:")
        logger.debug(f"   Decision Process: {final_state['reasoning']}")
        logger.debug(f"   Used Search: {'Yes' if final_state['needs_search'] else 'No'}")
        logger.debug(f"   Response Length: {len(final_state['final_answer'])} characters")
        logger.debug(f"   Total Time: {total_time:.2f} seconds")
        logger.debug(f"\nAnswer: {final_state['final_answer'][:200]}...")

        return {
            "question": question,
            "type": test_type,
            "success": True,
            "used_search": final_state['needs_search'],
            "total_time": round(total_time, 2),
            "reasoning": final_state['reasoning']
        }

    except Exception as e:
        logger.debug(f"Error: {str(e)}")
        return {
            "question": question,
            "type": test_type,
            "success": False,
            "error": str(e)
        }

"""
## Running Our Test Suite

Let's test our agent with questions that should trigger different behaviors. We have three types of questions that will help us understand how the agent makes decisions:

**Direct Answer**: Questions the model can answer from training data. These should show the agent choosing not to search, demonstrating efficient resource usage.

**Current Info**: Questions requiring recent information. These should trigger search behavior, showing how the agent handles information that changes over time.

**Factual Lookup**: Questions about established facts that might benefit from verification. These will show how the agent balances confidence in its training data against the value of current verification.

Watch how each question type flows through the system differently, and pay attention to the decision-making process that LangSmith captures.
"""
logger.info("## Running Our Test Suite")

test_cases = [
    {
        "question": "What is the capital of France?",
        "type": "direct_answer",
        "expected_search": False
    },
    {
        "question": "What happened in the 2024 US presidential election?",
        "type": "current_info",
        "expected_search": True
    },
    {
        "question": "Tell me about artificial intelligence",
        "type": "factual_lookup",
        "expected_search": False  # Should be answerable directly
    }
]

logger.debug("Starting LangSmith Observability Demo")
logger.debug("Each test will generate detailed traces in your LangSmith dashboard")
logger.debug("Visit https://smith.langchain.com to see real-time traces\n")

test_results = []

for i, test_case in enumerate(test_cases, 1):
    logger.debug(f"\nRunning Test {i}/{len(test_cases)}")

    result = run_test_with_observability(
        test_case["question"],
        test_case["type"]
    )

    test_results.append(result)

    time.sleep(1)

logger.debug(f"\n\nAll tests completed")
logger.debug(f"Generated {len(test_results)} traces in LangSmith")
logger.debug(f"Check your dashboard to explore the detailed execution data")

"""
## Understanding Your LangSmith Dashboard

Now that you've run the tests, let's explore what LangSmith has captured. Navigate to your LangSmith dashboard at [smith.langchain.com](https://smith.langchain.com) and select the `langsmith-tutorial-demo` project.

### What You'll See in LangSmith

**Trace List View**: You'll see a list of all your test executions. Each row represents one complete run of your agent, showing the input question, execution time, success/failure status, and total cost. This view gives you a high-level overview of system performance across different types of queries.

![Local image](./assets/1.png)

Think of this as your system's activity log, but with much richer information than traditional logs. You can sort by execution time to find slow queries, filter by tags to analyze specific types of questions, or look for patterns in cost per query.

**Individual Trace Details**: Click on any trace to see the complete execution flow. You'll see:

![Local image](./assets/2.png)
**Graph Visualization**: A visual representation of your workflow showing which nodes were executed and how data flowed between them. This is particularly powerful because you can see at a glance whether search was used and how long each step took.

**Step-by-step Execution**: Each function call with inputs, outputs, and timing. This granular view helps you understand not just what happened, but why it happened. You can see the exact prompts sent to the language model and the reasoning it provided.

![Local image](./assets/3.png)

**LLM Calls**: Every prompt sent to the language model with the exact response. This transparency is crucial for prompt optimization and understanding model behavior.

**Tool Executions**: When and how your search tool was called, including the query sent, response received, and execution time.

**Performance Analytics**: LangSmith automatically aggregates performance data across all your runs:

**Latency Patterns**: Which steps consistently take the longest? Is search always the bottleneck, or does decision-making sometimes slow things down?

**Cost Analysis**: How much does each type of query cost? Are search queries significantly more expensive than direct answers?

**Success Rates**: Are there categories of queries that consistently fail? Do certain question patterns lead to search errors?

**Tool Usage Patterns**: How often is search actually needed? Are there questions that trigger search unnecessarily?

## Key Observability Insights

Based on the traces you've just generated, here are the types of insights LangSmith enables. These insights transform how you understand and improve AI systems.

### Decision-Making Transparency

For each question, you can see exactly why the agent chose to search or answer directly. This transparency is crucial for debugging unexpected behavior and building trust in AI systems. When users ask why the system behaved a certain way, you can point to specific reasoning captured in the traces.

### Performance Optimization Opportunities

By comparing execution times across different question types, you can identify bottlenecks and optimization opportunities. For example, you might discover that search queries take significantly longer than direct answers, suggesting opportunities for caching or parallel execution.

### Cost Management

LangSmith shows you the token usage and estimated cost for each LLM call. This granular cost data helps you optimize in ways that would be impossible without observability. You can identify expensive operations and optimize prompts or routing logic to reduce costs without sacrificing quality.

### Quality Assurance

With complete traces, you can verify that the agent is making reasonable decisions consistently. You can spot patterns in successful versus failed executions, identify edge cases that need additional handling, and create regression tests based on real usage patterns.

This systematic approach to quality assurance is only possible with comprehensive observability.

## Taking This Further

This tutorial demonstrated the fundamental principles of AI observability using a simple agent. The patterns you've learned scale to much more complex systems. Here's are some tips to get started:

### For Your Own Projects

**Start with observability from day one**: Enable LangSmith tracing before building complex features. 

**Structure your workflows**: Use LangGraph's explicit workflow structure to make decision points visible. 

**Add meaningful metadata**: Tag your traces with business context to enable better analysis.

**Monitor key metrics**: Set up alerts for latency, cost, and error rates based on the patterns you observe in LangSmith.


The observability foundation you've built here becomes even more valuable in production environments where understanding system behavior is critical for maintaining service quality and user trust.

---
This tutorial was written by [Shivnarayan Rajappa](https://www.linkedin.com/in/shivnarayanrajappa/)
"""
logger.info("## Understanding Your LangSmith Dashboard")

logger.info("\n\n[DONE]", bright=True)