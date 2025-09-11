from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import CustomLogger
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_mongodb.agent_toolkit import MONGODB_AGENT_SYSTEM_PROMPT
from langchain_mongodb.agent_toolkit.database import MongoDBDatabase
from langchain_mongodb.agent_toolkit.toolkit import MongoDBDatabaseToolkit
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent
from pymongo import MongoClient
from typing import Any, Dict, Literal
import json
import msgpack
import os
import shutil
import time
import traceback
import uuid


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/agents/mongodb_building_a_text_to_mql_agent.ipynb)

# Build a Production-Ready Text-to-MQL Agent for MongoDB

Transform natural language into powerful MongoDB queries using AI agents that remember context, learn from conversations, and provide intelligent insights into your data.

## Overview

By the end of this notebook, you will have implemented a production-ready conversational database agent with the following capabilities:

- **Natural language processing**: Convert human language queries into MongoDB aggregation pipelines
- **Query generation**: Automatically generate complex MongoDB queries from simple descriptions
- **Conversation memory**: Maintain context across multiple related queries in a session
- **Debugging and observability**: Track step-by-step execution with detailed summaries
- **Architecture comparison**: Implement and compare ReAct vs. structured custom agent approaches

## Use Cases

Traditional database interaction requires knowledge of MongoDB aggregation syntax, collection schemas, and query validation. This agent abstracts these complexities, providing a natural language interface for database operations.

## Implementation Approaches

### ReAct Agent
- Flexible reasoning and tool selection
- Suitable for exploratory queries and rapid prototyping
- Autonomous decision-making for tool usage

### Custom LangGraph Agent
- Deterministic, structured workflow
- Enhanced debugging capabilities with full observability
- Designed for production environments with predictable behavior

## Memory System

The system implements a custom MongoDB-based memory system with LLM-powered summarization that provides:

```
User: Count query for movies
Schema: movies collection
Query: aggregation pipeline
Results: 5 documents returned
Response: formatted answer
```

Conversation memory enables multi-turn interactions:
- "List the top directors" → Agent returns top 3 directors
- "What was the count for the first one?" → Agent references previous results
- "Show me their best films" → Agent continues with context

## Business Applications

This system handles sophisticated analytical queries such as:

- **Analytics**: "Which states have the most theaters and what's the average occupancy?"
- **Recommendations**: "Find directors similar to Christopher Nolan with at least 10 films"
- **Trend Analysis**: "Show me movie rating trends by decade for sci-fi films"
- **Geographic Analysis**: "Which theaters are furthest west and what movies do they show?"

## Technical Components

- **MongoDB Atlas**: Data storage with aggregation pipeline support
- **Ollama GPT**: Natural language processing and query generation
- **LangGraph**: Deterministic agent workflow management
- **LangChain**: LLM integration and tool orchestration
- **Persistent Memory**: Conversation state management with enhanced debugging

## Prerequisites

To run this notebook, you need:

- MongoDB Atlas cluster with the `sample_mflix` dataset loaded
  - Follow the [sample data loading instructions](https://www.mongodb.com/docs/atlas/sample-data/#std-label-load-sample-data)
  - Or follow-along with the screenshots below
- Ollama API key
- Environment variables:
  - `MONGODB_URI`
#   - `OPENAI_API_KEY`

![Sample Data UI](../../misc/accompanying-images/atlas_ui_load_sample_data_01.png)

![Sample Data UI](../../misc/accompanying-images/atlas_ui_load_sample_data_02.png)

![Sample Data UI](../../misc/accompanying-images/atlas_ui_load_sample_data_03.png)

![Sample Data UI](../../misc/accompanying-images/atlas_ui_load_sample_data_04.png)

## 🌐 Network Setup: Connect to Your Atlas Cluster

Before we dive into the implementation, let's make sure your environment can reach MongoDB Atlas.

⚠️ **Quick IP Check** - Run this to get your current IP address for MongoDB Atlas network access list:

⚠️  Check your public IP — useful for updating MongoDB Atlas network access if needed.
"""
logger.info("# Build a Production-Ready Text-to-MQL Agent for MongoDB")

# !curl ifconfig.me

"""
# System Setup and Configuration

This section installs the required dependencies and configures the core components needed for the text-to-MQL system.

## Step 1: Install Dependencies

Installing the core libraries for AI-powered database interaction:

- **LangGraph**: Modern AI agent framework
- **LangChain MongoDB**: Database integration tools
- **Ollama Integration**: GPT model integration for query generation
- **MongoDB Checkpointing**: Persistent memory management
"""
logger.info("# System Setup and Configuration")

# !pip install -U langgraph langgraph-checkpoint-mongodb langchain-mongodb langchain-ollama ollama pymongo


logger.debug("📦 All dependencies installed successfully!")

"""
## Configure Credentials

**Configuration Requirements:**

1. **MongoDB Atlas Connection String**
   - Obtain from [MongoDB Atlas Console](https://www.mongodb.com/docs/manual/reference/connection-string/)
   - Ensure the `sample_mflix` dataset is loaded

2. **Ollama API Key**
   - Obtain from [Ollama Platform](https://help.ollama.com/en/articles/4936850-where-do-i-find-my-ollama-api-key)
   - GPT-4o-mini is used for optimal performance and cost balance

**Note**: In production environments, use secure environment variable management rather than hardcoded values.
"""
logger.info("## Configure Credentials")

os.environ["MONGODB_URI"] = "insert_your_mongodb_connection_string_here"
# os.environ["OPENAI_API_KEY"] = "insert_your_ollama_api_key_here"

logger.debug("🔑 Environment variables configured!")

"""
## Initialize Core Components

Initialize the foundation components required for the text-to-MQL system:

- **MongoDBDatabase wrapper**: Provides AI-accessible interface to database operations
- **ChatOllama interface**: Handles language model interactions
- **MongoDB client**: Powers the conversation memory system
"""
logger.info("## Initialize Core Components")

db = MongoDBDatabase.from_connection_string(
    os.getenv("MONGODB_URI"), database="sample_mflix"
)

llm = ChatOllama(model="llama3.2")

client = MongoClient(
    os.getenv("MONGODB_URI"), appname="devrel.showcase.notebook.agent.text_to_mql_agent"
)

logger.debug("✅ Database and LLM initialized successfully!")

"""
# MongoDB Toolkit Overview

The `MongoDBDatabaseToolkit` provides comprehensive MongoDB capabilities for AI agents:

| Tool | Purpose | Example Use Case |
|------|---------|------------------|
| `mongodb_list_collections` | Database discovery | "What collections are available?" |
| `mongodb_schema` | Schema inspection | "What is the structure of the movies collection?" |
| `mongodb_query_checker` | Query validation | "Validate this aggregation pipeline" |
| `mongodb_query` | Query execution | "Execute this MongoDB query" |

These tools enable the AI agent to understand database structure and execute queries autonomously.
"""
logger.info("# MongoDB Toolkit Overview")

toolkit = MongoDBDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
tool = {t.name: t for t in tools}

logger.debug("🛠️ Available Tools:", list(tool.keys()))

"""
# Data Discovery

Examine the sample dataset structure. The `sample_mflix` dataset provides:

- **Movies collection**: Film metadata including ratings, cast, and genres
- **Users collection**: User profiles and preferences
- **Comments collection**: User reviews and ratings
- **Theaters collection**: Theater locations and screening information

This dataset demonstrates real-world complexity suitable for testing aggregation queries and geographic analysis.
"""
logger.info("# Data Discovery")

logger.debug("\n📋 Available Collections:",
             list(db.get_usable_collection_names()))

logger.debug("\n📊 Movies Collection Schema Sample:")
logger.debug(db.get_collection_info(["movies"])[:500] + "...")

"""
# Persisting Agent Outputs

## Overview

Instead of saving outputs to a local file, you can persist them in MongoDB using the built-in LangGraph saver. Treat past runs as “memory” and reload them easily.
This extends MongoDB's standard `MongoDBSaver` checkpointer with LLM-generated step summaries, providing human-readable conversation histories instead of raw checkpoint data.

## Features

### Readable Step Summaries
```
User: "How many movies from the 1990s?"
LLM Summary: "Count query with date range filter"
MongoDB Query: Aggregation pipeline with $match and $count operations
```

### Enhanced Thread Inspection
```
Step 1 [14:23:45] User asks about top movies  
Step 2 [14:23:46] Schema lookup: movies collection
Step 3 [14:23:47] Aggregation query execution
Step 4 [14:23:48] 5 results returned
Step 5 [14:23:49] Formatted response delivered
```

### Enhanced Metadata
Each checkpoint includes:
- `step_summary`: LLM-generated description
- `step_timestamp`: Execution timestamp
- `step_number`: Sequential step counter

## Implementation

The LLM analyzes each conversation step and generates concise summaries:
- **User messages**: Categorizes query intent and patterns
- **Tool calls**: Describes the operation being performed
- **Results**: Summarizes returned data
- **Errors**: Explains failure conditions

## Usage

```python
# Drop-in replacement for standard MongoDBSaver
checkpointer = LLMSummarizingMongoDBSaver(client, llm)

# Use with any LangGraph agent
agent = create_react_agent(llm, tools, checkpointer=checkpointer)
```

## Benefits

- **Compatible interface**: No code changes required from standard `MongoDBSaver`
- **Enhanced debugging**: Clear visibility into agent execution steps
- **Human-readable logs**: Understand conversation flow at a glance
- **Flexible implementation**: Works with any LangGraph agent and domain

This maintains all functionality of the standard LangGraph memory system while adding intelligent logging capabilities.
"""
logger.info("# Persisting Agent Outputs")


class LLMSummarizingMongoDBSaver(MongoDBSaver):
    """MongoDB saver with LLM-powered intelligent summarization"""

    def __init__(self, client, llm):
        super().__init__(client)
        self.llm = llm

        self._summary_cache = {}

    def summarize_step(self, checkpoint_data: Dict[str, Any]) -> str:
        """Generate contextual summary using LLM"""
        try:
            channel_values = checkpoint_data.get("channel_values", {})
            messages = channel_values.get("messages", [])

            if not messages:
                return "🔄 Initial state"

            last_message = messages[-1]

            if not last_message:
                return "📭 Empty step"

            message_type = (
                type(last_message).__name__
                if hasattr(last_message, "__class__")
                else "unknown"
            )
            content = getattr(last_message, "content", "") or ""
            tool_calls = getattr(last_message, "tool_calls", [])

            if isinstance(last_message, dict):
                message_type = last_message.get("type", "unknown")
                content = last_message.get("content", "")
                tool_calls = last_message.get("tool_calls", [])

            cache_key = f"{message_type}:{content[:50]}:{len(tool_calls)}"
            if cache_key in self._summary_cache:
                return self._summary_cache[cache_key]

            context_parts = []
            if content:
                context_parts.append(f"Content: {content[:200]}")
            if tool_calls:
                tool_info = []
                for tc in tool_calls[:2]:  # Limit to first 2 tool calls
                    tool_name = tc.get("name", "unknown")
                    tool_args = str(tc.get("args", {}))[:100]
                    tool_info.append(f"{tool_name}({tool_args})")
                context_parts.append(f"Tool calls: {', '.join(tool_info)}")

            context = "\n".join(
                context_parts) if context_parts else "No content"

            prompt = f"""Summarize this conversation step in 2-5 words with a relevant emoji.

Message type: {message_type}
{context}

Guidelines:
- Use emojis: 👤 for user, 🤖 for AI, 🔧 for tools, 📊 for data, ✨ for results
- Be concise and descriptive
- Focus on the action/intent

Examples:
- "👤 Count movies query"
- "🔧 Schema lookup: movies"
- "📊 Aggregation pipeline"
- "✨ Formatted results"
- "❌ Query validation error"

Summary:"""

            response = self.llm.invoke(prompt)
            summary = response.content.strip()[:60]  # Limit length

            self._summary_cache[cache_key] = summary

            if len(self._summary_cache) > 100:
                oldest_keys = list(self._summary_cache.keys())[:50]
                for key in oldest_keys:
                    del self._summary_cache[key]

            return summary

        except Exception as e:
            error_msg = str(e)[:30]
            return f"❓ Step (error: {error_msg}...)"

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Dict[str, Any],
        metadata: Dict[str, Any],
        new_versions: Dict[str, Any],
    ) -> RunnableConfig:
        """Override put method to add LLM-generated step summary"""
        try:
            step_summary = self.summarize_step(checkpoint)

            enhanced_metadata = metadata.copy() if metadata else {}
            enhanced_metadata["step_summary"] = step_summary
            enhanced_metadata["step_timestamp"] = checkpoint.get(
                "ts", "unknown")

            messages = checkpoint.get("channel_values", {}).get("messages", [])
            enhanced_metadata["step_number"] = len(messages)

            return super().put(config, checkpoint, enhanced_metadata, new_versions)

        except Exception as e:
            logger.debug(f"❌ Error adding LLM summary: {e}")
            return super().put(config, checkpoint, metadata, new_versions)


"""
## Thread Inspection and Debugging

### `inspect_thread_with_summaries_enhanced(thread_id: str, limit: int = 20, show_details: bool = False)`

This function provides a human-readable view of agent conversation history by fetching checkpoints from MongoDB and displaying LLM-generated step summaries in chronological order with timestamps.

**Features:**
- Automatic grouping of consecutive similar operations to reduce clutter
- Handles both dictionary and binary metadata formats
- Essential for debugging complex multi-step queries and understanding agent decision-making

**Example output:**
```
Thread History: session_123
Total steps: 5

Step 1 [14:23:45]
   User: count movies query

Step 2 [14:23:46]
   Schema lookup: movies

Step 3 [14:23:47]
   Aggregation pipeline

Step 4 [14:23:48]
   157 results returned

Step 5 [14:23:49]
   Formatted response
```

**Parameters:**
- `show_details=True`: Display all steps without grouping
- `limit`: Adjust to focus on recent activity
"""
logger.info("## Thread Inspection and Debugging")


def inspect_thread_with_summaries_enhanced(
    thread_id: str, limit: int = 20, show_details: bool = False
):
    """Enhanced thread inspection with better formatting"""
    try:
        db_checkpoints = client["checkpointing_db"]
        collection = db_checkpoints.checkpoints

        checkpoints = list(
            collection.find({"thread_id": thread_id}).sort(
                "_id", 1).limit(limit)
        )

        if not checkpoints:
            logger.debug(f"❌ No checkpoints found for thread: {thread_id}")
            return []

        logger.debug(f"\n🔍 Thread History: {thread_id}")
        logger.debug(f"📊 Total steps: {len(checkpoints)}")
        logger.debug("=" * 80)

        last_summary = None
        consecutive_count = 0

        for i, checkpoint_doc in enumerate(checkpoints, 1):
            timestamp = checkpoint_doc["_id"].generation_time
            time_str = timestamp.strftime("%H:%M:%S")

            metadata = checkpoint_doc.get("metadata", {})

            if isinstance(metadata, dict):
                step_summary = metadata.get("step_summary", "No summary")
            else:
                try:

                    decoded_metadata = msgpack.unpackb(
                        metadata, raw=False, strict_map_key=False
                    )
                    step_summary = decoded_metadata.get(
                        "step_summary", "No summary")
                except (msgpack.UnpackException, ValueError) as e:
                    step_summary = "Unable to decode"

            if isinstance(step_summary, bytes):
                step_summary = step_summary.decode("utf-8", errors="replace")

            if step_summary == last_summary and not show_details:
                consecutive_count += 1
            else:
                if consecutive_count > 0:
                    logger.debug(
                        f"   └─ (repeated {consecutive_count} more times)")

                logger.debug(f"\n📍 Step {i} [{time_str}]")
                logger.debug(f"   {step_summary}")

                last_summary = step_summary
                consecutive_count = 0

        if consecutive_count > 0:
            logger.debug(f"   └─ (repeated {consecutive_count} more times)")

        logger.debug("\n" + "=" * 80)
        return checkpoints

    except Exception as e:
        logger.debug(f"❌ Error inspecting thread: {e}")

        traceback.print_exc()
        return []


logger.debug("🔄 UPDATING AGENTS WITH LLM-POWERED SUMMARIZATION")
logger.debug("=" * 60)

"""
# ReAct Agent Creation Functions

### `create_react_agent_with_enhanced_memory()`

Creates a LangChain ReAct agent with persistent memory powered by the `LLMSummarizingMongoDBSaver`.

**Functionality:**
- Combines the standard MongoDB agent system prompt with enhanced checkpointer
- Provides ReAct agent with conversation memory across sessions
- Generates intelligent step summaries using LLM
- Uses the complete MongoDB toolkit for database operations

**Returns:** LangChain ReAct agent with MongoDB tools and LLM-powered memory

**Usage:**
```python
agent = create_react_agent_with_enhanced_memory()
config = {"configurable": {"thread_id": "my_session"}}
agent.invoke({"messages": [("user", "Count all movies")]}, config)
```
"""
logger.info("# ReAct Agent Creation Functions")


def create_react_agent_with_enhanced_memory():
    """Create ReAct agent with LLM-powered summarizing checkpointer"""
    system_message = MONGODB_AGENT_SYSTEM_PROMPT.format(top_k=5)
    summarizing_checkpointer = LLMSummarizingMongoDBSaver(client, llm)

    return create_react_agent(
        llm,
        toolkit.get_tools(),
        prompt=system_message,
        checkpointer=summarizing_checkpointer,
    )


"""
# Core LangGraph Components

This section defines the individual nodes and functions that comprise the custom LangGraph agent workflow.

### Workflow Design
Creates a deterministic, debuggable pipeline:
1. **Discovery**: List collections
2. **Schema Analysis**: Get relevant collection schemas
3. **Query Generation**: Convert natural language to MongoDB
4. **Validation**: Check and sanitize query (optional)
5. **Execution**: Run query against database
6. **Formatting**: Present results in readable format

Each step is a separate node, enabling easy debugging, modification, or workflow extension.

### Tool Nodes
Wraps MongoDB tools in LangGraph `ToolNode` format for the state machine.
"""
logger.info("# Core LangGraph Components")

schema_node = ToolNode([tool["mongodb_schema"]], name="get_schema")
run_node = ToolNode([tool["mongodb_query"]], name="run_query")

"""
### Workflow Node Functions

#### `list_collections(state: MessagesState)`
Deterministic node that automatically lists all available MongoDB collections. Always runs first to provide agent context about available data.
"""
logger.info("### Workflow Node Functions")


def list_collections(state: MessagesState):
    """Deterministic node to list available collections"""
    call = {
        "name": "mongodb_list_collections",
        "args": {},
        "id": "abc",
        "type": "tool_call",
    }
    call_msg = AIMessage(content="", tool_calls=[call])
    resp = tool["mongodb_list_collections"].invoke(call)
    summary = AIMessage(f"Available collections: {resp.content}")
    return {"messages": [call_msg, resp, summary]}


"""
#### `call_get_schema(state: MessagesState)`
LLM decision node that prompts the LLM to select which collections to examine and calls the schema tool. The LLM determines required schema information based on the user's query.
"""
logger.info("#### `call_get_schema(state: MessagesState)`")


def call_get_schema(state: MessagesState):
    """Prompt LLM to select and call schema tool"""
    llm_with = llm.bind_tools([tool["mongodb_schema"]], tool_choice="any")
    resp = llm_with.invoke(state["messages"])
    return {"messages": [resp]}


"""
#### `generate_query(state: MessagesState)`
Core query generation that converts user natural language into MongoDB aggregation pipeline. Uses the complete agent system prompt with conversation context.
"""
logger.info("#### `generate_query(state: MessagesState)`")


def generate_query(state: MessagesState):
    """Generate MongoDB aggregation pipeline"""
    llm_with = llm.bind_tools([tool["mongodb_query"]])
    resp = llm_with.invoke(
        [{"role": "system", "content": MONGODB_AGENT_SYSTEM_PROMPT}] + state["messages"]
    )
    return {"messages": [resp]}


"""
#### `check_query(state: MessagesState)`
Query validation that verifies and sanitizes the generated query before execution. Helps identify syntax errors and potential issues.
"""
logger.info("#### `check_query(state: MessagesState)`")


def check_query(state: MessagesState):
    """Validate and sanitize generated query"""
    original = state["messages"][-1].tool_calls[0]["args"]["query"]
    resp = llm.bind_tools([tool["mongodb_query"]], tool_choice="any").invoke(
        [
            {"role": "system", "content": MONGODB_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": original},
        ]
    )
    resp.id = state["messages"][-1].id
    return {"messages": [resp]}


"""
#### `format_answer(state: MessagesState)`
Result formatting that converts raw MongoDB JSON results into readable Markdown. Uses a dedicated formatting prompt to present data clearly to end users.
"""
logger.info("#### `format_answer(state: MessagesState)`")

FORMAT_SYS = """
You are an assistant that formats MongoDB query results for end-users.

Input variables
---------------
• {question} - the user's original natural-language query
• {docs}     - JSON array of documents returned by the database

Write a concise answer in Markdown:

1. Start with: **Answer to:** "<question>"
2. Present the documents clearly (numbered list, table, paragraph - whatever fits)
3. If the array is empty, say: "I couldn't find any matching documents."
Do NOT show the raw JSON.
"""


def format_answer(state):
    """Enhanced format function with large dataset handling"""

    raw_json = state["messages"][-1].content
    question = state["messages"][0].content

    try:
        data = json.loads(raw_json)

        if isinstance(data, list):
            data_size = len(data)

            if data_size == 0:
                return {
                    "messages": [
                        AIMessage(
                            content=f'**Answer to:** "{question}"\n\nI couldn\'t find any matching documents.'
                        )
                    ]
                }

            elif data_size > 50:  # Large dataset threshold
                sample_data = data[:10]
                response_parts = [
                    f'**Answer to:** "{question}"',
                    f"Found **{data_size}** results. Showing first 10:",
                    "",
                ]

                for i, item in enumerate(sample_data, 1):
                    if isinstance(item, dict) and "_id" in item:
                        if "movieCount" in item:
                            response_parts.append(
                                f"{i}. {item['_id']}: {item['movieCount']} movies"
                            )
                        else:
                            response_parts.append(f"{i}. {item['_id']}")

                response_parts.extend(
                    [
                        "",
                        f"... and {data_size - 10} more results.",
                        "💡 **Tip**: Try 'Show me the top 10...' for more manageable results",
                    ]
                )

                formatted_response = "\n".join(response_parts)

            else:  # Normal size dataset
                response_parts = [f'**Answer to:** "{question}"', ""]
                for i, item in enumerate(data, 1):
                    if isinstance(item, dict) and "_id" in item:
                        if "movieCount" in item:
                            response_parts.append(
                                f"{i}. {item['_id']}: {item['movieCount']} movies"
                            )
                        else:
                            response_parts.append(f"{i}. {item['_id']}")

                formatted_response = "\n".join(response_parts)
        else:
            formatted_response = f'**Answer to:** "{question}"\n\n{data!s}'

    except Exception as e:
        formatted_response = f"**Answer to:** \"{question}\"\n\n⚠️ Large dataset found but too big to display. Try limiting your query (e.g., 'top 10', 'first 5')."

    return {"messages": [AIMessage(content=formatted_response)]}


"""
### Control Flow

#### `need_checker(state: MessagesState) -> Literal[END, "check_query"]`
Conditional edge that determines if the generated query requires validation. Routes to query checker if tool calls are present, otherwise proceeds directly to execution.
"""
logger.info("### Control Flow")


def need_checker(state: MessagesState) -> Literal[END, "check_query"]:
    """Conditional edge: run checker if tool call present"""
    return "check_query" if state["messages"][-1].tool_calls else END


"""
## Custom LangGraph Agent Creation

### `create_langgraph_agent_with_enhanced_memory()`

Creates a custom LangGraph state machine agent with a deterministic, step-by-step workflow for MongoDB queries. Provides enhanced control and debuggability compared to the ReAct agent.

**Components:**
- **State Graph** with 7 distinct nodes for different operations
- **Linear workflow** with one conditional branch for query validation
- **LLM-powered checkpointer** for conversation memory and step summarization

**Workflow:**
```
START → list_collections → call_get_schema → get_schema → generate_query
                                                              ↓
                                                         need_checker?
                                                         ↙         ↘
                                                  check_query    run_query
                                                       ↓             ↓
                                                   run_query   format_answer
                                                                     ↓
                                                                   END
```

**Key Features:**
- **Deterministic flow**: Each step occurs in predictable order
- **Conditional validation**: Queries checked only when required
- **Memory persistence**: Complete conversation state saved with LLM summaries
- **Debuggable**: Individual nodes can be inspected or modified

**Returns:** Compiled LangGraph agent ready for execution
"""
logger.info("## Custom LangGraph Agent Creation")


def create_langgraph_agent_with_enhanced_memory():
    """Create custom LangGraph agent with LLM-powered summarizing checkpointer"""
    summarizing_checkpointer = LLMSummarizingMongoDBSaver(client, llm)

    g = StateGraph(MessagesState)

    g.add_node("list_collections", list_collections)
    g.add_node("call_get_schema", call_get_schema)
    g.add_node("get_schema", schema_node)
    g.add_node("generate_query", generate_query)
    g.add_node("check_query", check_query)
    g.add_node("run_query", run_node)
    g.add_node("format_answer", format_answer)

    g.add_edge(START, "list_collections")
    g.add_edge("list_collections", "call_get_schema")
    g.add_edge("call_get_schema", "get_schema")
    g.add_edge("get_schema", "generate_query")
    g.add_conditional_edges("generate_query", need_checker)
    g.add_edge("check_query", "run_query")
    g.add_edge("run_query", "format_answer")
    # Direct to END - checkpoints handle persistence
    g.add_edge("format_answer", END)

    return g.compile(checkpointer=summarizing_checkpointer)


"""
# Agent Initialization

### Creating Both Agent Types
```python
react_agent_with_memory = create_react_agent_with_enhanced_memory()
mongo_agent_with_memory = create_langgraph_agent_with_enhanced_memory()
```

This section instantiates both agent variants:
- **ReAct Agent**: Uses LangChain's prebuilt ReAct pattern for dynamic reasoning
- **LangGraph Agent**: Uses the custom state machine workflow for deterministic processing

Both agents share:
- **MongoDB toolkit** for schema, query, and validation operations
- **LLM-powered checkpointer** for conversation memory
- **Intelligent step summarization** for debugging

### System Capabilities

Key improvements over standard MongoDB agents:

- **Database flexibility**: Works with any MongoDB database beyond sample datasets
- **LLM intelligence**: Uses GPT models to understand and summarize agent behavior  
- **Adaptive processing**: Handles any natural language query pattern automatically
- **Natural language logs**: Step summaries are human-readable rather than technical
- **Performance optimization**: Caches LLM summaries to reduce API calls and latency

### Usage Options

- Use `react_agent_with_memory` for **flexible, autonomous reasoning**
- Use `mongo_agent_with_memory` for **predictable, step-by-step processing**

Both maintain complete conversation context and provide intelligent summarization for debugging and optimization.
"""
logger.info("# Agent Initialization")

react_agent_with_memory = create_react_agent_with_enhanced_memory()
mongo_agent_with_memory = create_langgraph_agent_with_enhanced_memory()

logger.debug("✅ Agents created with LLM-powered summarization!")
logger.debug("\n📖 Features:")
logger.debug("• Works with any MongoDB database and collection")
logger.debug("• Uses LLM to intelligently summarize each step")
logger.debug("• Adapts to any query type automatically")
logger.debug("• Provides natural language step descriptions")
logger.debug("• Caches summaries for better performance")

"""
## Agent Execution Functions

### `execute_react_with_memory(thread_id: str, user_input: str)`

Executes the ReAct agent with conversation persistence and streams results with formatted output.

**Parameters:**
- `thread_id`: Unique identifier for the conversation thread (enables memory)
- `user_input`: Natural language query to process

**Functionality:**
- Configures the agent to use the specified thread for memory persistence
- Displays execution header with thread ID, query, and agent type
- Streams the agent's execution in real-time using `stream_mode="values"`
- Formats each message as it's generated (tool calls, responses, etc.)

**Example:**
```python
execute_react_with_memory("session_1", "Count all movies from 2020")
```
"""
logger.info("## Agent Execution Functions")


def execute_react_with_memory(thread_id: str, user_input: str):
    """Execute ReAct agent with persistent memory"""
    config = {"configurable": {"thread_id": thread_id}}

    logger.debug(f"🧵 Thread: {thread_id}")
    logger.debug(f"❓ Query: {user_input}")
    logger.debug("🔄 Agent: ReAct")
    logger.debug("=" * 50)

    events = react_agent_with_memory.stream(
        {"messages": [("user", user_input)]}, config, stream_mode="values"
    )

    for event in events:
        event["messages"][-1].pretty_logger.debug()


"""
### `execute_graph_with_memory(thread_id: str, user_input: str)`

Executes the custom LangGraph agent with the same memory and streaming capabilities.

**Parameters:**
- `thread_id`: Unique identifier for the conversation thread
- `user_input`: Natural language query to process

**Key Differences from ReAct:**
- Uses the deterministic state machine workflow
- Input format is `{"messages": [{"role": "user", "content": user_input}]}`
- Each workflow step is visible as it executes

**Usage:**
Both functions provide identical interfaces but use different agent implementations. The LangGraph version provides visibility into the step-by-step workflow, while ReAct offers more autonomous reasoning.

**Memory Persistence:**
Both functions automatically save conversation state to MongoDB, enabling follow-up queries in the same thread to reference previous interactions.
"""
logger.info("### `execute_graph_with_memory(thread_id: str, user_input: str)`")


def execute_graph_with_memory(thread_id: str, user_input: str):
    """Execute LangGraph agent with persistent memory"""
    config = {"configurable": {"thread_id": thread_id}}

    logger.debug(f"🧵 Thread: {thread_id}")
    logger.debug(f"❓ Query: {user_input}")
    logger.debug("📊 Agent: Custom LangGraph")
    logger.debug("=" * 50)

    for step in mongo_agent_with_memory.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    ):
        step["messages"][-1].pretty_logger.debug()


"""
# Memory Management Functions

**Typical debugging sequence:**
1. `memory_system_stats()` - Check overall system health
2. `list_conversation_threads()` - View all available threads  
3. `inspect_thread_history("thread_id")` - Debug specific conversations
4. `clear_thread_history("thread_id")` - Clean up old or problematic threads

These functions provide complete visibility and control over the agent's memory system.

### `list_conversation_threads()`

Lists all available conversation threads stored in the MongoDB checkpoint database.

**Output:**
- All unique thread IDs that have been created
- Total number of checkpoints across all threads
- Number of checkpoints per individual thread

**Example output:**
```
Available Conversation Threads:
Total checkpoints: 147
==================================================
  1. Thread: session_123
     └─ 12 checkpoints
  2. Thread: demo_basic_1
     └─ 8 checkpoints
  3. Thread: interactive_abc
     └─ 25 checkpoints
```
**Usage:** `list_conversation_threads()`
"""
logger.info("# Memory Management Functions")


def list_conversation_threads():
    """List all available conversation threads"""
    try:
        db_checkpoints = client["checkpointing_db"]
        collection = db_checkpoints.checkpoints

        threads = collection.distinct("thread_id")
        total_checkpoints = collection.count_documents({})

        logger.debug("📋 Available Conversation Threads:")
        logger.debug(f"📊 Total checkpoints: {total_checkpoints}")
        logger.debug("=" * 50)

        for i, thread_id in enumerate(threads, 1):
            count = collection.count_documents({"thread_id": thread_id})
            logger.debug(f"  {i}. Thread: {thread_id}")
            logger.debug(f"     └─ {count} checkpoints")

        return threads

    except Exception as e:
        logger.debug(f"❌ Error listing threads: {e}")
        return []


"""
### `inspect_thread_history(thread_id: str, limit: int = 10)`

Inspects the conversation history for a specific thread, showing step-by-step execution details.

**Features:**
- **Smart fallback**: Uses enhanced inspection with LLM summaries if available, otherwise falls back to basic checkpoint analysis
- **Configurable limit**: Control how many recent steps to display
- **Detailed breakdown**: Shows messages, tool calls, and content for each step

**Parameters:**
- `thread_id`: The conversation thread to inspect
- `limit`: Maximum number of recent checkpoints to show (default: 10)

**Usage:** `inspect_thread_history("session_123", limit=5)`
"""
logger.info("### `inspect_thread_history(thread_id: str, limit: int = 10)`")


def inspect_thread_history(thread_id: str, limit: int = 10):
    """Inspect conversation history for a specific thread"""
    try:
        return inspect_thread_with_summaries_enhanced(thread_id, limit)
    except NameError:
        try:
            db_checkpoints = client["checkpointing_db"]
            collection = db_checkpoints.checkpoints

            checkpoints = list(
                collection.find({"thread_id": thread_id})
                .sort("checkpoint_ns", -1)
                .limit(limit)
            )

            if not checkpoints:
                logger.debug(f"❌ No checkpoints found for thread: {thread_id}")
                return []

            logger.debug(f"🔍 Thread History: {thread_id}")
            logger.debug(
                f"📊 Showing {len(checkpoints)} most recent checkpoints")
            logger.debug("=" * 60)

            for i, checkpoint in enumerate(reversed(checkpoints), 1):
                logger.debug(f"\n📍 Step {i}:")

                channel_values = checkpoint.get("channel_values", {})
                if "messages" in channel_values:
                    messages = channel_values["messages"]
                    logger.debug(f"   Messages: {len(messages)} total")

                    if messages:
                        last_msg = messages[-1]
                        if isinstance(last_msg, dict):
                            content = last_msg.get("content", "")
                            tool_calls = last_msg.get("tool_calls", [])

                            if tool_calls:
                                tool_name = tool_calls[0].get(
                                    "name", "unknown")
                                logger.debug(f"   🔧 Tool Call: {tool_name}")
                            elif content:
                                preview = (
                                    content[:100] + "..."
                                    if len(content) > 100
                                    else content
                                )
                                logger.debug(f"   💬 Content: {preview}")

            return checkpoints

        except Exception as e:
            logger.debug(f"❌ Error inspecting thread: {e}")
            return []


"""
### `clear_thread_history(thread_id: str)`

Completely removes all conversation history for a specific thread from MongoDB.

**What it clears:**
- Main checkpoints collection (conversation state)
- Checkpoint writes collection (operation logs)

**Warning:** This action is irreversible. The agent will lose all memory of previous interactions in this thread.

**Usage:** `clear_thread_history("old_session_456")`
"""
logger.info("### `clear_thread_history(thread_id: str)`")


def clear_thread_history(thread_id: str):
    """Clear conversation history for a specific thread"""
    try:
        db_checkpoints = client["checkpointing_db"]

        collection = db_checkpoints.checkpoints
        result = collection.delete_many({"thread_id": thread_id})
        logger.debug(
            f"🗑️ Cleared {result.deleted_count} checkpoints from thread: {thread_id}")

        writes_collection = db_checkpoints.checkpoint_writes
        writes_result = writes_collection.delete_many({"thread_id": thread_id})
        logger.debug(
            f"🗑️ Cleared {writes_result.deleted_count} checkpoint writes")

    except Exception as e:
        logger.debug(f"❌ Error clearing thread: {e}")


"""
### `memory_system_stats()`

Provides a comprehensive overview of the entire memory system's usage and health.

**Metrics displayed:**
- Total checkpoints across all threads
- Total checkpoint writes (operation logs)
- Number of unique conversation threads
- Database name being used

**Example output:**
```
Memory System Statistics
========================================
Total checkpoints: 147
Total checkpoint writes: 298
Total conversation threads: 8
Database: checkpointing_db
```

**Returns:** Dictionary with stats for programmatic use

**Usage:** `stats = memory_system_stats()`
"""
logger.info("### `memory_system_stats()`")


def memory_system_stats():
    """Show comprehensive memory statistics"""
    try:
        db_checkpoints = client["checkpointing_db"]
        checkpoints = db_checkpoints.checkpoints
        checkpoint_writes = db_checkpoints.checkpoint_writes

        total_checkpoints = checkpoints.count_documents({})
        total_writes = checkpoint_writes.count_documents({})
        total_threads = len(checkpoints.distinct("thread_id"))

        logger.debug("📊 Memory System Statistics")
        logger.debug("=" * 40)
        logger.debug(f"💾 Total checkpoints: {total_checkpoints}")
        logger.debug(f"✍️ Total checkpoint writes: {total_writes}")
        logger.debug(f"🧵 Total conversation threads: {total_threads}")
        logger.debug("🏛️ Database: checkpointing_db")

        return {
            "checkpoints": total_checkpoints,
            "writes": total_writes,
            "threads": total_threads,
        }

    except Exception as e:
        logger.debug(f"❌ Error getting stats: {e}")
        return {}


"""
# Demonstration Functions

This section provides ready-to-run examples that showcase different aspects of the Text-to-MQL system.

### Running Demos

Each function is self-contained and generates unique thread IDs to avoid conflicts. They provide formatted output showing:
- Query execution in real-time
- Step-by-step agent reasoning
- Final results and analysis
- Memory inspection summaries

**Quick start:** Run `test_enhanced_summarization()` to see the complete system in action with intelligent step tracking.

### `demo_basic_queries()`

Demonstrates core text-to-MQL functionality with 5 standalone queries of increasing complexity.

**Query types:**
- Top movies by IMDb rating
- Most active commenters  
- Theater distribution by state
- Westernmost theaters (geospatial)
- Complex director analysis with multiple criteria

**Purpose:** Shows the range of query types the system can handle, from simple sorting to complex aggregations.

**Usage:** `demo_basic_queries()`
"""
logger.info("# Demonstration Functions")


def demo_basic_queries():
    """Demonstrate basic text-to-MQL functionality"""
    logger.debug("🎬 DEMO: Basic Text-to-MQL Queries")
    logger.debug("=" * 50)

    queries = [
        "List the top 5 movies with highest IMDb ratings",
        "Who are the top 10 most active commenters?",
        "Which states have the most theaters?",
        "Which theaters are furthest west?",
        "Find directors with ≥20 films, highest avg IMDb rating (top-5)",
    ]

    for i, query in enumerate(queries, 1):
        thread_id = f"demo_basic_{i}"
        logger.debug(f"\n--- Demo Query {i} ---")
        logger.debug(f"Query: {query}")
        logger.debug()

        execute_graph_with_memory(thread_id, query)

        if i < len(queries):
            logger.debug("\n" + "=" * 50)


"""
### `demo_conversation_memory()`

Demonstrates multi-turn conversation where each query builds on previous results.

**Conversation flow:**
1. "List the top 3 directors by movie count"
2. "What was the movie count for the first director?" *(references previous result)*
3. "Show me movies by that director with highest ratings" *(continues context)*

**Key feature:** Shows how the agent remembers previous results and can answer follow-up questions without re-querying.

**Usage:** `demo_conversation_memory()`
"""
logger.info("### `demo_conversation_memory()`")


def demo_conversation_memory():
    """Demonstrate conversation memory across multiple related queries"""
    thread_id = f"conversation_demo_{uuid.uuid4().hex[:8]}"

    logger.debug("🎬 DEMO: Conversation Memory with Text-to-MQL")
    logger.debug("=" * 50)

    conversation = [
        "List the top 3 directors by movie count",
        "What was the movie count for the first director?",
        "Show me movies by that director with highest ratings",
    ]

    for i, query in enumerate(conversation, 1):
        logger.debug(f"\n--- Conversation Step {i} ---")
        execute_graph_with_memory(thread_id, query)

        if i < len(conversation):
            logger.debug("\n🔄 Building context for next query...")
            logger.debug("=" * 40)

    logger.debug("\n🔍 Complete Conversation Analysis:")
    logger.debug("=" * 40)
    inspect_thread_history(thread_id)


"""
### `compare_agents_with_memory()`

Side-by-side comparison of ReAct vs LangGraph agents using the same complex query.

**Comparison points:**
- **Execution style**: ReAct's autonomous reasoning vs LangGraph's structured workflow
- **Memory patterns**: How each agent stores conversation state
- **Output format**: Differences in result presentation

**Usage:** `compare_agents_with_memory()`
"""
logger.info("### `compare_agents_with_memory()`")

"""## Enhanced Agent Comparison Functions


Comprehensive comparison of ReAct vs LangGraph agents with configurable parameters and robust error handling.

**Parameters:**
- `query`: Natural language query to test with both agents
- `max_retries`: Maximum retry attempts if an agent fails (default: 3)
- `recursion_limit`: Maximum recursion depth to prevent infinite loops (default: 50)

**Comparison Analysis:**
- **Execution Style**: ReAct's autonomous reasoning vs LangGraph's structured workflow
- **Memory Patterns**: How each agent stores conversation state
- **Performance Metrics**: Success rates, execution time, and retry attempts
- **Error Handling**: How each agent responds to failures and complex queries

**Features:**
- Retry logic with fresh threads for each attempt
- Configurable recursion limits to prevent infinite loops
- Detailed execution step tracking and analysis
- Performance timing and success rate comparison
- Memory pattern inspection for successful executions
- Intelligent recommendations based on results

**Usage Examples:**
```python
compare_agents_with_memory("Count all movies in the database")

compare_agents_with_memory(
    "Find the top 5 directors with most award wins and at least 5 movies",
    max_retries=3,
    recursion_limit=50
)

compare_agents_with_memory("List top directors by movie count", max_retries=2, recursion_limit=40)
```

**Return Value:** Dictionary containing detailed results for both agents including success status, execution metrics, and configuration used.
"""


def compare_agents_with_memory(
    query: str, max_retries: int = 3, recursion_limit: int = 50
):
    """
    Side-by-side comparison of ReAct vs LangGraph agents using a specified query.

    Parameters:
    -----------
    query : str
        The natural language query to test with both agents
    max_retries : int, default=3
        Maximum number of retry attempts if an agent fails
    recursion_limit : int, default=50
        Maximum recursion depth for the ReAct agent to prevent infinite loops

    Comparison points:
    -----------------
    - Execution style: ReAct's autonomous reasoning vs LangGraph's structured workflow
    - Memory patterns: How each agent stores conversation state
    - Output format: Differences in result presentation
    - Error handling: How each agent responds to failures
    """
    base_thread = f"compare_{uuid.uuid4().hex[:8]}"

    logger.debug("Agent Comparison: ReAct vs LangGraph")
    logger.debug("=" * 60)
    logger.debug(f"Query: {query}")
    logger.debug(f"Max Retries: {max_retries}")
    logger.debug(f"Recursion Limit: {recursion_limit}")
    logger.debug("=" * 60)

    react_results = {
        "success": False,
        "attempts": 0,
        "error": None,
        "execution_time": None,
    }
    graph_results = {
        "success": False,
        "attempts": 0,
        "error": None,
        "execution_time": None,
    }

    logger.debug("\nReAct Agent Execution:")
    logger.debug("-" * 40)

    start_time = time.time()

    for attempt in range(max_retries):
        react_results["attempts"] = attempt + 1
        thread_id = f"{base_thread}_react_attempt_{attempt + 1}"

        logger.debug(f"\nAttempt {attempt + 1}/{max_retries}")
        logger.debug(f"Thread: {thread_id}")

        try:
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": recursion_limit,
            }

            step_count = 0
            events = react_agent_with_memory.stream(
                {"messages": [("user", query)]}, config, stream_mode="values"
            )

            logger.debug("Execution steps:")
            for event in events:
                step_count += 1
                logger.debug(f"  Step {step_count}:", end=" ")

                last_msg = event["messages"][-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    tool_name = last_msg.tool_calls[0]["name"]
                    logger.debug(f"Tool call: {tool_name}")
                elif hasattr(last_msg, "content") and last_msg.content:
                    content_preview = last_msg.content[:50] + (
                        "..." if len(last_msg.content) > 50 else ""
                    )
                    logger.debug(f"Response: {content_preview}")
                else:
                    logger.debug("Processing...")

                if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
                    logger.debug("\nFinal ReAct Response:")
                    last_msg.pretty_logger.debug()

                if step_count > recursion_limit - 5:
                    logger.debug(
                        f"\nApproaching recursion limit at step {step_count}")
                    break

            react_results["success"] = True
            react_results["execution_time"] = time.time() - start_time
            logger.debug(f"\nReAct agent succeeded in {step_count} steps")
            break

        except Exception as e:
            react_results["error"] = str(e)
            logger.debug(f"\nReAct attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                logger.debug("Retrying with fresh thread...")
            else:
                logger.debug("Max retries reached for ReAct agent")
                react_results["execution_time"] = time.time() - start_time

    logger.debug("\nLangGraph Agent Execution:")
    logger.debug("-" * 40)

    start_time = time.time()

    for attempt in range(max_retries):
        graph_results["attempts"] = attempt + 1
        thread_id = f"{base_thread}_graph_attempt_{attempt + 1}"

        logger.debug(f"\nAttempt {attempt + 1}/{max_retries}")
        logger.debug(f"Thread: {thread_id}")

        try:
            config = {"configurable": {"thread_id": thread_id}}

            step_count = 0
            logger.debug("Execution steps:")
            for step in mongo_agent_with_memory.stream(
                {"messages": [{"role": "user", "content": query}]},
                config,
                stream_mode="values",
            ):
                step_count += 1
                last_msg = step["messages"][-1]

                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    tool_name = last_msg.tool_calls[0]["name"]
                    logger.debug(
                        f"  Step {step_count}: Tool call: {tool_name}")
                elif hasattr(last_msg, "content") and last_msg.content:
                    content_preview = last_msg.content[:50] + (
                        "..." if len(last_msg.content) > 50 else ""
                    )
                    logger.debug(
                        f"  Step {step_count}: Response: {content_preview}")

                if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
                    logger.debug("\nFinal LangGraph Response:")
                    last_msg.pretty_logger.debug()

            graph_results["success"] = True
            graph_results["execution_time"] = time.time() - start_time
            logger.debug(f"\nLangGraph agent succeeded in {step_count} steps")
            break

        except Exception as e:
            graph_results["error"] = str(e)
            logger.debug(f"\nLangGraph attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                logger.debug("Retrying with fresh thread...")
            else:
                logger.debug("Max retries reached for LangGraph agent")
                graph_results["execution_time"] = time.time() - start_time

    logger.debug("\nComparison Summary:")
    logger.debug("=" * 60)

    logger.debug("\nReAct Agent Results:")
    logger.debug(f"  Success: {'✅' if react_results['success'] else '❌'}")
    logger.debug(f"  Attempts: {react_results['attempts']}/{max_retries}")
    logger.debug(
        f"  Execution Time: {react_results['execution_time']:.2f}s"
        if react_results["execution_time"]
        else "  Execution Time: N/A"
    )
    if react_results["error"]:
        logger.debug(f"  Final Error: {react_results['error']}")

    logger.debug("\nLangGraph Agent Results:")
    logger.debug(f"  Success: {'✅' if graph_results['success'] else '❌'}")
    logger.debug(f"  Attempts: {graph_results['attempts']}/{max_retries}")
    logger.debug(
        f"  Execution Time: {graph_results['execution_time']:.2f}s"
        if graph_results["execution_time"]
        else "  Execution Time: N/A"
    )
    if graph_results["error"]:
        logger.debug(f"  Final Error: {graph_results['error']}")

    logger.debug("\nExecution Style Analysis:")
    logger.debug("  ReAct Agent:")
    logger.debug("    - Autonomous reasoning and tool selection")
    logger.debug("    - Dynamic decision making based on previous results")
    logger.debug("    - Can get stuck in reasoning loops with complex queries")
    logger.debug("    - More flexible but less predictable workflow")

    logger.debug("  LangGraph Agent:")
    logger.debug("    - Structured, deterministic workflow")
    logger.debug("    - Predefined step sequence with conditional branches")
    logger.debug("    - Better error isolation and recovery")
    logger.debug("    - More predictable but less flexible execution")

    if react_results["success"] or graph_results["success"]:
        logger.debug("\nMemory Pattern Analysis:")

        if react_results["success"]:
            logger.debug("  ReAct Agent Memory:")
            react_thread = f"{base_thread}_react_attempt_{react_results['attempts']}"
            try:
                inspect_thread_history(react_thread, limit=3)
            except Exception as e:
                logger.debug("Unable to inspect ReAct memory")

        if graph_results["success"]:
            logger.debug("  LangGraph Agent Memory:")
            graph_thread = f"{base_thread}_graph_attempt_{graph_results['attempts']}"
            try:
                inspect_thread_history(graph_thread, limit=3)
            except Exception as e:
                logger.debug("Unable to inspect LangGraph memory")

    logger.debug("\nRecommendations:")
    if react_results["success"] and graph_results["success"]:
        if react_results["execution_time"] < graph_results["execution_time"]:
            logger.debug("  - ReAct agent was faster for this query")
        else:
            logger.debug(
                "  - LangGraph agent was more efficient for this query")
        logger.debug("  - Both agents handled the query successfully")
    elif graph_results["success"] and not react_results["success"]:
        logger.debug("  - Use LangGraph agent for this type of query")
        logger.debug(
            "  - ReAct agent struggled with the complexity/validation")
    elif react_results["success"] and not graph_results["success"]:
        logger.debug("  - ReAct agent was more robust for this query")
        logger.debug("  - Consider debugging LangGraph workflow")
    else:
        logger.debug(
            "  - Query may be too complex or have data structure issues")
        logger.debug(
            "  - Consider simplifying the query or debugging the dataset")

    return {
        "react": react_results,
        "langgraph": graph_results,
        "query": query,
        "config": {"max_retries": max_retries, "recursion_limit": recursion_limit},
    }


"""
### `test_memory_functionality()`

Simple two-step test focused specifically on memory capabilities.

**Test sequence:**
1. Initial query about directors
2. Follow-up question that requires remembering the first result

**Purpose:** Quick validation that conversation memory is working correctly.

**Usage:** `test_memory_functionality()`
"""
logger.info("### `test_memory_functionality()`")


def test_memory_functionality():
    """Test memory functionality with a simple example"""
    thread_id = f"memory_test_{uuid.uuid4().hex[:8]}"

    logger.debug("🧪 TESTING: Memory Functionality")
    logger.debug("=" * 50)

    logger.debug("Step 1: Ask about directors")
    execute_graph_with_memory(thread_id, "List top 3 directors by movie count")

    logger.debug("\nStep 2: Follow up question (tests memory)")
    execute_graph_with_memory(
        thread_id, "What was the movie count for the first director?"
    )

    logger.debug("\n🔍 Memory Analysis:")
    inspect_thread_history(thread_id)

    return thread_id


"""
### `test_enhanced_summarization()`

Tests the LLM-powered summarization system with various query patterns.

**Functionality:**
- Runs 3 different query types (count, average, top results)
- Executes each with full step tracking
- Displays enhanced thread analysis with LLM-generated summaries

**Purpose:** Validates that the summarization system correctly categorizes and describes different types of operations.

**Usage:** `test_enhanced_summarization()`
"""
logger.info("### `test_enhanced_summarization()`")


def test_enhanced_summarization():
    """Test the enhanced summarization system with various query patterns"""
    logger.debug("\n🧪 TESTING ENHANCED SUMMARIZATION SYSTEM")
    logger.debug("=" * 60)

    thread_id = f"enhanced_test_{uuid.uuid4().hex[:8]}"

    test_queries = [
        "How many movies are in the database?",
        "Find the average rating of all movies",
        "Show me the top 5 directors by movie count",
    ]

    logger.debug(f"Testing thread: {thread_id}")
    logger.debug("Running query patterns with enhanced summarization...")
    logger.debug("=" * 50)

    for i, query in enumerate(test_queries, 1):
        logger.debug(f"\n📌 Test {i}: {query}")
        execute_graph_with_memory(thread_id, query)
        logger.debug(f"✅ Test {i} complete")

    logger.debug("\n🔍 Enhanced Thread Analysis:")
    logger.debug("=" * 50)
    inspect_thread_history(thread_id)

    return thread_id


"""
## Supporting Test Functions

These functions provide pre-configured test scenarios for validating agent comparison functionality across different query complexity levels.

*   `test_simple_comparison()` uses basic counting queries with conservative retry settings,
*   `test_moderate_comparison()` tests standard aggregation patterns,
* `test_complex_comparison()` validates the original problematic query using enhanced error handling
*   `run_comparison_tests()` function executes all three scenarios in sequence, providing comprehensive assessment of both ReAct and LangGraph agent capabilities with automatic error isolation and performance benchmarking.
"""
logger.info("## Supporting Test Functions")


def test_simple_comparison():
    """Test with a simple query that should work"""
    simple_query = "Count the total number of movies in the database"
    return compare_agents_with_memory(simple_query, max_retries=2, recursion_limit=30)


def test_moderate_comparison():
    """Test with a moderately complex query"""
    moderate_query = "List the top 5 directors who have directed the most movies"
    return compare_agents_with_memory(moderate_query, max_retries=2, recursion_limit=40)


def test_complex_comparison():
    """Test with the original complex query that caused issues"""
    complex_query = (
        "Find the top 5 directors with most award wins and at least 5 movies"
    )
    return compare_agents_with_memory(complex_query, max_retries=3, recursion_limit=50)


def run_comparison_tests():
    """Run a series of comparison tests with different query complexities"""
    logger.debug("Running Comparison Test Suite")
    logger.debug("=" * 60)

    tests = [
        ("Simple Query", test_simple_comparison),
        ("Moderate Query", test_moderate_comparison),
        ("Complex Query", test_complex_comparison),
    ]

    results = {}
    for test_name, test_func in tests:
        logger.debug(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.debug(f"❌ {test_name} failed with error: {e}")
            results[test_name] = None

    return results


logger.debug("✅ Enhanced agent comparison functions loaded!")
logger.debug("\nUsage examples:")
logger.debug('compare_agents_with_memory("Count all movies", max_retries=2)')
logger.debug(
    'compare_agents_with_memory("Find top directors", max_retries=3, recursion_limit=40)'
)
logger.debug("run_comparison_tests()  # Run multiple test scenarios")

"""
# Interactive Query Interface

### `interactive_query()`

Provides a command-line interface for real-time interaction with the Text-to-MQL agent. Creates a conversational session where you can ask multiple related questions and manage conversation threads.

**Features:**
- **Persistent conversation**: Maintains context across multiple queries in the same thread
- **Thread management**: Switch between different conversation contexts
- **Built-in debugging**: Inspect conversation history without leaving the interface
- **Error handling**: Graceful handling of interruptions and errors

### Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `<natural language>` | Execute MongoDB query | `"Count movies from 2020"` |
| `exit` | Quit the interface | `exit` |
| `threads` | List all conversation threads | `threads` |
| `switch <thread_id>` | Change to different thread | `switch session_123` |
| `debug` | Inspect current thread history | `debug` |

### Interactive Session Example

```
Interactive Text-to-MQL Query Interface
Commands: 'exit' to quit, 'threads' to list, 'switch <thread>' to change thread
======================================================================

[interactive_abc123] Enter your query: Count all movies in the database

Thread: interactive_abc123
Query: Count all movies in the database
Agent: Custom LangGraph
==================================================
[Agent execution with step-by-step output...]

[interactive_abc123] Enter your query: What about just movies from 2020?

[Continues conversation with memory of previous query...]

[interactive_abc123] Enter your query: debug

Thread History: interactive_abc123
Total steps: 8
================================================================================
[Shows conversation history...]

[interactive_abc123] Enter your query: exit
Goodbye!
```

### Session Management

**Automatic thread creation:** Each session starts with a unique thread ID (`interactive_<random>`)

**Thread switching:** Use `switch <thread_id>` to continue previous conversations:
```
[interactive_abc123] Enter your query: switch session_older
Switched to thread: session_older
[session_older] Enter your query: What did we discuss last time?
```

**Memory persistence:** All queries and results are saved to MongoDB, allowing you to return to any conversation later.

### Usage

**Start interactive session:** `interactive_query()`

**Best practices:**
- Use meaningful thread names when switching (`switch movie_analysis_2024`)
- Use `debug` command to review conversation context
- Use `threads` to see all available conversation histories

This interface is ideal for exploratory data analysis sessions where you want to ask follow-up questions and build on previous results.
"""
logger.info("# Interactive Query Interface")


def interactive_query():
    """Interactive query interface with memory"""
    logger.debug("🔍 Interactive Text-to-MQL Query Interface")
    logger.debug(
        "Commands: 'exit' to quit, 'threads' to list, 'switch <thread>' to change thread"
    )
    logger.debug("=" * 70)

    thread_id = f"interactive_{uuid.uuid4().hex[:8]}"

    while True:
        try:
            user_input = input(f"\n[{thread_id}] Enter your query: ").strip()

            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "threads":
                list_conversation_threads()
                continue
            elif user_input.lower().startswith("switch "):
                thread_id = user_input[7:].strip()
                logger.debug(f"🔄 Switched to thread: {thread_id}")
                continue
            elif user_input.lower() == "debug":
                inspect_thread_history(thread_id)
                continue
            elif not user_input:
                continue

            logger.debug()
            execute_graph_with_memory(thread_id, user_input)

        except KeyboardInterrupt:
            logger.debug("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.debug(f"❌ Error: {e}")


"""
# System Initialization and Quick Reference

This section provides the startup summary and quick reference guide for the Text-to-MQL system.

### System Status Display

**Startup sequence:**
```
Text-to-MQL Agent with MongoDB Memory - Ready
============================================================
Memory System Statistics
========================================
Total checkpoints: 0
Total checkpoint writes: 0  
Total conversation threads: 0
Database: checkpointing_db
```

Automatically displays current memory system health and usage statistics.

### Available Functions Reference

**Demonstration Functions:**
- `demo_basic_queries()` - Showcase core text-to-MQL capabilities
- `demo_conversation_memory()` - Multi-turn conversation examples
- `compare_agents_with_memory()` - ReAct vs LangGraph comparison
- `test_memory_functionality()` - Simple memory validation
- `test_enhanced_summarization()` - LLM summarization testing
- `interactive_query()` - Real-time query interface

**Memory Management Tools:**
- `list_conversation_threads()` - View all conversation threads
- `inspect_thread_history(thread_id)` - Debug specific conversations
- `inspect_thread_with_summaries_enhanced(thread_id)` - Enhanced thread analysis
- `clear_thread_history(thread_id)` - Delete conversation history
- `memory_system_stats()` - System health overview

### Quick Start Recommendations

**For first-time users:**
1. `test_enhanced_summarization()` - See the complete system in action
2. `demo_conversation_memory()` - Experience multi-turn conversations  
3. `interactive_query()` - Try your own queries

### System Capabilities Summary

**Core features confirmed operational:**
- **Dual agent architecture**: Both ReAct and LangGraph agents ready
- **LLM-powered memory**: Intelligent step summarization active
- **MongoDB persistence**: Conversation state saved automatically
- **Enhanced debugging**: Human-readable conversation histories

**Key improvements over standard agents:**
- Query categorization using natural language understanding
- Conversation-aware step descriptions  
- Better thread inspection with LLM insights
- Performance-optimized memory debugging

This summary serves as both a system health check and a quick reference guide for exploring the system's capabilities.
"""
logger.info("# System Initialization and Quick Reference")

logger.debug("\n🚀 Text-to-MQL Agent with MongoDB Memory - Ready!")
logger.debug("=" * 60)

memory_system_stats()

"""
## Initial Test Execution

### Automatic Startup Test

```python
if __name__ == "__main__":
    # Start with the enhanced summarization test
    test_enhanced_summarization()
```

**Purpose:** When the notebook/script is run directly, automatically executes a demonstration to verify the system is working correctly.

**What happens:**
1. **System initialization**: All agents and memory components are loaded
2. **Test execution**: Runs `test_enhanced_summarization()` which:
   - Creates a new conversation thread
   - Executes 3 different query patterns
   - Demonstrates LLM-powered step summarization
   - Shows enhanced thread inspection capabilities

**Expected output:**
```
Testing Enhanced Summarization System
============================================================
Testing thread: enhanced_test_abc12345
Running query patterns with enhanced summarization...
==================================================

Test 1: How many movies are in the database?
[Agent execution with step-by-step summaries...]
Test 1 complete

Test 2: Find the average rating of all movies
[Agent execution...]
Test 2 complete

Test 3: Show me the top 5 directors by movie count
[Agent execution...]
Test 3 complete

Enhanced Thread Analysis:
==================================================
[Thread history with LLM-generated summaries...]
```

**Validation checks:**
- MongoDB connection working
- Ollama API accessible
- Agent workflow functioning
- Memory persistence active
- LLM summarization operational

**Note:** In Colab notebooks, this section typically won't auto-execute since notebooks run cell-by-cell. You can manually run `test_enhanced_summarization()` to perform the same validation.

This serves as a **smoke test** to ensure all system components are properly initialized and functioning before manual exploration.
"""
logger.info("## Initial Test Execution")

if __name__ == "__main__":
    test_enhanced_summarization()

"""
# Demos

## Demo 1: Run Basic Queries w/ `demo_basic_queries()`
"""
logger.info("# Demos")

demo_basic_queries()

"""
## Demo 2: Multi-turn conversations - `demo_conversation_memory()`
"""
logger.info("## Demo 2: Multi-turn conversations - `demo_conversation_memory()`")

demo_conversation_memory()

"""
## Demo 3: Enhanced Agent Comparison with Different Query Complexities"""
logger.info(
    "## Demo 3: Enhanced Agent Comparison with Different Query Complexities")
"""

logger.debug("📊 Demo 3a: Simple Query Comparison")
logger.debug("=" * 50)
compare_agents_with_memory("Count all movies in the database", max_retries=2)

logger.debug("\n" + "=" * 80 + "\n")

logger.debug("📊 Demo 3b: Moderate Complexity Comparison")
logger.debug("=" * 50)
compare_agents_with_memory(
    "List the top 5 directors by movie count", max_retries=2, recursion_limit=40
)

logger.debug("\n" + "=" * 80 + "\n")

logger.debug("📊 Demo 3c: Complex Query with Enhanced Error Handling")
logger.debug("=" * 50)
compare_agents_with_memory(
    "Find the top 5 directors with most award wins and at least 5 movies",
    max_retries=3,
    recursion_limit=50,
)

"""  # Demo 3d: Comprehensive Test Suite"""

logger.debug("\n" + "=" * 80 + "\n")
logger.debug("📊 Demo 3d: Comprehensive Agent Test Suite")
logger.debug("=" * 50)

results = run_comparison_tests()

logger.debug("\nTest Suite Summary:")
logger.debug("=" * 30)
for test_name, result in results.items():
    if result:
        react_success = "✅" if result["react"]["success"] else "❌"
        graph_success = "✅" if result["langgraph"]["success"] else "❌"
        logger.debug(
            f"{test_name}: ReAct {react_success} | LangGraph {graph_success}")
    else:
        logger.debug(f"{test_name}: ❌ Test Failed")

"""
## Demo 4: List all threads - `list_conversation_threads()`
"""
logger.info("## Demo 4: List all threads - `list_conversation_threads()`")

list_conversation_threads()

"""
## Demo 5: Enhanced inspection - `inspect_thread_with_summaries_enhanced(thread_id)`
"""
logger.info(
    "## Demo 5: Enhanced inspection - `inspect_thread_with_summaries_enhanced(thread_id)`")


"""
## Demo 6: Interactive Query Interface
"""
logger.info("## Demo 6: Interactive Query Interface")

interactive_query()

logger.info("\n\n[DONE]", bright=True)
