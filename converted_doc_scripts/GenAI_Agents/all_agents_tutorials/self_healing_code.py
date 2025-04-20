from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage
from jet.llm.ollama.base_langchain import ChatOllama
import chromadb
from pydantic import BaseModel
from typing import Optional, Callable
import uuid
import json
import os
import types
import inspect
import sys

initialize_ollama_settings()

"""
# Secret Agents: A Self-Healing Codebase Agentic Workflow

## Overview
This code implements a workflow-based error detection and correction system that combines LangGraph, LLM capabilities, and vector database technology to detect runtime errors, generate fixes, and maintain a memory of bug patterns. The system takes function definitions and runtime arguments, processes them through a graph-based workflow, and maintains a hierarchical error management system enriched by vector-based similarity search.

## Motivation
Several key factors motivate this implementation:

1. **Automated Error Resolution**
   - Manual debugging is time-consuming and error-prone
   - Automated fix generation streamlines the correction process
   - LLMs can provide context-aware code repairs

2. **Pattern-Based Learning**
   - Vector databases enable similarity-based bug pattern recognition
   - Previous fixes can inform future error resolution
   - Semantic search capabilities improve fix relevance

3. **Structured Bug Knowledge**
   - Vector embeddings capture semantic relationships between errors
   - ChromaDB enables efficient storage and retrieval of bug patterns
   - Hierarchical error categorization through vector spaces

4. **Runtime Code Modification**
   - Safe deployment of generated fixes
   - State tracking during modifications
   - Validation of applied patches

## Key Components
1. **State Management System**: 
   - Maintains workflow state using Pydantic models
   - Tracks function references, errors, and fixes
   - Ensures type safety and execution validation

2. **LLM Integration**: 
   - Leverages LLM for code analysis and generation
   - Produces fixes based on error types:
     - Runtime Errors
     - Logic Errors
     - Type Errors
     - Resource Errors

3. **Vector-Based Memory System**:
   - Uses ChromaDB for efficient storage
   - Enables semantic search of bug patterns
   - Maintains contextual relationships between errors
   - Supports pattern-based learning

4. **Graph-based Workflow**: 
   - Uses LangGraph's StateGraph for orchestration
   - Implements error detection nodes
   - Controls fix generation through edges

## Vector Databases and ChromaDB

### What is a Vector Database?
A vector database is specialized storage system designed to handle high-dimensional vectors, which are mathematical representations of data points. These vectors capture semantic meaning, making them ideal for:
- Similarity search operations
- Pattern recognition
- Semantic relationships
- Nearest neighbor queries

### Why Vector DBs Matter for ML
Vector databases are crucial for modern ML systems because they:
1. Enable semantic search capabilities
2. Support efficient similarity computations
3. Scale well with large datasets
4. Maintain context and relationships
5. Facilitate pattern recognition

### ChromaDB Implementation
ChromaDB provides a lightweight, embedded vector database that offers:
1. Simple API:
```python
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name='bug-reports')
```

2. Easy Data Management:
```python
# Adding documents
collection.add(
    ids=[id],
    documents=[document],
)

# Querying
results = collection.query(
    query_texts=[query],
    n_results=10
)
```

3. Automatic embedding generation
4. Efficient similarity search
5. Zero configuration requirements

## Memory Architecture
The system implements a sophisticated memory architecture:

1. **Vector Storage**:
   - Bug reports converted to embeddings
   - Semantic relationships preserved
   - Efficient similarity search

2. **Pattern Recognition**:
   - Similar bugs identified through vector similarity
   - Historical fixes inform new solutions
   - Pattern evolution tracked over time

3. **Memory Updates**:
   - New patterns integrated into existing knowledge
   - Related patterns merged and refined
   - Obsolete patterns pruned

## Visual Overview
A flowchart representing the design and flow of the workflow.

<div style="max-width:600px;">
    
![image.png](../images/self_healing_code.png)
    
</div>

## Conclusion
This implementation demonstrates a practical approach to automated code healing, enhanced by vector database technology. The system combines graph-based workflow management with LLM capabilities and vector-based pattern recognition, allowing for structured error correction while maintaining clear process control.

Key advantages include:
- Automated error detection and correction
- Semantic pattern recognition
- Efficient similarity-based search
- Safe runtime code modification

Future improvements could focus on:
- Enhanced embedding strategies
- Multi-modal pattern recognition
- Distributed vector storage
- Advanced pattern evolution tracking

This system provides a foundation for building more sophisticated self-healing systems, particularly in applications requiring runtime error correction and pattern learning, with the added benefit of efficient vector-based memory management through ChromaDB.

# Dependencies and Imports
Install dependencies and import libraries.
"""

# %%capture

# !pip install langgraph
# !pip install langgraph-sdk
# !pip install langgraph-checkpoint-sqlite
# !pip install langchain-community
# !pip install langchain-core
# !pip install langchain-openai
# !pip install chromadb


"""
## Clients
Import API keys and instantiate clients.
"""

# os.environ['OPENAI_API_KEY'] = 'YOUR-API-KEY'
llm = ChatOllama(model="llama3.1")

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name='bug-reports')

"""
## Define Agent State
We'll define the state that our agent will maintain throughout its operation.
"""


class State(BaseModel):
    function: Callable
    function_string: str
    arguments: list
    error: bool
    error_description: str = ''
    new_function_string: str = ''
    bug_report: str = ''
    memory_search_results: list = []
    memory_ids_to_update: list = []


"""
## Define Code Healing Node Functions
Now we'll define the code healing node functions that our agent will use: code_execution_node, code_update_node and code_patching_node.
"""


def code_execution_node(state: State):
    ''' Run Arbitrary Code '''
    try:
        logger.debug('\nRunning Arbitrary Function')
        logger.debug('--------------------------\n')
        result = state.function(*state.arguments)
        logger.debug('\n✅ Arbitrary Function Ran Without Error')
        logger.debug(f'Result: {result}')
        logger.debug('---------------------------------------\n')
    except Exception as e:
        logger.debug(f'❌ Function Raised an Error: {e}')
        state.error = True
        state.error_description = str(e)
    return state


def code_update_node(state: State):
    ''' Update Arbitratry Code '''
    prompt = ChatPromptTemplate.from_template(
        'You are tasked with fixing a Python function that raised an error.'
        'Function: {function_string}'
        'Error: {error_description}'
        'You must provide a fix for the present error only.'
        'The bug fix should handle the thrown error case gracefully by returning an error message.'
        'Do not raise an error in your bug fix.'
        'The function must use the exact same name and parameters.'
        'Your response must contain only the function definition with no additional text.'
        'Your response must not contain any additional formatting, such as code delimiters or language declarations.'
    )
    message = HumanMessage(content=prompt.format(
        function_string=state.function_string, error_description=state.error_description))
    new_function_string = llm.invoke([message]).content.strip()

    logger.debug('\n🐛 Buggy Function')
    logger.debug('-----------------\n')
    logger.debug(state.function_string)
    logger.debug('\n🩹 Proposed Bug Fix')
    logger.debug('-------------------\n')
    logger.debug(new_function_string)

    state.new_function_string = new_function_string
    return state


def code_patching_node(state: State):
    ''' Fix Arbitrary Code '''
    try:
        logger.debug('\n*******************')
        logger.debug('\n❤️‍🩹 Patching code...')
        new_code = state.new_function_string

        namespace = {}

        exec(new_code, namespace)

        func_name = state.function.__name__

        new_function = namespace[func_name]

        state.function = new_function
        state.error = False

        result = state.function(*state.arguments)

        logger.debug('...patch complete 😬\n')

    except Exception as e:
        logger.debug(f'...patch failed: {e}')
        logger.debug(f'Error details: {str(e)}')

    logger.debug('******************\n')
    return state


"""
## Define Bug Reporting Node Functions
Now we'll define the bug reporting node functions that our agent will use: bug_report_node, memory_search_node, memory_generation_node and memory_modification_node.
"""


def bug_report_node(state: State):
    ''' Generate Bug Report '''
    prompt = ChatPromptTemplate.from_template(
        'You are tasked with generating a bug report for a Python function that raised an error.'
        'Function: {function_string}'
        'Error: {error_description}'
        'Your response must be a comprehensive string including only crucial information on the bug report'
    )
    message = HumanMessage(content=prompt.format(
        function_string=state.function_string, error_description=state.error_description))
    bug_report = llm.invoke([message]).content.strip()

    logger.debug('\n📝 Generating Bug Report')
    logger.debug('------------------------\n')
    logger.debug(bug_report)

    state.bug_report = bug_report
    return state


def memory_search_node(state: State):
    ''' Find memories relevant to the current bug report '''
    prompt = ChatPromptTemplate.from_template(
        'You are tasked with archiving a bug report for a Python function that raised an error.'
        'Bug Report: {bug_report}.'
        'Your response must be a concise string including only crucial information on the bug report for future reference.'
        'Format: # function_name ## error_description ### error_analysis'
    )

    message = HumanMessage(content=prompt.format(
        bug_report=state.bug_report,
    ))

    response = llm.invoke([message]).content.strip()

    results = collection.query(query_texts=[response])

    logger.debug('\n🔎 Searching bug reports...')
    if results['ids'][0]:
        logger.debug(f'...{len(results["ids"][0])} found.\n')
        logger.debug(results)
        state.memory_search_results = [{'id': results['ids'][0][index], 'memory': results['documents'][0]
                                        [index], 'distance': results['distances'][0][index]} for index, id in enumerate(results['ids'][0])]
    else:
        logger.debug('...none found.\n')

    return state


def memory_filter_node(state: State):
    logger.debug('\n🗑️ Filtering bug reports...')
    for memory in state.memory_search_results:
        if memory['distance'] < 0.3:
            state.memory_ids_to_update.append(memory['id'])

    if state.memory_ids_to_update:
        logger.debug(f'...{len(state.memory_ids_to_update)} selected.\n')
    else:
        logger.debug('...none selected.\n')

    return state


def memory_generation_node(state: State):
    ''' Generate relevant memories based on new bug report '''
    prompt = ChatPromptTemplate.from_template(
        'You are tasked with archiving a bug report for a Python function that raised an error.'
        'Bug Report: {bug_report}.'
        'Your response must be a concise string including only crucial information on the bug report for future reference.'
        'Format: # function_name ## error_description ### error_analysis'
    )

    message = HumanMessage(content=prompt.format(
        bug_report=state.bug_report,
    ))

    response = llm.invoke([message]).content.strip()

    logger.debug('\n💾 Saving Bug Report to Memory')
    logger.debug('------------------------------\n')
    logger.debug(response)

    id = str(uuid.uuid4())
    collection.add(
        ids=[id],
        documents=[response],
    )
    return state


def memory_modification_node(state: State):
    ''' Modify relevant memories based on new interaction '''
    prompt = ChatPromptTemplate.from_template(
        'Update the following memories based on the new interaction:'
        'Current Bug Report: {bug_report}'
        'Prior Bug Report: {memory_to_update}'
        'Your response must be a concise but cumulative string including only crucial information on the current and prior bug reports for future reference.'
        'Format: # function_name ## error_description ### error_analysis'
    )
    memory_to_update_id = state.memory_ids_to_update.pop(0)
    state.memory_search_results.pop(0)
    results = collection.get(ids=[memory_to_update_id])
    memory_to_update = results['documents'][0]
    message = HumanMessage(content=prompt.format(
        bug_report=state.bug_report,
        memory_to_update=memory_to_update,
    ))

    response = llm.invoke([message]).content.strip()

    logger.debug('\nCurrent Bug Report')
    logger.debug('------------------\n')
    logger.debug(memory_to_update)
    logger.debug('\nWill be Replaced With')
    logger.debug('---------------------\n')
    logger.debug(response)

    collection.update(
        ids=[memory_to_update_id],
        documents=[response],
    )

    return state


"""
## Define Edge Functions
Now we'll define the conditional edge function that our agent will use to control the workflow.
"""


def error_router(state: State):
    if state.error:
        return 'bug_report_node'
    else:
        return END


def memory_filter_router(state: State):
    if state.memory_search_results:
        return 'memory_filter_node'
    else:
        return 'memory_generation_node'


def memory_generation_router(state: State):
    if state.memory_ids_to_update:
        return 'memory_modification_node'
    else:
        return 'memory_generation_node'


def memory_update_router(state: State):
    if state.memory_ids_to_update:
        return 'memory_modification_node'
    else:
        return 'code_update_node'


"""
## Build Workflow
Now we'll create our workflow and compile it.
"""

builder = StateGraph(State)

builder.add_node('code_execution_node', code_execution_node)
builder.add_node('code_update_node', code_update_node)
builder.add_node('code_patching_node', code_patching_node)
builder.add_node('bug_report_node', bug_report_node)
builder.add_node('memory_search_node', memory_search_node)
builder.add_node('memory_filter_node', memory_filter_node)
builder.add_node('memory_modification_node', memory_modification_node)
builder.add_node('memory_generation_node', memory_generation_node)


builder.set_entry_point('code_execution_node')
builder.add_conditional_edges('code_execution_node', error_router)
builder.add_edge('bug_report_node', 'memory_search_node')
builder.add_conditional_edges('memory_search_node', memory_filter_router)
builder.add_conditional_edges('memory_filter_node', memory_generation_router)
builder.add_edge('memory_generation_node', 'code_update_node')
builder.add_conditional_edges('memory_modification_node', memory_update_router)

builder.add_edge('code_update_node', 'code_patching_node')
builder.add_edge('code_patching_node', 'code_execution_node')

graph = builder.compile()

"""
# Main Function
Define the function that runs the instanciates the workflow and its state.
"""


def execute_self_healing_code_system(function, arguments):

    state = State(
        error=False,
        function=function,
        function_string=inspect.getsource(function),
        arguments=arguments,
    )

    return graph.invoke(state)


"""
# Run Program
Instanciate the main function and observe outputs.
"""


def process_list(lst, index):
    return lst[index] * 2


def parse_date(date_string):
    year, month, day = date_string.split('-')
    return {'year': int(year), 'month': int(month), 'day': int(day)}


def divide_two_numbers(a, b):
    return a/b


logger.debug("*******************************")
logger.debug("*******************************")
logger.debug("** Testing Division Function **")
logger.debug("*******************************")
logger.debug("*******************************")
execute_self_healing_code_system(divide_two_numbers, [10, 0])
execute_self_healing_code_system(divide_two_numbers, ['a', 0])

logger.debug("**************************************")
logger.debug("**************************************")
logger.debug("** Testing List Processing Function **")
logger.debug("**************************************")
logger.debug("**************************************")
execute_self_healing_code_system(process_list, [[1, 2, 3], 5])
execute_self_healing_code_system(process_list, [None, 1])

logger.debug("***********************************")
logger.debug("***********************************")
logger.debug("** Testing Date Parsing Function **")
logger.debug("***********************************")
logger.debug("***********************************")
execute_self_healing_code_system(parse_date, ["2024/01/01"])
execute_self_healing_code_system(parse_date, ["abc-def-ghi"])

logger.info("\n\n[DONE]", bright=True)
