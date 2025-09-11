from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.messages import (
AIMessage,
BaseMessage,
HumanMessage,
SystemMessage,
ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from typing import Dict
from typing import List, Optional
import os
import shutil
import uuid


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
---
sidebar_position: 2
---

# How to add examples to the prompt for query analysis

As our query analysis becomes more complex, the LLM may struggle to understand how exactly it should respond in certain scenarios. In order to improve performance here, we can [add examples](/docs/concepts/few_shot_prompting/) to the prompt to guide the LLM.

Let's take a look at how we can add examples for a LangChain YouTube video query analyzer.

## Setup
#### Install dependencies
"""
logger.info("# How to add examples to the prompt for query analysis")



"""
#### Set environment variables

We'll use Ollama in this example:
"""
logger.info("#### Set environment variables")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass()

"""
## Query schema

We'll define a query schema that we want our model to output. To make our query analysis a bit more interesting, we'll add a `sub_queries` field that contains more narrow questions derived from the top level question.
"""
logger.info("## Query schema")



sub_queries_description = """\
If the original question contains multiple distinct sub-questions, \
or if there are more generic questions that would be helpful to answer in \
order to answer the original question, write a list of all relevant sub-questions. \
Make sure this list is comprehensive and covers all parts of the original question. \
It's ok if there's redundancy in the sub-questions. \
Make sure the sub-questions are as narrowly focused as possible."""


class Search(BaseModel):
    """Search over a database of tutorial videos about a software library."""

    query: str = Field(
        ...,
        description="Primary similarity search query applied to video transcripts.",
    )
    sub_queries: List[str] = Field(
        default_factory=list, description=sub_queries_description
    )
    publish_year: Optional[int] = Field(None, description="Year video was published")

"""
## Query generation
"""
logger.info("## Query generation")


system = """You are an expert at converting user questions into database queries. \
You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
Given a question, return a list of database queries optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        MessagesPlaceholder("examples", optional=True),
        ("human", "{question}"),
    ]
)
llm = ChatOllama(model="llama3.2")
structured_llm = llm.with_structured_output(Search)
query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm

"""
Let's try out our query analyzer without any examples in the prompt:
"""
logger.info("Let's try out our query analyzer without any examples in the prompt:")

query_analyzer.invoke(
    "what's the difference between web voyager and reflection agents? do both use langgraph?"
)

"""
## Adding examples and tuning the prompt

This works pretty well, but we probably want it to decompose the question even further to separate the queries about Web Voyager and Reflection Agents.

To tune our query generation results, we can add some examples of inputs questions and gold standard output queries to our prompt.
"""
logger.info("## Adding examples and tuning the prompt")

examples = []

question = "What's chat langchain, is it a langchain template?"
query = Search(
    query="What is chat langchain and is it a langchain template?",
    sub_queries=["What is chat langchain", "What is a langchain template"],
)
examples.append({"input": question, "tool_calls": [query]})

question = "How to build multi-agent system and stream intermediate steps from it"
query = Search(
    query="How to build multi-agent system and stream intermediate steps from it",
    sub_queries=[
        "How to build multi-agent system",
        "How to stream intermediate steps from multi-agent system",
        "How to stream intermediate steps",
    ],
)

examples.append({"input": question, "tool_calls": [query]})

question = "LangChain agents vs LangGraph?"
query = Search(
    query="What's the difference between LangChain agents and LangGraph? How do you deploy them?",
    sub_queries=[
        "What are LangChain agents",
        "What is LangGraph",
        "How do you deploy LangChain agents",
        "How do you deploy LangGraph",
    ],
)
examples.append({"input": question, "tool_calls": [query]})

"""
Now we need to update our prompt template and chain so that the examples are included in each prompt. Since we're working with Ollama function-calling, we'll need to do a bit of extra structuring to send example inputs and outputs to the model. We'll create a `tool_example_to_messages` helper function to handle this for us:
"""
logger.info("Now we need to update our prompt template and chain so that the examples are included in each prompt. Since we're working with Ollama function-calling, we'll need to do a bit of extra structuring to send example inputs and outputs to the model. We'll create a `tool_example_to_messages` helper function to handle this for us:")




def tool_example_to_messages(example: Dict) -> List[BaseMessage]:
    messages: List[BaseMessage] = [HumanMessage(content=example["input"])]
    ollama_tool_calls = []
    for tool_call in example["tool_calls"]:
        ollama_tool_calls.append(
            {
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    "name": tool_call.__class__.__name__,
                    "arguments": tool_call.json(),
                },
            }
        )
    messages.append(
        AIMessage(content="", additional_kwargs={"tool_calls": ollama_tool_calls})
    )
    tool_outputs = example.get("tool_outputs") or [
        "You have correctly called this tool."
    ] * len(ollama_tool_calls)
    for output, tool_call in zip(tool_outputs, ollama_tool_calls):
        messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
    return messages


example_msgs = [msg for ex in examples for msg in tool_example_to_messages(ex)]


query_analyzer_with_examples = (
    {"question": RunnablePassthrough()}
    | prompt.partial(examples=example_msgs)
    | structured_llm
)

query_analyzer_with_examples.invoke(
    "what's the difference between web voyager and reflection agents? do both use langgraph?"
)

"""
Thanks to our examples we get a slightly more decomposed search query. With some more prompt engineering and tuning of our examples we could improve query generation even more.

You can see that the examples are passed to the model as messages in the [LangSmith trace](https://smith.langchain.com/public/aeaaafce-d2b1-4943-9a61-bc954e8fc6f2/r).
"""
logger.info("Thanks to our examples we get a slightly more decomposed search query. With some more prompt engineering and tuning of our examples we could improve query generation even more.")

logger.info("\n\n[DONE]", bright=True)