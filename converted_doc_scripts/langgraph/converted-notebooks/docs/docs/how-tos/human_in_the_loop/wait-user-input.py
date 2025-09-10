from IPython.display import Image, display
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph import MessagesState, START
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from pydantic import BaseModel
from typing_extensions import TypedDict
import os
import shutil

async def main():
    
    
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
    # How to wait for user input using `interrupt`
    
    !!! tip "Prerequisites"
    
        This guide assumes familiarity with the following concepts:
    
        * [Human-in-the-loop](../../../concepts/human_in_the_loop)
        * [LangGraph Glossary](../../../concepts/low_level)
        
    
    **Human-in-the-loop (HIL)** interactions are crucial for [agentic systems](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#human-in-the-loop). Waiting for human input is a common HIL interaction pattern, allowing the agent to ask the user clarifying questions and await input before proceeding. 
    
    We can implement this in LangGraph using the [`interrupt()`][langgraph.types.interrupt] function. `interrupt` allows us to stop graph execution to collect input from a user and continue execution with collected input.
    
    ## Setup
    
    First we need to install the packages required
    """
    logger.info("# How to wait for user input using `interrupt`")
    
    # %%capture --no-stderr
    # %pip install --quiet -U langgraph jet.adapters.langchain.chat_ollama
    
    """
    Next, we need to set API keys for Ollama and / or Ollama(the LLM(s) we will use)
    """
    logger.info("Next, we need to set API keys for Ollama and / or Ollama(the LLM(s) we will use)")
    
    # import getpass
    
    
    def _set_env(var: str):
        if not os.environ.get(var):
    #         os.environ[var] = getpass.getpass(f"{var}: ")
    
    
    # _set_env("ANTHROPIC_API_KEY")
    
    """
    <div class="admonition tip">
        <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
        <p style="padding-top: 5px;">
            Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
        </p>
    </div>
    
    ## Simple Usage
    
    Let's explore a basic example of using human feedback. A straightforward approach is to create a node, **`human_feedback`**, designed specifically to collect user input. This allows us to gather feedback at a specific, chosen point in our graph.
    
    Steps:
    
    1. **Call `interrupt()`** inside the **`human_feedback`** node.  
    2. **Set up a [checkpointer](https://langchain-ai.github.io/langgraph/concepts/low_level/#checkpointer)** to save the graph's state up to this node.  
    3. **Use `Command(resume=...)`** to provide the requested value to the **`human_feedback`** node and resume execution.
    """
    logger.info("## Simple Usage")
    
    
    
    
    class State(TypedDict):
        input: str
        user_feedback: str
    
    
    def step_1(state):
        logger.debug("---Step 1---")
        pass
    
    
    def human_feedback(state):
        logger.debug("---human_feedback---")
        feedback = interrupt("Please provide feedback:")
        return {"user_feedback": feedback}
    
    
    def step_3(state):
        logger.debug("---Step 3---")
        pass
    
    
    builder = StateGraph(State)
    builder.add_node("step_1", step_1)
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("step_3", step_3)
    builder.add_edge(START, "step_1")
    builder.add_edge("step_1", "human_feedback")
    builder.add_edge("human_feedback", "step_3")
    builder.add_edge("step_3", END)
    
    memory = InMemorySaver()
    
    graph = builder.compile(checkpointer=memory)
    
    display(Image(graph.get_graph().draw_mermaid_png()))
    
    """
    Run until our `interrupt()` at `human_feedback`:
    """
    logger.info("Run until our `interrupt()` at `human_feedback`:")
    
    initial_input = {"input": "hello world"}
    
    thread = {"configurable": {"thread_id": "1"}}
    
    for event in graph.stream(initial_input, thread, stream_mode="updates"):
        logger.debug(event)
        logger.debug("\n")
    
    """
    Now, we can manually update our graph state with the user input:
    """
    logger.info("Now, we can manually update our graph state with the user input:")
    
    for event in graph.stream(
        Command(resume="go to step 3!"),
        thread,
        stream_mode="updates",
    ):
        logger.debug(event)
        logger.debug("\n")
    
    """
    We can see our feedback was added to state -
    """
    logger.info("We can see our feedback was added to state -")
    
    graph.get_state(thread).values
    
    """
    ## Agent
    
    In the context of [agents](../../../concepts/agentic_concepts), waiting for user feedback is especially useful for asking clarifying questions. To illustrate this, we’ll create a simple [ReAct-style agent](../../../concepts/agentic_concepts#react-implementation) capable of [tool calling](https://python.langchain.com/docs/concepts/tool_calling/). 
    
    For this example, we’ll use Ollama's chat model along with a **mock tool** (purely for demonstration purposes).
    
    <div class="admonition note">
        <p class="admonition-title">Using Pydantic with LangChain</p>
        <p>
            This notebook uses Pydantic v2 <code>BaseModel</code>, which requires <code>langchain-core >= 0.3</code>. Using <code>langchain-core < 0.3</code> will result in errors due to mixing of Pydantic v1 and v2 <code>BaseModels</code>.
        </p>
    </div>
    """
    logger.info("## Agent")
    
    
    
    
    @tool
    def search(query: str):
        """Call to surf the web."""
        return f"I looked up: {query}. Result: It's sunny in San Francisco, but you better look out if you're a Gemini 😈."
    
    
    tools = [search]
    tool_node = ToolNode(tools)
    
    
    model = ChatOllama(model="llama3.2")
    
    
    
    class AskHuman(BaseModel):
        """Ask the human a question"""
    
        question: str
    
    
    model = model.bind_tools(tools + [AskHuman])
    
    
    
    def should_continue(state):
        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls:
            return END
        elif last_message.tool_calls[0]["name"] == "AskHuman":
            return "ask_human"
        else:
            return "action"
    
    
    def call_model(state):
        messages = state["messages"]
        response = model.invoke(messages)
        return {"messages": [response]}
    
    
    def ask_human(state):
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        ask = AskHuman.model_validate(state["messages"][-1].tool_calls[0]["args"])
        location = interrupt(ask.question)
        tool_message = [{"tool_call_id": tool_call_id, "type": "tool", "content": location}]
        return {"messages": tool_message}
    
    
    
    
    workflow = StateGraph(MessagesState)
    
    workflow.add_node("agent", call_model)
    workflow.add_node("action", tool_node)
    workflow.add_node("ask_human", ask_human)
    
    workflow.add_edge(START, "agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        path_map=["ask_human", "action", END],
    )
    
    workflow.add_edge("action", "agent")
    
    workflow.add_edge("ask_human", "agent")
    
    
    memory = InMemorySaver()
    
    app = workflow.compile(checkpointer=memory)
    
    display(Image(app.get_graph().draw_mermaid_png()))
    
    """
    ## Interacting with the Agent
    
    We can now interact with the agent. Let's ask it to ask the user where they are, then tell them the weather. 
    
    This should make it use the `ask_human` tool first, then use the normal tool.
    """
    logger.info("## Interacting with the Agent")
    
    config = {"configurable": {"thread_id": "2"}}
    for event in app.stream(
        {
            "messages": [
                (
                    "user",
                    "Ask the user where they are, then look up the weather there",
                )
            ]
        },
        config,
        stream_mode="values",
    ):
        if "messages" in event:
            event["messages"][-1].pretty_logger.debug()
    
    app.get_state(config).next
    
    """
    You can see that our graph got interrupted inside the `ask_human` node, which is now waiting for a `location` to be provided. We can provide this value by invoking the graph with a `Command(resume="<location>")` input:
    """
    logger.info("You can see that our graph got interrupted inside the `ask_human` node, which is now waiting for a `location` to be provided. We can provide this value by invoking the graph with a `Command(resume="<location>")` input:")
    
    for event in app.stream(
        Command(resume="san francisco"),
        config,
        stream_mode="values",
    ):
        if "messages" in event:
            event["messages"][-1].pretty_logger.debug()
    
    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())