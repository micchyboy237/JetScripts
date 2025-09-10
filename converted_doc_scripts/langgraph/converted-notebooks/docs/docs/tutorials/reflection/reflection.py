from jet.transformers.formatters import format_json
from jet.logger import logger
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_fireworks import ChatFireworks
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from typing import Annotated, List, Sequence
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
    # Reflection
    
    
    In the context of LLM agent building, reflection refers to the process of prompting an LLM to observe its past steps (along with potential observations from tools/the environment) to assess the quality of the chosen actions.
    This is then used downstream for things like re-planning, search, or evaluation.
    
    ![Reflection](attachment:fc393f72-3401-4b86-b0d3-e4789b640a27.png)
    
    This notebook demonstrates a very simple form of reflection in LangGraph.
    
    ## Setup
    
    First, let's install our required packages and set our API keys
    """
    logger.info("# Reflection")
    
    # %pip install -U --quiet  langgraph langchain-fireworks
    # %pip install -U --quiet tavily-python
    
    # import getpass
    
    
    def _set_if_undefined(var: str) -> None:
        if os.environ.get(var):
            return
    #     os.environ[var] = getpass.getpass(var)
    
    
    _set_if_undefined("TAVILY_API_KEY")
    _set_if_undefined("FIREWORKS_API_KEY")
    
    """
    <div class="admonition tip">
        <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
        <p style="padding-top: 5px;">
            Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
        </p>
    </div>
    
    ## Generate
    
    For our example, we will create a "5 paragraph essay" generator. First, create the generator:
    """
    logger.info("## Generate")
    
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an essay assistant tasked with writing excellent 5-paragraph essays."
                " Generate the best essay possible for the user's request."
                " If the user provides critique, respond with a revised version of your previous attempts.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    llm = ChatFireworks(
        model="accounts/fireworks/models/mixtral-8x7b-instruct", max_tokens=32768
    )
    generate = prompt | llm
    
    essay = ""
    request = HumanMessage(
        content="Write an essay on why the little prince is relevant in modern childhood"
    )
    for chunk in generate.stream({"messages": [request]}):
        logger.debug(chunk.content, end="")
        essay += chunk.content
    
    """
    ### Reflect
    """
    logger.info("### Reflect")
    
    reflection_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a teacher grading an essay submission. Generate critique and recommendations for the user's submission."
                " Provide detailed recommendations, including requests for length, depth, style, etc.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    reflect = reflection_prompt | llm
    
    reflection = ""
    for chunk in reflect.stream({"messages": [request, HumanMessage(content=essay)]}):
        logger.debug(chunk.content, end="")
        reflection += chunk.content
    
    """
    ### Repeat
    
    And... that's all there is too it! You can repeat in a loop for a fixed number of steps, or use an LLM (or other check) to decide when the finished product is good enough.
    """
    logger.info("### Repeat")
    
    for chunk in generate.stream(
        {"messages": [request, AIMessage(content=essay), HumanMessage(content=reflection)]}
    ):
        logger.debug(chunk.content, end="")
    
    """
    ## Define graph
    
    Now that we've shown each step in isolation, we can wire it up in a graph.
    """
    logger.info("## Define graph")
    
    
    
    class State(TypedDict):
        messages: Annotated[list, add_messages]
    
    
    async def generation_node(state: State) -> State:
        return {"messages": [await generate.ainvoke(state["messages"])]}
    
    
    async def reflection_node(state: State) -> State:
        cls_map = {"ai": HumanMessage, "human": AIMessage}
        translated = [state["messages"][0]] + [
            cls_map[msg.type](content=msg.content) for msg in state["messages"][1:]
        ]
        res = await reflect.ainvoke(translated)
        logger.success(format_json(res))
        return {"messages": [HumanMessage(content=res.content)]}
    
    
    builder = StateGraph(State)
    builder.add_node("generate", generation_node)
    builder.add_node("reflect", reflection_node)
    builder.add_edge(START, "generate")
    
    
    def should_continue(state: State):
        if len(state["messages"]) > 6:
            return END
        return "reflect"
    
    
    builder.add_conditional_edges("generate", should_continue)
    builder.add_edge("reflect", "generate")
    memory = InMemorySaver()
    graph = builder.compile(checkpointer=memory)
    
    config = {"configurable": {"thread_id": "1"}}
    
    for event in graph.stream(
        {
            "messages": [
                HumanMessage(
                    content="Generate an essay on the topicality of The Little Prince and its message in modern life"
                )
            ],
        },
        config,
    ):
        logger.debug(event)
        logger.debug("---")
    
    state = graph.get_state(config)
    
    ChatPromptTemplate.from_messages(state.values["messages"]).pretty_logger.debug()
    
    """
    ## Conclusion
    
    Now that you've applied reflection to an LLM agent, I'll note one thing: self-reflection is inherently cyclic: it is much more effective if the reflection step has additional context or feedback (from tool observations, checks, etc.). If, like in the scenario above, the reflection step simply prompts the LLM to reflect on its output, it can still benefit the output quality (since the LLM then has multiple "shots" at getting a good output), but it's less guaranteed.
    """
    logger.info("## Conclusion")
    
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