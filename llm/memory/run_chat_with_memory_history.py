from jet.logger import CustomLogger
from jet.llm.ollama.base import initialize_ollama_settings
import os
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from jet.adapters.langchain.chat_ollama import ChatOllama
import uuid
from IPython.display import Image, display
from langchain_core.messages import HumanMessage
from langgraph.graph import START, MessagesState, StateGraph
import uuid
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from jet.adapters.langchain.chat_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from jet.adapters.langchain.chat_ollama import ChatOllama


def setup_logger():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(
        script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    initialize_ollama_settings()
    return logger


def get_trimmed_messages(messages: list[BaseMessage], max_tokens: int, model=None) -> list[BaseMessage]:
    return trim_messages(
        messages,
        token_counter=model or len,
        max_tokens=max_tokens,
        strategy="last",
        start_on="human",
        include_system=True,
        allow_partial=False,
    )


def create_thread_config(thread_id: uuid.UUID | str) -> dict:
    return {"configurable": {"thread_id": str(thread_id)}}


def run_app_stream(app, input_messages: list[BaseMessage], config: dict, logger):
    for event in app.stream({"messages": input_messages}, config, stream_mode="values"):
        logger.debug(event["messages"][-1].content)


def build_memory_graph_app(model, tools=None, system_prompt=None, prompt_fn=None, memory=None):
    from langgraph.prebuilt import create_react_agent

    if tools:
        return create_react_agent(
            model=model,
            tools=tools,
            checkpointer=memory,
            prompt=system_prompt or prompt_fn
        )
    else:
        workflow = StateGraph(state_schema=MessagesState)
        workflow.add_node("model", lambda state: {
            "messages": model.invoke(
                get_trimmed_messages(state["messages"], 5)
            )
        })
        workflow.add_edge(START, "model")
        return workflow.compile(checkpointer=memory)


if __name__ == "__main__":
    # ---------- Tools ----------
    @tool
    def get_user_age(name: str) -> str:
        """Use this tool to find the user's age."""
        if "bob" in name.lower():
            return "42 years old"
        return "41 years old"

    # ---------- USAGE EXAMPLES ----------
    logger = setup_logger()
    model = ChatOllama(model="llama3.1")
    memory = MemorySaver()

    # Example 1: With plain memory graph (no tools)
    logger.debug("=== Running basic memory graph ===")
    basic_app = build_memory_graph_app(model, memory=memory)
    thread_id = uuid.uuid4()
    config = create_thread_config(thread_id)
    run_app_stream(basic_app, [HumanMessage(
        content="hi! I'm bob")], config, logger)
    run_app_stream(basic_app, [HumanMessage(
        content="do you remember my name?")], config, logger)

    # Example 2: With tools and system prompt
    logger.debug("=== Running tool-enhanced app ===")
    system_prompt = SystemMessage(
        content="You are a helpful assistant. Use the tools available to assist the user."
    )
    tool_app = build_memory_graph_app(
        model, tools=[get_user_age], system_prompt=system_prompt, memory=memory)
    tool_thread_id = uuid.uuid4()
    tool_config = create_thread_config(tool_thread_id)

    run_app_stream(tool_app, [HumanMessage(
        content="do you remember my name?")], tool_config, logger)
    run_app_stream(tool_app, [HumanMessage(
        content="What is my age?")], tool_config, logger)

    logger.info("\n\n[DONE]", bright=True)
