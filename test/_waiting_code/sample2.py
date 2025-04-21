from jet.logger import CustomLogger
from jet.llm.ollama.base import initialize_ollama_settings
import os
import uuid
from langchain_core.chat_history import InMemoryChatMessageHistory
from jet.llm.ollama.base_langchain import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, MessagesState, StateGraph

chats_by_session_id = {}


def setup_logger(script_path: str) -> CustomLogger:
    script_dir = os.path.dirname(os.path.abspath(script_path))
    log_file = os.path.join(
        script_dir, "generated", f"{os.path.splitext(os.path.basename(script_path))[0]}.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    return logger


def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    chat_history = chats_by_session_id.get(session_id)
    if chat_history is None:
        chat_history = InMemoryChatMessageHistory()
        chats_by_session_id[session_id] = chat_history
    return chat_history


def create_graph() -> StateGraph:
    builder = StateGraph(state_schema=MessagesState)
    model = ChatOllama(model="llama3.1")

    def call_model(state: MessagesState, config: RunnableConfig) -> dict:
        if "configurable" not in config or "session_id" not in config["configurable"]:
            raise ValueError(
                "Config must include {'configurable': {'session_id': 'some_value'}}")
        chat_history = get_chat_history(config["configurable"]["session_id"])
        messages = list(chat_history.messages) + state["messages"]
        ai_message = model.invoke(messages)
        chat_history.add_messages(state["messages"] + [ai_message])
        return {"messages": ai_message}

    builder.add_edge(START, "model")
    builder.add_node("model", call_model)
    return builder.compile()


def run_conversation(graph: StateGraph, session_id: uuid.UUID, messages: list[HumanMessage], logger: CustomLogger) -> None:
    config = {"configurable": {"session_id": session_id}}
    for input_message in messages:
        for event in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
            logger.debug(event["messages"][-1])
        for msg, _ in graph.stream({"messages": [input_message]}, config, stream_mode="messages"):
            if msg.content and not isinstance(msg, HumanMessage):
                logger.debug(msg.content, end="|", flush=True)


def main():
    logger = setup_logger(__file__)
    initialize_ollama_settings()
    graph = create_graph()
    session_id = uuid.uuid4()

    messages = [
        SystemMessage(
            content="You are a helpful assistant. Use the tools available, including memory tools, to assist the user."),
        HumanMessage(content="hi! I'm bob"),
        HumanMessage(content="what was my name?")
    ]

    run_conversation(graph, session_id, messages, logger)
    logger.info("\n\n[DONE]", bright=True)


if __name__ == "__main__":
    main()
