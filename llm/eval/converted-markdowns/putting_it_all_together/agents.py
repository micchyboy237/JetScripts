from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import ReActAgent
from jet.llm.ollama.base import Ollama
from llama_index.core.tools import FunctionTool
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()


def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)

llm = Ollama(model="llama3.2", request_timeout=300.0, context_window=4096)

agent = ReActAgent.from_tools([multiply_tool], llm=llm, verbose=True)


query_engine_tools = [
    QueryEngineTool(
        query_engine=sql_agent,
        metadata=ToolMetadata(
            name="sql_agent", description="Agent that can execute SQL queries."
        ),
    ),
]

agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True)

logger.info("\n\n[DONE]", bright=True)
