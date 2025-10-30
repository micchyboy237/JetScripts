from typing import List, Dict, Any, Union, Type, Callable
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool, BaseTool
from langchain_core.output_parsers import PydanticToolsParser
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class MathOperation(BaseModel):
    """Base class for math operations."""
    a: int = Field(description="First integer")
    b: int = Field(description="Second integer")

class Add(MathOperation):
    """Add two integers."""

class Multiply(MathOperation):
    """Multiply two integers."""

@tool
def add_tool(operation: Add) -> int:
    """Add two integers."""
    return operation.a + operation.b

@tool
def multiply_tool(operation: Multiply) -> int:
    """Multiply two integers."""
    return operation.a * operation.b

def bind_tools_to_llm(
    llm: BaseLanguageModel,
    tools: List[Union[Type[BaseModel], Callable, BaseTool]],
) -> BaseLanguageModel:
    """Bind tools to an LLM for tool-calling capabilities."""
    return llm.bind_tools(tools)

def parse_and_execute_tool_calls(
    response: AIMessage,
    tools_map: Dict[str, Callable],
    parser: PydanticToolsParser | None = None,
) -> List[Any]:
    """Parse tool calls from LLM response and execute them."""
    results = []
    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            args = tool_call["args"]
            if tool_name in tools_map:
                result = tools_map[tool_name](**args)
                results.append(result)
    return results

def create_agent_with_tools(
    llm: BaseLanguageModel,
    tools: List[BaseTool],
    system_prompt: str = "You are a helpful assistant.",
) -> Callable:
    """Create a LangGraph-based agent for handling tool loops."""
    return create_agent(llm, tools, system_prompt=system_prompt)

# Example usage
llm = ChatOpenAI(model="qwen3-instruct-2507:4b", base_url="http://shawn-pc.local:8080/v1")  # Compatible with M1; adjust for your provider
tools = [Add, Multiply]  # Pydantic models as tools
llm_with_tools = bind_tools_to_llm(llm, tools)

# Simple invocation and execution
query = "What is 5 plus 3?"
response = llm_with_tools.invoke(query)
tools_map = {"add": lambda a, b: a + b, "multiply": lambda a, b: a * b}  # Map to executables
results = parse_and_execute_tool_calls(response, tools_map)
print(results)  # e.g., [8] if add is called

# Agent workflow
agent_tools = [add_tool, multiply_tool]  # Function-based tools for agent
agent = create_agent_with_tools(llm, agent_tools)
agent_response = agent.invoke({"messages": [{"role": "user", "content": "Multiply 4 by 7 then add 2."}]})
print(agent_response["messages"][-1].content)  # Final answer after loops