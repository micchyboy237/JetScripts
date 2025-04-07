import asyncio
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool


def calculator(a: float, b: float, operator: str) -> str:
    try:
        if operator == "+":
            return str(a + b)
        elif operator == "-":
            return str(a - b)
        elif operator == "*":
            return str(a * b)
        elif operator == "/":
            if b == 0:
                return "Error: Division by zero"
            return str(a / b)
        else:
            return "Error: Invalid operator. Please use +, -, *, or /"
    except Exception as e:
        return f"Error: {str(e)}"


# Create calculator tool
calculator_tool = FunctionTool(
    name="calculator",
    description="A simple calculator that performs basic arithmetic operations",
    func=calculator,
    global_imports=[],
)


async def main() -> None:
    # Set up the model client for Ollama
    model_client = OllamaChatCompletionClient(model="llama3.2")

    # Initialize the assistant agent with the calculator tool
    assistant = AssistantAgent(
        "Assistant",
        model_client=model_client,
        tools=[calculator_tool],
    )

    # Define termination condition (based on a specific trigger like "TERMINATE")
    termination = TextMentionTermination("TERMINATE")

    # Set up a team (here only one assistant agent is used)
    team = RoundRobinGroupChat([assistant], termination_condition=termination)

    # The task is for the assistant to handle multiple user requests
    # Let's simulate a simple user interaction for calculations
    task = "Can you calculate 5 + 3 for me?"

    # The assistant will start processing the task
    await Console(team.run_stream(task=task))


if __name__ == "__main__":
    asyncio.run(main())
