import asyncio
import os
import shutil
from jet.transformers.formatters import format_json
from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
from jet.logger import CustomLogger
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field


async def main():
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    """
    When you run your agent or multi-agent framework, you might want it to output the result in a specific format.
    In this notebook, we will see a simple example of how to apply this to a FunctionAgent!🦙🚀
    Let's first install the needed dependencies
    """
    logger.info("\n\nLet's now define our structured output format")

    class MathResult(BaseModel):
        operation: str = Field(
            description="The operation that has been performed")
        result: int = Field(description="Result of the operation")

    logger.info("And a very simple calculator agent")
    llm = OllamaFunctionCalling(model="llama3.2")

    # Define tools as FunctionTool instances
    def add(x: int, y: int) -> int:
        """Add two numbers"""
        return int(x) + int(y)

    def multiply(x: int, y: int) -> int:
        """Multiply two numbers"""
        return int(x) * int(y)

    # Convert functions to FunctionTool instances
    add_tool = FunctionTool.from_defaults(fn=add)
    multiply_tool = FunctionTool.from_defaults(fn=multiply)

    agent = FunctionAgent(
        llm=llm,
        output_cls=MathResult,
        tools=[add_tool, multiply_tool],
        system_prompt="You are a calculator agent that can add or multiply two numbers by calling tools",
        name="calculator",
    )

    logger.info("Let's now run the agent")
    try:
        response = await agent.run("What is the result of 10 multiplied by 4?")
        if response and hasattr(response, 'structured_response'):
            logger.success(format_json(response.structured_response))
            logger.info("Finally, we can get the structured output")
            logger.debug(response.structured_response)
            logger.debug(response.get_pydantic_model(MathResult))
        else:
            logger.error("No valid response received from agent")
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")
        raise

    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    asyncio.run(main())
