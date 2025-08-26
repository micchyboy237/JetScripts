from jet.actions.generation import call_ollama_chat
from jet.logger import logger
from jet.llm.llm_types import Tool, MessageRole

if __name__ == "__main__":
    # Define a sample tool for a calculator
    calculator_tool = Tool(
        type="function",
        function={
            "name": "calculate",
            "description": "Performs basic arithmetic calculations.",
            "parameters": {
                "type": "object",
                "required": ["expression"],
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A mathematical expression to evaluate (e.g., '2 + 2')."
                    }
                }
            }
        }
    )

    # Updated prompt to leverage the calculator tool
    prompt = "I found a treasure map with a code: 'The treasure is worth 50 * 3 gold coins.' How much is the treasure worth? Use the calculator tool to compute the value."

    response = ""
    for chunk in call_ollama_chat(
        prompt,
        model="llama3.2",
        tools=[calculator_tool],
        stream=True,
        verbose=True
    ):
        response += chunk
        logger.success(chunk, flush=True)

    # Log the final response
    logger.newline()
    logger.info("Final response:", response)
