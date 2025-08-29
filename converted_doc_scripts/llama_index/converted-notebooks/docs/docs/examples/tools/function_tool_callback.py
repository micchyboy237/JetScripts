async def main():
    from jet.transformers.formatters import format_json
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.tools import FunctionTool
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    # Function call with callback
    
    This is a feature that allows applying some human-in-the-loop concepts in FunctionTool.
    
    Basically, a callback function is added that enables the developer to request user input in the middle of an agent interaction, as well as allowing any programmatic action.
    """
    logger.info("# Function call with callback")
    
    # %pip install llama-index-llms-ollama
    # %pip install llama-index-agents-openai
    
    
    # os.environ["OPENAI_API_KEY"] = "sk-"
    
    """
    Function to display to the user the data produced for function calling and request their input to return to the interaction.
    """
    logger.info("Function to display to the user the data produced for function calling and request their input to return to the interaction.")
    
    def callback(message):
        confirmation = input(
            f"{message[1]}\nDo you approve of sending this greeting?\nInput(Y/N):"
        )
    
        if confirmation.lower() == "y":
            return "Greeting sent successfully."
        else:
            return (
                "Greeting has not been approved, talk a bit about how to improve"
            )
    
    """
    Simple function that only requires a recipient and a greeting message.
    """
    logger.info("Simple function that only requires a recipient and a greeting message.")
    
    def send_hello(destination: str, message: str) -> str:
        """
        Say hello with a rhyme
        destination: str - Name of recipient
        message: str - Greeting message with a rhyme to the recipient's name
        """
    
        return destination, message
    
    
    hello_tool = FunctionTool.from_defaults(fn=send_hello, callback=callback)
    
    llm = OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096)
    agent = FunctionAgent(tools=[hello_tool])
    
    response = await agent.run("Send hello to Karen")
    logger.success(format_json(response))
    logger.debug(str(response))
    
    response = await agent.run("Send hello to Joe")
    logger.success(format_json(response))
    logger.debug(str(response))
    
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