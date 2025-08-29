async def main():
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_hub.tools.ionic_shopping.base import IonicShoppingToolSpec
    from llama_index.core.agent.workflow import FunctionAgent
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    # Ionic Shopping Tool
    
    [Ionic](https://www.ioniccommerce.com/) is a plug and play ecommerce marketplace for AI Assistants. By using Ionic, you are effortlessly providing your users with the ability to shop and transact directly within your agent, and you'll get a cut of the transaction.
    
    This is a basic jupyter notebook demonstrating how to integrate the Ionic Shopping Tool. For more information on setting up your Agent with Ionic, see the Ionic [documentation](https://docs.ioniccommerce.com/introduction).
    
    This Jupyter Notebook demonstrates how to use the Ionic tool with an Agent.
    
    ---
    
    ## Setup the Tool
    ### First, let's install our dependencies
    """
    logger.info("# Ionic Shopping Tool")
    
    # !pip install llama-index llama-hub ionic-api-sdk
    
    """
    ### Configure OllamaFunctionCallingAdapter
    """
    logger.info("### Configure OllamaFunctionCallingAdapter")
    
    
    # os.environ["OPENAI_API_KEY"] = "sk-api-key"
    
    
    """
    ### Import and configure the Ionic Shopping Tool
    """
    logger.info("### Import and configure the Ionic Shopping Tool")
    
    
    ionic_tool = IonicShoppingToolSpec().to_tool_list()
    
    for tool in ionic_tool:
        logger.debug(tool.metadata.name)
    
    """
    ### Use Ionic
    """
    logger.info("### Use Ionic")
    
    agent = FunctionAgent(
        tools=ionic_tool,
        llm=OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096),
    )
    
    logger.debug(
        await agent.run(
            "I'm looking for a 5k monitor can you find me 3 options between $600 and $1000"
        )
    )
    
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