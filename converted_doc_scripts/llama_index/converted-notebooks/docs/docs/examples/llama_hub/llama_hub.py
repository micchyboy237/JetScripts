async def main():
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core import VectorStoreIndex
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.readers.web import SimpleWebPageReader
    from llama_index.tools.google import GmailToolSpec
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    # LlamaHub Demostration
    
    Here we give a simple overview of how to use data loaders and tools (for agents) within [LlamaHub](llamahub.ai).
    
    **NOTES**: 
    
    - You can learn how to use everything in LlamaHub by clicking into each module and looking at the code snippet.
    - Also, you can find a [full list of agent tools here](https://llamahub.ai/?tab=tools).
    - In this guide we'll show how to use `download_loader` and `download_tool`. You can also install `llama-hub` [as a package](https://github.com/run-llama/llama-hub#usage-use-llama-hub-as-pypi-package).
    
    ## Using a Data Loader
    
    In this example we show how to use `SimpleWebPageReader`.
    
    **NOTE**: for any module on LlamaHub, to use with `download_` functions, note down the class name.
    """
    logger.info("# LlamaHub Demostration")
    
    # %pip install llama-index-agent-openai
    # %pip install llama-index-readers-web
    # %pip install llama-index-tools-google
    
    
    reader = SimpleWebPageReader(html_to_text=True)
    
    docs = reader.load_data(urls=["https://eugeneyan.com/writing/llm-patterns/"])
    
    logger.debug(docs[0].get_content()[:400])
    
    """
    Now you can plug these docs into your downstream LlamaIndex pipeline.
    """
    logger.info("Now you can plug these docs into your downstream LlamaIndex pipeline.")
    
    
    index = VectorStoreIndex.from_documents(docs)
    query_engine = index.as_query_engine()
    
    response = query_engine.query("What are ways to evaluate LLMs?")
    logger.debug(str(response))
    
    """
    ## Using an Agent Tool Spec
    
    In this example we show how to load an agent tool.
    """
    logger.info("## Using an Agent Tool Spec")
    
    
    tool_spec = GmailToolSpec()
    
    
    agent = FunctionAgent(
        tools=tool_spec.to_tool_list(),
        llm=OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096),
    )
    
    await agent.run("What is my most recent email")
    
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