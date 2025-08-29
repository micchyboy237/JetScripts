async def main():
    from jet.transformers.formatters import format_json
    from jet.logger import CustomLogger
    from llama_index.packs.cohere_citation_chat import CohereCitationChatEnginePack
    from llama_index.readers.web import SimpleWebPageReader
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    <a href="https://github.com/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-cohere-citation-chat/examples/cohere_citation_chat_example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    ## Install and Import Dependencies
    """
    logger.info("## Install and Import Dependencies")
    
    # %pip install llama-index
    # %pip install llama-index-llms-cohere
    # %pip install llama-index-embeddings-cohere
    # %pip install cohere
    # %pip install llama-index-readers-web
    # %pip install llama-index-packs-cohere-citation-chat
    
    
    
    """
    Configure your Cohere API key.
    """
    logger.info("Configure your Cohere API key.")
    
    os.environ["COHERE_API_KEY"] = "your-api-key-here"
    
    """
    Parse your documents and pass to your LlamaPack. In this example, use nodes from a Paul Graham essay as input. Run the LlamaPack to create a chat engine.
    """
    logger.info("Parse your documents and pass to your LlamaPack. In this example, use nodes from a Paul Graham essay as input. Run the LlamaPack to create a chat engine.")
    
    documents = SimpleWebPageReader().load_data(
        [
            "https://raw.githubusercontent.com/jerryjliu/llama_index/adb054429f642cc7bbfcb66d4c232e072325eeab/examples/paul_graham_essay/data/paul_graham_essay.txt"
        ]
    )
    cohere_citation_chat_pack = CohereCitationChatEnginePack(documents=documents)
    chat_engine = cohere_citation_chat_pack.run()
    
    """
    Run a set of queries via the chat engine methods
    """
    logger.info("Run a set of queries via the chat engine methods")
    
    queries = [
        "What did Paul Graham do growing up?",
        "When and how did Paul Graham's mother die?",
        "What, in Paul Graham's opinion, is the most distinctive thing about YC?",
        "When and how did Paul Graham meet Jessica Livingston?",
        "What is Bel, and when and where was it written?",
    ]
    for query in queries:
        logger.debug("Query ")
        logger.debug("=====")
        logger.debug(query)
        logger.debug("Chat")
        response = chat_engine.chat(query)
        logger.debug("Chat Response")
        logger.debug("========")
        logger.debug(response)
        logger.debug(f"Citations: {response.citations}")
        logger.debug(f"Documents: {response.documents}")
        logger.debug("Async Chat")
        response = chat_engine.chat(query)
        logger.success(format_json(response))
        logger.debug("Async Chat Response")
        logger.debug("========")
        logger.debug(response)
        logger.debug(f"Citations: {response.citations}")
        logger.debug(f"Documents: {response.documents}")
        logger.debug("Stream Chat")
        response = chat_engine.stream_chat(query)
        logger.debug("Stream Chat Response")
        logger.debug("========")
        response.print_response_stream()
        logger.debug(f"Citations: {response.citations}")
        logger.debug(f"Documents: {response.documents}")
        logger.debug("Async Stream Chat")
        response = chat_engine.stream_chat(query)
        logger.success(format_json(response))
        logger.debug("Async Stream Chat Response")
        logger.debug("========")
        await response.aprint_response_stream()
        logger.debug(f"Citations: {response.citations}")
        logger.debug(f"Documents: {response.documents}")
        logger.debug()
    
    """
    You can access the internals of the LlamaPack, including your Cohere LLM and your vector store index, via the `get_modules` method.
    """
    logger.info("You can access the internals of the LlamaPack, including your Cohere LLM and your vector store index, via the `get_modules` method.")
    
    modules = cohere_citation_chat_pack.get_modules()
    display(modules)
    
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