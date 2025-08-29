async def main():
    from jet.transformers.formatters import format_json
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.agent.workflow import ReActAgent
    from llama_index.memory.mem0 import Mem0Memory
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/memory/Mem0Memory.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # Mem0
    
    Mem0 (pronounced “mem-zero”) enhances AI assistants and agents with an intelligent memory layer, enabling personalized AI interactions. It remembers user preferences and traits and continuously updates over time, making it ideal for applications like customer support chatbots and AI assistants.
    
    Mem0 offers two powerful ways to leverage our technology: our [managed platform](https://docs.mem0.ai/platform/overview) and our [open source solution](https://docs.mem0.ai/open-source/quickstart).
    
    If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
    """
    logger.info("# Mem0")
    
    # %pip install llama-index llama-index-memory-mem0
    
    """
    ### Setup with Mem0 Platform
    
    Set your Mem0 Platform API key as an environment variable. You can replace `<your-mem0-api-key>` with your actual API key:
    
    > Note: You can obtain your Mem0 Platform API key from the [Mem0 Platform](https://app.mem0.ai/login).
    """
    logger.info("### Setup with Mem0 Platform")
    
    
    os.environ["MEM0_API_KEY"] = "m0-..."
    
    """
    Using `from_client` (for Mem0 platform API):
    """
    logger.info("Using `from_client` (for Mem0 platform API):")
    
    
    context = {"user_id": "test_users_1"}
    memory_from_client = Mem0Memory.from_client(
        context=context,
        api_key="m0-...",
        search_msg_limit=4,  # Default is 5
    )
    
    """
    Mem0 Context is used to identify the user, agent or the conversation in the Mem0. It is required to be passed in the at least one of the fields in the `Mem0Memory` constructor.
    
    `search_msg_limit` is optional, default is 5. It is the number of messages from the chat history to be used for memory retrieval from Mem0. More number of messages will result in more context being used for retrieval but will also increase the retrieval time and might result in some unwanted results.
    
    Using `from_config` (for Mem0 OSS)
    """
    logger.info("Mem0 Context is used to identify the user, agent or the conversation in the Mem0. It is required to be passed in the at least one of the fields in the `Mem0Memory` constructor.")
    
    # os.environ["OPENAI_API_KEY"] = "<your-api-key>"
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "test_9",
                "host": "localhost",
                "port": 6333,
                "embedding_model_dims": 1536,  # Change this according to your local model's dimensions
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o",
                "temperature": 0.2,
                "max_tokens": 1500,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {"model": "mxbai-embed-large"},
        },
        "version": "v1.1",
    }
    memory_from_config = Mem0Memory.from_config(
        context=context,
        config=config,
        search_msg_limit=4,  # Default is 5
    )
    
    """
    ### Initialize LLM
    """
    logger.info("### Initialize LLM")
    
    
    llm = OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096, api_key="sk-...")
    
    """
    ## Mem0 for Function Calling Agents
    
    Use `Mem0` as memory for `FunctionAgent`s.
    
    ### Initialize Tools
    """
    logger.info("## Mem0 for Function Calling Agents")
    
    def call_fn(name: str):
        """Call the provided name.
        Args:
            name: str (Name of the person)
        """
        logger.debug(f"Calling... {name}")
    
    
    def email_fn(name: str):
        """Email the provided name.
        Args:
            name: str (Name of the person)
        """
        logger.debug(f"Emailing... {name}")
    
    
    agent = FunctionAgent(
        tools=[email_fn, call_fn],
        llm=llm,
    )
    
    response = await agent.run("Hi, My name is Mayank.", memory=memory_from_client)
    logger.success(format_json(response))
    logger.debug(str(response))
    
    response = await agent.run(
            "My preferred way of communication would be Email.",
            memory=memory_from_client,
        )
    logger.success(format_json(response))
    logger.debug(str(response))
    
    response = await agent.run(
            "Send me an update of your product.", memory=memory_from_client
        )
    logger.success(format_json(response))
    logger.debug(str(response))
    
    """
    ## Mem0 for ReAct Agents
    
    Use `Mem0` as memory for `ReActAgent`.
    """
    logger.info("## Mem0 for ReAct Agents")
    
    
    agent = ReActAgent(
        tools=[call_fn, email_fn],
        llm=llm,
    )
    
    response = await agent.run("Hi, My name is Mayank.", memory=memory_from_client)
    logger.success(format_json(response))
    logger.debug(str(response))
    
    response = await agent.run(
            "My preferred way of communication would be Email.",
            memory=memory_from_client,
        )
    logger.success(format_json(response))
    logger.debug(str(response))
    
    response = await agent.run(
            "Send me an update of your product.", memory=memory_from_client
        )
    logger.success(format_json(response))
    logger.debug(str(response))
    
    response = await agent.run(
            "First call me and then communicate me requirements.",
            memory=memory_from_client,
        )
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