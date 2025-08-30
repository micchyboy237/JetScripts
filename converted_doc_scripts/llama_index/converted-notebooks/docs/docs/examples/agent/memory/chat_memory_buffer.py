async def main():
    from jet.transformers.formatters import format_json
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import ReActAgent, FunctionAgent
    from llama_index.core.llms import ChatMessage
    from llama_index.core.memory import ChatMemoryBuffer
    from llama_index.core.workflow import Context
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    LOG_DIR = f"{OUTPUT_DIR}/logs"
    
    log_file = os.path.join(LOG_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.orange(f"Logs: {log_file}")
    
    """
    # Chat Memory Buffer
    
    **NOTE:** This example of memory is deprecated in favor of the newer and more flexible `Memory` class. See the [latest docs](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/memory/).
    
    The `ChatMemoryBuffer` is a memory buffer that simply stores the last X messages that fit into a token limit.
    
    %pip install llama-index-core
    
    ## Setup
    """
    logger.info("# Chat Memory Buffer")
    
    
    memory = ChatMemoryBuffer.from_defaults(token_limit=40000)
    
    """
    ## Using Standalone
    """
    logger.info("## Using Standalone")
    
    
    chat_history = [
        ChatMessage(role="user", content="Hello, how are you?"),
        ChatMessage(role="assistant", content="I'm doing well, thank you!"),
    ]
    
    memory.put_messages(chat_history)
    
    history = memory.get()
    
    all_history = memory.get_all()
    
    memory.reset()
    
    """
    ## Using with Agents
    
    You can set the memory in any agent in the `.run()` method.
    """
    logger.info("## Using with Agents")
    
    
    # os.environ["OPENAI_API_KEY"] = "sk-proj-..."
    
    
    
    memory = ChatMemoryBuffer.from_defaults(token_limit=40000)
    
    agent = FunctionAgent(tools=[], llm=OllamaFunctionCallingAdapter(model="llama3.2"))
    
    ctx = Context(agent)
    
    resp = await agent.run("Hello, how are you?", ctx=ctx, memory=memory)
    logger.success(format_json(resp))
    
    logger.debug(memory.get_all())
    
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