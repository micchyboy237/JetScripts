async def main():
    from jet.transformers.formatters import format_json
    from jet.logger import CustomLogger
    from llama_index.core.base.llms.types import ChatMessage, MessageRole
    from llama_index.llms.reka import RekaLLM
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    pip install llama-index-llms-reka
    
    """
    To obtain API key, please visit [https://platform.reka.ai/](https://platform.reka.ai/)
    
    # Chat completion
    """
    logger.info("# Chat completion")
    
    
    api_key = os.getenv("REKA_API_KEY")
    reka_llm = RekaLLM(
        model="reka-flash",
        api_key=api_key,
    )
    
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="What is the capital of France?"),
    ]
    response = reka_llm.chat(messages)
    logger.debug(response.message.content)
    
    prompt = "The capital of France is"
    response = reka_llm.complete(prompt)
    logger.debug(response.text)
    
    """
    # Streaming example
    """
    logger.info("# Streaming example")
    
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(
            role=MessageRole.USER, content="List the first 5 planets in the solar system."
        ),
    ]
    for chunk in reka_llm.stream_chat(messages):
        logger.debug(chunk.delta, end="", flush=True)
    
    prompt = "List the first 5 planets in the solar system:"
    for chunk in reka_llm.stream_complete(prompt):
        logger.debug(chunk.delta, end="", flush=True)
    
    """
    # Async use cases (chat/completion)
    """
    logger.info("# Async use cases (chat/completion)")
    
    async def main():
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            ChatMessage(
                role=MessageRole.USER,
                content="What is the largest planet in our solar system?",
            ),
        ]
        response = reka_llm.chat(messages)
        logger.success(format_json(response))
        logger.success(format_json(response))
        logger.debug(response.message.content)
    
        prompt = "The largest planet in our solar system is"
        response = reka_llm.complete(prompt)
        logger.success(format_json(response))
        logger.success(format_json(response))
        logger.debug(response.text)
    
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            ChatMessage(
                role=MessageRole.USER,
                content="Name the first 5 elements in the periodic table.",
            ),
        ]
        for chunk in reka_llm.stream_chat(messages):
            logger.debug(chunk.delta, end="", flush=True)
    
        prompt = "List the first 5 elements in the periodic table:"
        for chunk in reka_llm.stream_complete(prompt):
            logger.debug(chunk.delta, end="", flush=True)
    
    
    await main()
    
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