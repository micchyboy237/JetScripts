async def main():
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.workflow import Context
    from llama_index.tools.google_calendar.base import GoogleCalendarToolSpec
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    
    # os.environ["OPENAI_API_KEY"] = "sk-your-key"
    
    
    
    tool_spec = GoogleCalendarToolSpec()
    
    agent = FunctionAgent(
        tools=tool_spec.to_tool_list(),
        llm=OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096),
    )
    
    ctx = Context(agent)
    
    await agent.run("what is the first thing on my calendar today", ctx=ctx)
    
    await agent.run(
        "Please create an event for june 15th, 2023 at 5pm for 1 hour and invite"
        " adam@example.com to discuss tax laws",
        ctx=ctx,
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