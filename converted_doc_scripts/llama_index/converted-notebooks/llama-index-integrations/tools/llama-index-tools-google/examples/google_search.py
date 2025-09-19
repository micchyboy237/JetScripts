async def main():
    from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.workflow import Context
    from llama_index.tools.google_search.base import GoogleSearchToolSpec
    from llama_index.tools.tool_spec.load_and_search.base import LoadAndSearchToolSpec
    import os
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    # os.environ["OPENAI_API_KEY"] = "sk-your-key"

    google_spec = GoogleSearchToolSpec(key="your-key", engine="your-engine")

    tools = LoadAndSearchToolSpec.from_defaults(
        google_spec.to_tool_list()[0],
    ).to_tool_list()

    agent = FunctionAgent(
        tools=tools,
        llm=OllamaFunctionCalling(model="llama3.2"),
    )

    ctx = Context(agent)

    await agent.run("who is barack obama", ctx=ctx)

    await agent.run("when is the last time barrack obama visited michigan", ctx=ctx)

    await agent.run("when else did he visit michigan", ctx=ctx)

    await agent.run("what is his favourite sport", ctx=ctx)

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
