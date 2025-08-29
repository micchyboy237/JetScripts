async def main():
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.workflow import Context
    from llama_index.tools.tool_spec.load_and_search.base import LoadAndSearchToolSpec
    from llama_index.tools.yelp.base import YelpToolSpec
    import os
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    # os.environ["OPENAI_API_KEY"] = "sk-your-key"

    tool_spec = YelpToolSpec(api_key="your-key", client_id="your-id")

    tools = tool_spec.to_tool_list()
    agent = FunctionAgent(
        tools=[
            *LoadAndSearchToolSpec.from_defaults(tools[0]).to_tool_list(),
            *LoadAndSearchToolSpec.from_defaults(tools[1]).to_tool_list(),
        ],
        llm=OllamaFunctionCallingAdapter(model="llama3.2"),
    )

    ctx = Context(agent)

    logger.debug(await agent.run("what good resturants are in toronto", ctx=ctx))
    logger.debug(await agent.run("what are the details of lao lao bar", ctx=ctx))

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
