async def main():
    from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.tools.zapier.base import ZapierToolSpec
    import os
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    # os.environ["OPENAI_API_KEY"] = "sk-your-key"

    zapier_spec = ZapierToolSpec(api_key="sk-ak-your-key")
    tools = zapier_spec.to_tool_list()

    agent = FunctionAgent(
        tools=zapier_spec.to_tool_list(),
        llm=OllamaFunctionCalling(model="llama3.2"),
    )

    logger.debug(await agent.run("what actions are available"))
    logger.debug(await agent.run("Can you find the taco night file in google drive"))

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
