async def main():
    from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.workflow import Context
    from llama_index.tools.azure_speech.base import AzureSpeechToolSpec
    from llama_index.tools.azure_translate.base import AzureTranslateToolSpec
    import os
    import shutil
    import urllib.request

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    # os.environ["OPENAI_API_KEY"] = "sk-proj-1234567890"

    speech_tool = AzureSpeechToolSpec(speech_key="your-key", region="eastus")
    translate_tool = AzureTranslateToolSpec(
        api_key="your-key", region="eastus")

    agent = FunctionAgent(
        tools=[*speech_tool.to_tool_list(), *translate_tool.to_tool_list()],
        llm=OllamaFunctionCalling(model="llama3.2"),
    )
    ctx = Context(agent)

    logger.debug(await agent.run('Say "hello world"', ctx=ctx))

    urllib.request.urlretrieve(
        "https://speechstudiorawgithubscenarioscdn.azureedge.net/call-center/sampledata/Call1_separated_16k_health_insurance.wav",
        "data/speech.wav",
    )

    logger.debug(await agent.run("transcribe and format conversation in data/speech.wav", ctx=ctx))

    logger.debug(await agent.run("translate the conversation into spanish", ctx=ctx))

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
