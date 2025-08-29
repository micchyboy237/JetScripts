async def main():
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.tools.chatgpt_plugin.base import ChatGPTPluginToolSpec
    from llama_index.tools.requests.base import RequestsToolSpec
    import os
    import requests
    import shutil
    import yaml

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    # os.environ["OPENAI_API_KEY"] = "sk-your-key"

    f = requests.get(
        "https://raw.githubusercontent.com/sisbell/chatgpt-plugin-store/main/manifests/today-currency-converter.oiconma.repl.co.json"
    ).text
    manifest = yaml.safe_load(f)

    requests_spec = RequestsToolSpec()
    plugin_spec = ChatGPTPluginToolSpec(manifest)
    plugin_spec = ChatGPTPluginToolSpec(
        manifest_url="https://raw.githubusercontent.com/sisbell/chatgpt-plugin-store/main/manifests/today-currency-converter.oiconma.repl.co.json"
    )

    agent = FunctionAgent(
        tools=[*plugin_spec.to_tool_list(), *requests_spec.to_tool_list()],
        llm=Openai(model="llama3.2")
    )

    logger.debug(await agent.run("Can you give me info on the OpenAPI plugin that was loaded"))

    logger.debug(await agent.run("Can you convert 100 euros to CAD"))

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
