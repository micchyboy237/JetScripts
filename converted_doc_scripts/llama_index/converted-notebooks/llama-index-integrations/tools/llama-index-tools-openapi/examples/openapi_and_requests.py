async def main():
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.tools.openapi.base import OpenAPIToolSpec
    from llama_index.tools.requests.base import RequestsToolSpec
    from llama_index.tools.tool_spec.load_and_search.base import LoadAndSearchToolSpec
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
    
    
    # os.environ["OPENAI_API_KEY"] = "sk-your-api-key"
    
    
    
    f = requests.get(
        "https://raw.githubusercontent.com/APIs-guru/openapi-directory/main/APIs/openai.com/1.2.0/openapi.yaml"
    ).text
    open_api_spec = yaml.safe_load(f)
    
    
    open_spec = OpenAPIToolSpec(open_api_spec)
    open_spec = OpenAPIToolSpec(
        url="https://raw.githubusercontent.com/APIs-guru/openapi-directory/main/APIs/openai.com/1.2.0/openapi.yaml"
    )
    
    requests_spec = RequestsToolSpec(
        {
            "api.openai.com": {
                "Authorization": "Bearer sk-your-key",
                "Content-Type": "application/json",
            }
        }
    )
    
    wrapped_tools = LoadAndSearchToolSpec.from_defaults(
        open_spec.to_tool_list()[0],
    ).to_tool_list()
    
    agent = FunctionAgent(
        tools=[*wrapped_tools, *requests_spec.to_tool_list()],
        llm=OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096),
    )
    
    logger.debug(
        await agent.run("what is the base url for the server")
    )
    
    logger.debug(
        await agent.run("what is the completions api")
    )
    
    logger.debug(
        await agent.run("ask the completions api for a joke")
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