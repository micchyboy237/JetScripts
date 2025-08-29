async def main():
    from jet.transformers.formatters import format_json
    from box_sdk_gen import DeveloperTokenConfig, BoxDeveloperTokenAuth, BoxClient
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.tools.box import BoxSearchToolSpec
    import os
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    BOX_DEV_TOKEN = "your_box_dev_token"

    config = DeveloperTokenConfig(BOX_DEV_TOKEN)
    auth = BoxDeveloperTokenAuth(config)
    box_client = BoxClient(auth)

    # os.environ["OPENAI_API_KEY"] = "your-key"

    box_tool_spec = BoxSearchToolSpec(box_client)

    agent = FunctionAgent(
        tools=box_tool_spec.to_tool_list(),
        llm=OllamaFunctionCallingAdapter(model="llama3.2"),
    )

    answer = await agent.run("search all invoices")
    logger.success(format_json(answer))
    logger.debug(answer)

    """
    ```
    I found the following invoices:
    
    1. **Invoice-A5555.txt**
       - Size: 150 bytes
       - Created By: RB Admin
       - Created At: 2024-04-30 06:22:18
    
    2. **Invoice-Q2468.txt**
       - Size: 176 bytes
       - Created By: RB Admin
       - Created At: 2024-04-30 06:22:19
    
    3. **Invoice-B1234.txt**
       - Size: 168 bytes
       - Created By: RB Admin
       - Created At: 2024-04-30 06:22:15
    
    4. **Invoice-Q8888.txt**
       - Size: 164 bytes
       - Created By: RB Admin
       - Created At: 2024-04-30 06:22:14
    
    5. **Invoice-C9876.txt**
       - Size: 189 bytes
       - Created By: RB Admin
       - Created At: 2024-04-30 06:22:17
    
    These are the invoices found in the search. Let me know if you need more information or assistance with these invoices.
    ```
    """
    logger.info("I found the following invoices:")

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
