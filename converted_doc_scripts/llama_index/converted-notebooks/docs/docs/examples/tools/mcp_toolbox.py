async def main():
    from jet.transformers.formatters import format_json
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import AgentWorkflow
    from llama_index.core.workflow import Context
    from llama_index.llms.google_genai import GoogleGenAI
    from toolbox_llamaindex import ToolboxClient
    import asyncio
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    # Using MCP Toolbox with LlamaIndex
    
    Integrate your databases with LlamaIndex agents using MCP Toolbox.
    
    ## Overview
    
    [MCP Toolbox for Databases](https://github.com/googleapis/genai-toolbox) is an open source MCP server for databases. It was designed with enterprise-grade and production-quality in mind. It enables you to develop tools easier, faster, and more securely by handling the complexities such as connection pooling, authentication, and more.
    
    Toolbox Tools can be seamlessly integrated with LlamaIndex applications. For more information on [getting started](https://googleapis.github.io/genai-toolbox/getting-started/local_quickstart/) or [configuring](https://googleapis.github.io/genai-toolbox/getting-started/configure/) MCP Toolbox, see the [documentation](https://googleapis.github.io/genai-toolbox/getting-started/introduction/).
    
    ![architecture](https://raw.githubusercontent.com/googleapis/genai-toolbox/refs/heads/main/docs/en/getting-started/introduction/architecture.png)
    
    ## Configure and deploy
    
    Toolbox is an open source server that you deploy and manage yourself. For more
    instructions on deploying and configuring, see the official Toolbox
    documentation:
    
    * [Installing the Server](https://googleapis.github.io/genai-toolbox/getting-started/introduction/#installing-the-server)
    * [Configuring Toolbox](https://googleapis.github.io/genai-toolbox/getting-started/configure/)
    
    ### Install client SDK
    
    Install the LlamaIndex compatible MCP Toolbox SDK package before getting started:
    """
    logger.info("# Using MCP Toolbox with LlamaIndex")
    
    pip install toolbox-llamaindex
    
    """
    ### Loading Toolbox Tools
    
    Once your Toolbox server is configured and up and running, you can load tools from your server:
    """
    logger.info("### Loading Toolbox Tools")
    
    
    prompt = """
      You're a helpful hotel assistant. You handle hotel searching, booking and
      cancellations. When the user searches for a hotel, mention it's name, id,
      location and price tier. Always mention hotel ids while performing any
      searches. This is very important for any operations. For any bookings or
      cancellations, please provide the appropriate confirmation. Be sure to
      update checkin or checkout dates if mentioned by the user.
      Don't ask for confirmations from the user.
    """
    
    queries = [
        "Find hotels in Basel with Basel in it's name.",
        "Can you book the Hilton Basel for me?",
        "Oh wait, this is too expensive. Please cancel it and book the Hyatt Regency instead.",
        "My check in dates would be from April 10, 2024 to April 19, 2024.",
    ]
    
    
    async def run_application():
        llm = GoogleGenAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model="gemini-2.0-flash-001",
        )
    
    
        async with ToolboxClient("http://127.0.0.1:5000") as client:
                tools = await client.aload_toolset()
            
                agent = AgentWorkflow.from_tools_or_functions(
                    tools,
                    llm=llm,
                )
            
                for tool in tools:
                    logger.debug(tool.metadata)
            
                ctx = Context(agent)
            
                for query in queries:
                    response = await agent.run(user_msg=query, ctx=ctx)
                    logger.debug()
                    logger.debug(f"---- {query} ----")
                    logger.debug(str(response))
            
            
        logger.success(format_json(result))
    await run_application()
    
    """
    ### Advanced Toolbox Features
    
    Toolbox has a variety of features to make developing Gen AI tools for databases.
    For more information, read more about the following features:
    
    * [Authenticated Parameters](https://googleapis.github.io/genai-toolbox/resources/tools/#authenticated-parameters): bind tool inputs to values from OIDC tokens automatically, making it easy to run sensitive queries without potentially leaking data
    * [Authorized Invocations](https://googleapis.github.io/genai-toolbox/resources/tools/#authorized-invocations): restrict access to use a tool based on the users Auth token
    * [OpenTelemetry](https://googleapis.github.io/genai-toolbox/how-to/export_telemetry/): get metrics and tracing from Toolbox with OpenTelemetry
    """
    logger.info("### Advanced Toolbox Features")
    
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