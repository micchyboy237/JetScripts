async def main():
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.workflow import Context
    from llama_index.tools.database.base import DatabaseToolSpec
    import os
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    """
    ## OllamaFunctionCallingAdapter 
    
    For this notebook we will use the OllamaFunctionCallingAdapter ChatGPT models. We import the OllamaFunctionCallingAdapter agent and set the api_key, you will have to provide your own API key.
    """
    logger.info("## OllamaFunctionCallingAdapter")

    # os.environ["OPENAI_API_KEY"] = "sk-your-key"

    """
    ## Database tool
    
    This tool connects to a database (using SQLAlchemy under the hood) and allows an Agent to query the database and get information about the tables.
    
    We import the ToolSpec and initialize it so that it can connect to our database
    """
    logger.info("## Database tool")

    db_spec = DatabaseToolSpec(
        scheme="postgresql",  # Database Scheme
        host="localhost",  # Database Host
        port="5432",  # Database Port
        user="postgres",  # Database User
        password="x",  # Database Password
        dbname="your_db",  # Database Name
    )

    """
    After initializing the Tool Spec we have an instance of the ToolSpec. A ToolSpec can have many tools that it implements and makes available to agents. We can see the Tools by converting to the spec to a list of FunctionTools, using `to_tool_list`
    """
    logger.info("After initializing the Tool Spec we have an instance of the ToolSpec. A ToolSpec can have many tools that it implements and makes available to agents. We can see the Tools by converting to the spec to a list of FunctionTools, using `to_tool_list`")

    tools = db_spec.to_tool_list()
    for tool in tools:
        logger.debug(tool.metadata.name)
        logger.debug(tool.metadata.description)
        logger.debug(tool.metadata.fn_schema)

    """
    We can see that the database tool spec provides 3 functions for the OllamaFunctionCallingAdapter agent. One to execute a SQL query, one to describe a list of tables in the database, and one to list all of the tables available in the database. 
    
    We can pass the tool list to our OllamaFunctionCallingAdapter agent and test it out:
    """
    logger.info("We can see that the database tool spec provides 3 functions for the OllamaFunctionCallingAdapter agent. One to execute a SQL query, one to describe a list of tables in the database, and one to list all of the tables available in the database.")

    agent = FunctionAgent(
        tools=db_tools.to_tool_list(),
        llm=OllamaFunctionCallingAdapter(model="llama3.2"),
    )

    ctx = Context(agent)

    """
    At this point our Agent is fully ready to start making queries to our database:
    """
    logger.info(
        "At this point our Agent is fully ready to start making queries to our database:")

    logger.debug(await agent.run("What tables does this database contain", ctx=ctx))

    logger.debug(await agent.run("Can you describe the messages table", ctx=ctx))

    logger.debug(await agent.run("Fetch the most recent message and display the body", ctx=ctx))

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
