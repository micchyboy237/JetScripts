async def main():
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.tools.ondemand_loader_tool import OnDemandLoaderTool
    from llama_index.core.workflow import Context
    from llama_index.tools.google import GmailToolSpec
    from llama_index.tools.multion import MultionToolSpec
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    # MultiOn Demo
    
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-multion/examples/multion.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    This notebook walks through an example of using LlamaIndex with MultiOn to browse the web on the users behalf.
    
    First, we import the FunctionAgent that will control the Multion session:
    """
    logger.info("# MultiOn Demo")
    
    
    # os.environ["OPENAI_API_KEY"] = "sk-your-key"
    
    
    """
    We then import the MultiOn tool and initialize our agent with the tool.
    """
    logger.info("We then import the MultiOn tool and initialize our agent with the tool.")
    
    
    multion_tool = MultionToolSpec(api_key="your-multion-key")
    
    """
    To support the MultiOn browsing session, we will also give our LlamaIndex agent a tool to search and summarize a users gmail inbox. We set up that tool below. For more information on the gmail tool, see the [Gmail notebook here](https://github.com/emptycrown/llama-hub/blob/main/llama_hub/tools/notebooks/gmail.ipynb).
    
    We will use this tool later on to allow the agent to gain more context around our emails
    """
    logger.info("To support the MultiOn browsing session, we will also give our LlamaIndex agent a tool to search and summarize a users gmail inbox. We set up that tool below. For more information on the gmail tool, see the [Gmail notebook here](https://github.com/emptycrown/llama-hub/blob/main/llama_hub/tools/notebooks/gmail.ipynb).")
    
    
    gmail_tool = GmailToolSpec()
    
    gmail_loader_tool = OnDemandLoaderTool.from_tool(
        gmail_tool.to_tool_list()[1],
        name="gmail_search",
        description="""
            This tool allows you to search the users gmail inbox and give directions for how to summarize or process the emails
    
            You must always provide a query to filter the emails, as well as a query_str to process the retrieved emails.
            All parameters are required
    
            If you need to reply to an email, ask this tool to build the reply directly
            Examples:
                query='from:adam subject:dinner', max_results=5, query_str='Where are adams favourite places to eat'
                query='dentist appointment', max_results=1, query_str='When is the next dentist appointment'
                query='to:jerry', max_results=1, query_str='summarize and then create a response email to jerrys latest email'
                query='is:inbox', max_results=5, query_str='Summarize these emails'
            """,
    )
    
    agent = FunctionAgent(
        tools=[*multion_tool.to_tool_list(), gmail_loader_tool],
        system_prompt="""
        You are an AI agent that assists the user in crafting email responses based on previous conversations.
    
        The gmail_search tool connects directly to an API to search and retrieve emails, and answer questions based on the content.
        The browse tool allows you to control a web browser with natural language to complete arbitrary actions on the web.
    
        Use these two tools together to gain context on past emails and respond to conversations for the user.
        """,
        llm=OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096),
    )
    
    
    ctx = Context(agent)
    
    """
    Our agent is now set up and ready to browse the web!
    """
    logger.info("Our agent is now set up and ready to browse the web!")
    
    logger.debug(await agent.run("browse to the latest email from Julian and open the email", ctx=ctx))
    
    logger.debug(
        await agent.run(
            "Summarize the email chain with julian and create a response to the last email"
            " that confirms all the details",
            ctx=ctx,
        )
    )
    
    logger.debug(
        await agent.run(
            "pass the entire generated email to the browser and have it send the email as a"
            " reply to the chain",
            ctx=ctx,
        )
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