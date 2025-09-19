async def main():
    from jet.transformers.formatters import format_json
    from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.tools.tool_spec.load_and_search import (
        LoadAndSearchToolSpec,
    )
    import os
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    """
    # Building a Exa Search Powered Data Agent
    
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-exa/examples/exa.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    This tutorial walks through using the LLM tools provided by the [Exa API](https://exa.ai) to allow LLMs to use semantic queries to search for and retrieve rich web content from the internet.
    
    To get started, you will need an [OllamaFunctionCalling api key](https://platform.openai.com/account/api-keys) and an [Exa API key](https://dashboard.exa.ai/api-keys)
    
    We will import the relevant agents and tools and pass them our keys here:
    """
    logger.info("# Building a Exa Search Powered Data Agent")

    # !pip install llama-index llama-index-core llama-index-tools-exa

    exa_tool = ExaToolSpec(
        api_key=os.environ["EXA_API_KEY"],
    )

    exa_tool_list = exa_tool.to_tool_list()
    for tool in exa_tool_list:
        logger.debug(tool.metadata.name)

    """
    ## Testing the Exa tools
    
    We've imported our OllamaFunctionCalling agent, set up the API keys, and initialized our tool, checking the methods that it has available. Let's test out the tool before setting up our Agent.
    
    All of the Exa search tools make use of the `AutoPrompt` option where Exa will pass the query through an LLM to refine it in line with Exa query best-practice.
    """
    logger.info("## Testing the Exa tools")

    exa_tool.search_and_retrieve_documents(
        "machine learning transformers", num_results=3)

    exa_tool.find_similar(
        "https://www.mihaileric.com/posts/transformers-attention-in-disguise/"
    )

    exa_tool.search_and_retrieve_documents(
        "This is a summary of recent research around diffusion models:", num_results=1
    )

    """
    While `search_and_retrieve_documents` returns raw text from the source document, `search_and_retrieve_highlights` returns relevant curated snippets.
    """
    logger.info("While `search_and_retrieve_documents` returns raw text from the source document, `search_and_retrieve_highlights` returns relevant curated snippets.")

    exa_tool.search_and_retrieve_highlights(
        "This is a summary of recent research around diffusion models:", num_results=1
    )

    """
    ### Exploring other Exa functionalities
    There are additional parameters that you can pass to Exa methods.
    
    You can filter return results based on the date that entity was published
    """
    logger.info("### Exploring other Exa functionalities")

    exa_tool.search_and_retrieve_documents(
        "Advancements in quantum computing",
        num_results=3,
        start_published_date="2024-01-01",
        end_published_date="2024-07-10",
    )

    """
    You can constrain results to only return from specified domains (or exclude domains)
    """
    logger.info(
        "You can constrain results to only return from specified domains (or exclude domains)")

    exa_tool.search_and_retrieve_documents(
        "Climate change mitigation strategies",
        num_results=3,
        include_domains=["www.nature.com",
                         "www.sciencemag.org", "www.pnas.org"],
    )

    """
    You can turn off autoprompt, enabling more direct and fine grained control of Exa querying.
    """
    logger.info(
        "You can turn off autoprompt, enabling more direct and fine grained control of Exa querying.")

    exa_tool.search_and_retrieve_documents(
        "Here is an article on the advancements of quantum computing",
        num_results=3,
        use_autoprompt=False,
    )

    """
    Exa also has an option to do standard keyword based seach by specifying `type="keyword"`.
    """
    logger.info("Exa also has an option to do standard keyword based seach by specifying `type="keyword"`.")

    exa_tool.search_and_retrieve_highlights(
        "Advancements in quantum computing", num_results=3, type="keyword"
    )

    """
    Last, Magic Search is a new feature available in Exa, where queries will route to the best suited search type intelligently: either their proprietary neural search or industry-standard keyword search mentioned above
    """
    logger.info("Last, Magic Search is a new feature available in Exa, where queries will route to the best suited search type intelligently: either their proprietary neural search or industry-standard keyword search mentioned above")

    exa_tool.search_and_retrieve_highlights(
        "Advancements in quantum computing", num_results=3, type="magic"
    )

    """
    We can see we have different tools to search for results, retrieve the results, find similar results to a web page, and finally a tool that combines search and document retrieval into a single tool. We will test them out in LLM Agents below:
    
    ### Using the Search and Retrieve documents tools in an Agent
    
    We can create an agent with access to the above tools and start testing it out:
    """
    logger.info("### Using the Search and Retrieve documents tools in an Agent")

    agent = FunctionAgent(
        tools=exa_tool_list,
        llm=OllamaFunctionCalling(model="llama3.2"),
    )

    logger.debug(await agent.run("What are the best resturants in toronto?"))

    """
    ## Avoiding Context Window Issues
    
    The above example shows the core uses of the Exa tool. We can easily retrieve a clean list of links related to a query, and then we can fetch the content of the article as a cleaned up html extract. Alternatively, the search_and_retrieve_documents tool directly returns the documents from our search result.
    
    We can see that the content of the articles is somewhat long and may overflow current LLM context windows.  
    
    1. Use `search_and_retrieve_highlights`: This is an endpoint offered by Exa that directly retrieves relevant highlight snippets from the web, instead of full web articles. As a result you don't need to worry about indexing/chunking offline yourself!
    
    2. Wrap `search_and_retrieve_documents` with `LoadAndSearchToolSpec`: We set up and use a "wrapper" tool from LlamaIndex that allows us to load text from any tool into a VectorStore, and query it for retrieval. This is where the `search_and_retrieve_documents` tool become particularly useful. The Agent can make a single query to retrieve a large number of documents, using a very small number of tokens, and then make queries to retrieve specific information from the documents.
    
    ### 1. Using `search_and_retrieve_highlights`
    
    The easiest is to just use `search_and_retrieve_highlights` from Exa. This is essentially a "web RAG" endpoint - they handle chunking/embedding under the hood.
    """
    logger.info("## Avoiding Context Window Issues")

    tools = exa_tool.to_tool_list(
        spec_functions=["search_and_retrieve_highlights", "current_date"]
    )

    agent = FunctionAgent(
        tools=tools,
        llm=OllamaFunctionCalling(model="llama3.2"),
    )

    response = await agent.run("Tell me more about the recent news on semiconductors")
    logger.success(format_json(response))
    logger.debug(f"Response: {str(response)}")

    """
    ### 2. Using `LoadAndSearchToolSpec`
    
    Here we wrap the `search_and_retrieve_documents` functionality with the `load_and_search_tool_spec`.
    """
    logger.info("### 2. Using `LoadAndSearchToolSpec`")

    search_and_retrieve_docs_tool = exa_tool.to_tool_list(
        spec_functions=["search_and_retrieve_documents"]
    )[0]
    date_tool = exa_tool.to_tool_list(spec_functions=["current_date"])[0]
    wrapped_retrieve = LoadAndSearchToolSpec.from_defaults(
        search_and_retrieve_docs_tool)

    """
    Our wrapped retrieval tools separate loading and reading into separate interfaces. We use `load` to load the documents into the vector store, and `read` to query the vector store. Let's try it out again
    """
    logger.info("Our wrapped retrieval tools separate loading and reading into separate interfaces. We use `load` to load the documents into the vector store, and `read` to query the vector store. Let's try it out again")

    wrapped_retrieve.load(
        "This is the best explanation for machine learning transformers:")
    logger.debug(wrapped_retrieve.read("what is a transformer"))
    logger.debug(wrapped_retrieve.read(
        "who wrote the first paper on transformers"))

    """
    ## Creating the Agent
    
    We now are ready to create an Agent that can use Exa's services to their full potential. We will use our wrapped read and load tools, as well as the `get_date` utility for the following agent and test it out below:
    """
    logger.info("## Creating the Agent")

    agent = FunctionAgent(
        tools=[*wrapped_retrieve.to_tool_list(), date_tool],
        llm=OllamaFunctionCalling(model="llama3.2"),
    )

    logger.debug(
        await agent.run(
            "Can you summarize everything published in the last month regarding news on superconductors"
        )
    )

    """
    We asked the agent to retrieve documents related to superconductors from this month. It used the `get_date` tool to determine the current month, and then applied the filters in Exa based on publication date when calling `search`. It then loaded the documents using `retrieve_documents` and read them using `read_retrieve_documents`.
    
    We can make another query to the vector store to read from it again, now that the articles are loaded:
    """
    logger.info("We asked the agent to retrieve documents related to superconductors from this month. It used the `get_date` tool to determine the current month, and then applied the filters in Exa based on publication date when calling `search`. It then loaded the documents using `retrieve_documents` and read them using `read_retrieve_documents`.")

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
