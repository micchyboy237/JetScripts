async def main():
    from graphql import parse
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.file.sdl.base import SDLReader
    from llama_index.tools.ondemand_loader_tool import OnDemandLoaderTool
    from llama_index.tools.shopify.base import ShopifyToolSpec
    import json
    import openai
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    
    openai.api_key = "sk-your-key"
    
    """
    ## Leveraging the GraphQL schema in our Agent
    
    The schema was retrieved using the `apollo client:download-schema` command: `apollo client:download-schema download3.json --endpoint=https://your-store.myshopify.com/admin/api/2023-01/graphql.json --header="X-Shopify-Access-Token: your-token"`
    
    All in all, the file is over 50,000 lines and close to 1.5 million characters, well beyond what we could hope to process directly with any Large Language Model. Instead, we have to get creative with how we will process and retrieve it.
    
    In the below code block we open the GraphQL schema for the Shopify store and parse out the **QueryRoot** objects.
    These are then directly passed into the system prompt, so that the Agent is aware of the objects it can query against.
    From the schema, a **QueryRoot** is `The schema's entry-point for queries. This acts as the public, top-level API from which all queries must start.` Because these obejcts are so critical to writing good queries, it's worth passing them into the agent.
    """
    logger.info("## Leveraging the GraphQL schema in our Agent")
    
    
    with open("data/shopify_graphql.txt", "r") as f:
        txt = f.read()
    
    ast = parse(txt)
    
    query_root_node = next(
        (
            defn
            for defn in ast.definitions
            if defn.kind == "object_type_definition" and defn.name.value == "QueryRoot"
        )
    )
    query_roots = [field.name.value for field in query_root_node.fields]
    logger.debug(query_roots)
    
    """
    ## Setting up SDLReader and OnDemandLoaderTool
    
    We've successfully parsed out the **QueryRoot** fields that are usable for top level GraphQL queries. Now we can combine the **SDLReader** and **OnDemandLoaderTool** to create an interface that our Agent can use to query and process the GraphQL schema.
    
    The **SDLReader** is consuming our GraphQL spec and intelligently breaking it into chunks based on the definitions in the schema. By wrapping the **SDLReader** with the **OnDemandLoaderTool**, there is essentially a sub-model that is processing our query_str, retriving any relevant chunks of data from the GraphQL schema, and then intrpreting those chunks in relation to our query. This lets us ask arbitrary natural language questions, and get back intelligent responses based on the GraphQL schema.
    """
    logger.info("## Setting up SDLReader and OnDemandLoaderTool")
    
    
    documentation_tool = OnDemandLoaderTool.from_defaults(
        SDLReader(),
        name="graphql_writer",
        description="""
            The GraphQL schema file is located at './data/shopify_graphql.txt', this is always the file argument.
            A tool for processing the Shopify GraphQL spec, and writing queries from the documentation.
    
            You should pass a query_str to this tool in the form of a request to write a GraphQL query.
    
            Examples:
                file: './data/shopify_graphql.txt', query_str='Write a graphql query to find unshipped orders'
                file: './data/shopify_graphql.txt', query_str='Write a graphql query to retrieve the stores products'
                file: './data/shopify_graphql.txt', query_str='What fields can you retrieve from the orders object'
    
            """,
    )
    
    logger.debug(
        documentation_tool(
            "./data/shopify_graphql.txt",
            query_str="Write a graphql query to retrieve the first 3 products from a store",
        )
    )
    logger.debug(
        documentation_tool(
            "./data/shopify_graphql.txt",
            query_str="what fields can you retrieve from the products object",
        )
    )
    
    """
    ## Setting up the Shopify Tool
    
    We've now set up a tool that ourselves or an Agent can call with natural language, and get information or create queries based on our schema. We can now initialize the Shopify tool and even test it out with the prompt that was written, adding in some of the extra fields the documentation returned us:
    """
    logger.info("## Setting up the Shopify Tool")
    
    
    shopify_tool = ShopifyToolSpec("your-store.myshopify.com", "2023-04", "your-key")
    
    shopify_tool.run_graphql_query(
        """
    query {
      products(first: 3) {
        edges {
          node {
            title
            vendor
            productType
            status
          }
        }
      }
    }"""
    )
    
    """
    ## Creating a Data Agent
    
    So now we have two tools, one that can create working GraphQL queries and provide information from a GraphQL schema using natural language strings, and one that can execute the GraphQL queries and return the results.
    
    Our next step is to pass these tools to a Data Agent, and allow them access to use the tools and interpret the outputs for the user. We supply the Agent with a system prompt on initilization that gives them some extra info, like the **QueryRoot** objects we processed above, and some instructions on how to effectively use the tools:
    """
    logger.info("## Creating a Data Agent")
    
    
    agent = FunctionAgent(
        tools=[*shopify_tool.to_tool_list(), documentation_tool],
        system_prompt=f"""
        You are a specialized Agent with access to the Shopify Admin GraphQL API for this Users online store.
        Your job is to chat with store owners and help them run GraphQL queries, interpreting the results for the user
    
        For you conveinence, the QueryRoot objects are listed here.
    
        {query_roots}
    
        QueryRoots are the schema's entry-point for queries. This acts as the public, top-level API from which all queries must start.
    
        You can use graphql_writer to query the schema and assist in writing queries.
    
        If the GraphQL you execute returns an error, either directly fix the query, or directly ask the graphql_writer questions about the schema instead of writing graphql queries.
        Then use that information to write the correct graphql query
        """,
        llm=OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096),
    )
    
    logger.debug(await agent.run("What are the most recent orders my store received"))
    
    """
    ## Conclusion
    
    We can see the Agent was able to handle the errors from the GraphQL endpoint to modify the queries, and used our documentation tool to gather more information on the schema to ulimately return a helpful response to the user. Neat!
    """
    logger.info("## Conclusion")
    
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