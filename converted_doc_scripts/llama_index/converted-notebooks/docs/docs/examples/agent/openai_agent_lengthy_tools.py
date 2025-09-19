async def main():
    from jet.transformers.formatters import format_json
    from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
    from jet.logger import CustomLogger
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core import get_response_synthesizer
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.tools import QueryEngineTool
    from llama_index.core.tools import QueryPlanTool
    from llama_index.core.tools.types import ToolMetadata
    import os
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    """
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/openai_agent_query_plan.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # OllamaFunctionCalling Agent Workarounds for Lengthy Tool Descriptions
    In this demo, we illustrate a workaround for defining an OllamaFunctionCalling tool
    whose description exceeds OllamaFunctionCalling's current limit of 1024 characters.
    For simplicity, we will build upon the `QueryPlanTool` notebook
    example.
    
    If you're opening this Notebook on Colab, you will probably need to install LlamaIndex 🦙.
    """
    logger.info(
        "# OllamaFunctionCalling Agent Workarounds for Lengthy Tool Descriptions")

    # %pip install llama-index-agent-openai
    # %pip install llama-index-llms-ollama

    # !pip install llama-index

    # %load_ext autoreload
    # %autoreload 2

    # os.environ["OPENAI_API_KEY"] = "sk-..."

    llm = OllamaFunctionCalling(
        temperature=0, model="llama3.2")

    """
    ## Download Data
    """
    logger.info("## Download Data")

    # !mkdir -p 'data/10q/'
    # !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10q/uber_10q_march_2022.pdf' -O 'data/10q/uber_10q_march_2022.pdf'
    # !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10q/uber_10q_june_2022.pdf' -O 'data/10q/uber_10q_june_2022.pdf'
    # !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10q/uber_10q_sept_2022.pdf' -O 'data/10q/uber_10q_sept_2022.pdf'

    """
    ## Load data
    """
    logger.info("## Load data")

    march_2022 = SimpleDirectoryReader(
        input_files=[
            "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp/10q/uber_10q_march_2022.pdf"]
    ).load_data()
    june_2022 = SimpleDirectoryReader(
        input_files=[
            "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp/10q/uber_10q_june_2022.pdf"]
    ).load_data()
    sept_2022 = SimpleDirectoryReader(
        input_files=[
            "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp/10q/uber_10q_sept_2022.pdf"]
    ).load_data()

    """
    ## Build indices
    
    We build a vector index / query engine over each of the documents (March, June, September).
    """
    logger.info("## Build indices")

    march_index = VectorStoreIndex.from_documents(march_2022)
    june_index = VectorStoreIndex.from_documents(june_2022)
    sept_index = VectorStoreIndex.from_documents(sept_2022)

    march_engine = march_index.as_query_engine(similarity_top_k=3, llm=llm)
    june_engine = june_index.as_query_engine(similarity_top_k=3, llm=llm)
    sept_engine = sept_index.as_query_engine(similarity_top_k=3, llm=llm)

    """
    ## Defining an Excessively Lengthy Query Plan
    
    Although a `QueryPlanTool` may be composed from many `QueryEngineTools`,
    a single OllamaFunctionCalling tool is ultimately created from the `QueryPlanTool`
    when the OllamaFunctionCalling API call is made. The description of this tool begins with
    general instructions about the query plan approach, followed by the
    descriptions of each constituent `QueryEngineTool`.
    
    Currently, each OllamaFunctionCalling tool description has a maximum length of 1024 characters.
    As you add more `QueryEngineTools` to your `QueryPlanTool`, you may exceed
    this limit. If the limit is exceeded, LlamaIndex will raise an error when it
    attempts to construct the OllamaFunctionCalling tool.
    
    Let's demonstrate this scenario with an exaggerated example, where we will
    give each query engine tool a very lengthy and redundant description.
    """
    logger.info("## Defining an Excessively Lengthy Query Plan")

    description_10q_general = """\
    A Form 10-Q is a quarterly report required by the SEC for publicly traded companies,
    providing an overview of the company's financial performance for the quarter.
    It includes unaudited financial statements (income statement, balance sheet,
    and cash flow statement) and the Management's Discussion and Analysis (MD&A),
    where management explains significant changes and future expectations.
    The 10-Q also discloses significant legal proceedings, updates on risk factors,
    and information on the company's internal controls. Its primary purpose is to keep
    investors informed about the company's financial status and operations,
    enabling informed investment decisions."""

    description_10q_specific = (
        "This 10-Q provides Uber quarterly financials ending"
    )

    query_tool_sept = QueryEngineTool.from_defaults(
        query_engine=sept_engine,
        name="sept_2022",
        description=f"{description_10q_general} {description_10q_specific} September 2022",
    )
    query_tool_june = QueryEngineTool.from_defaults(
        query_engine=june_engine,
        name="june_2022",
        description=f"{description_10q_general} {description_10q_specific} June 2022",
    )
    query_tool_march = QueryEngineTool.from_defaults(
        query_engine=march_engine,
        name="march_2022",
        description=f"{description_10q_general} {description_10q_specific} March 2022",
    )

    logger.debug(len(query_tool_sept.metadata.description))
    logger.debug(len(query_tool_june.metadata.description))
    logger.debug(len(query_tool_march.metadata.description))

    """
    From the print statements above, we see that we will easily exceed the
    maximum character limit of 1024 when composing these tools into the `QueryPlanTool`.
    """
    logger.info(
        "From the print statements above, we see that we will easily exceed the")

    query_engine_tools = [query_tool_sept, query_tool_june, query_tool_march]

    response_synthesizer = get_response_synthesizer()
    query_plan_tool = QueryPlanTool.from_defaults(
        query_engine_tools=query_engine_tools,
        response_synthesizer=response_synthesizer,
    )

    openai_tool = query_plan_tool.metadata.to_openai_tool()

    """
    ## Moving Tool Descriptions to the Prompt
    
    One obvious solution to this problem would be to shorten the tool
    descriptions themselves, however with sufficiently many tools,
    we will still eventually exceed the character limit.
    
    A more scalable solution would be to move the tool descriptions to the prompt.
    This solves the character limit issue, since without the descriptions
    of the query engine tools, the query plan description will remain fixed
    in size. Of course, token limits imposed by the selected LLM will still
    bound the tool descriptions, however these limits are far larger than the
    1024 character limit.
    
    There are two steps involved in moving these tool descriptions to the
    prompt. First, we must modify the metadata property of the `QueryPlanTool`
    to omit the `QueryEngineTool` descriptions, and make a slight modification
    to the default query planning instructions (telling the LLM to look for the
    tool names and descriptions in the prompt.)
    """
    logger.info("## Moving Tool Descriptions to the Prompt")

    introductory_tool_description_prefix = """\
    This is a query plan tool that takes in a list of tools and executes a \
    query plan over these tools to answer a query. The query plan is a DAG of query nodes.
    
    Given a list of tool names and the query plan schema, you \
    can choose to generate a query plan to answer a question.
    
    The tool names and descriptions will be given alongside the query.
    """

    new_metadata = ToolMetadata(
        introductory_tool_description_prefix,
        query_plan_tool.metadata.name,
        query_plan_tool.metadata.fn_schema,
    )
    query_plan_tool.metadata = new_metadata
    query_plan_tool.metadata

    """
    Second, we must concatenate our tool names and descriptions alongside
    the query being posed.
    """
    logger.info(
        "Second, we must concatenate our tool names and descriptions alongside")

    agent = FunctionAgent(
        tools=[query_plan_tool],
        llm=OllamaFunctionCalling(
            temperature=0, model="llama3.2"),
    )

    query = "What were the risk factors in sept 2022?"

    tools_description = "\n\n".join(
        [
            f"Tool Name: {tool.metadata.name}\n"
            + f"Tool Description: {tool.metadata.description} "
            for tool in query_engine_tools
        ]
    )

    query_planned_query = f"{tools_description}\n\nQuery: {query}"
    query_planned_query

    response = await agent.run(query_planned_query)
    logger.success(format_json(response))
    response

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
