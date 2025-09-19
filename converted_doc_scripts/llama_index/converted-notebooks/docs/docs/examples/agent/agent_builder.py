async def main():
    from jet.models.config import MODELS_CACHE_DIR
    from jet.transformers.formatters import format_json
    from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
    from jet.logger import CustomLogger
    from llama_index.core import ChatPromptTemplate
    from llama_index.core import Settings
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core import VectorStoreIndex
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.agent.workflow import ToolCallResult
    from llama_index.core.llms import ChatMessage
    from llama_index.core.objects import ObjectIndex
    from llama_index.core.tools import FunctionTool
    from llama_index.core.tools import QueryEngineTool
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from pathlib import Path
    from typing import List
    import os
    import requests
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    """
    # GPT Builder Demo
    
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/agent_builder.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    Inspired by GPTs interface, presented at OllamaFunctionCalling Dev Day 2023. Construct an agent with natural language.
    
    Here you can build your own agent...with another agent!
    """
    logger.info("# GPT Builder Demo")

    # %pip install llama-index-embeddings-huggingface
    # %pip install llama-index-llms-ollama
    # %pip install llama-index-readers-file

    # os.environ["OPENAI_API_KEY"] = "sk-..."

    llm = OllamaFunctionCalling(model="llama3.2")
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)

    """
    ## Define Candidate Tools
    
    We also define a tool retriever to retrieve candidate tools.
    
    In this setting we define tools as different Wikipedia pages.
    """
    logger.info("## Define Candidate Tools")

    wiki_titles = ["Toronto", "Seattle", "Chicago", "Boston", "Houston"]

    for title in wiki_titles:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "explaintext": True,
            },
        ).json()
        page = next(iter(response["query"]["pages"].values()))
        wiki_text = page["extract"]

        data_path = Path("data")
        if not data_path.exists():
            Path.mkdir(data_path)

        with open(data_path / f"{title}.txt", "w") as fp:
            fp.write(wiki_text)

    city_docs = {}
    for wiki_title in wiki_titles:
        city_docs[wiki_title] = SimpleDirectoryReader(
            input_files=[f"data/{wiki_title}.txt"]
        ).load_data()

    """
    ### Build Query Tool for Each Document
    """
    logger.info("### Build Query Tool for Each Document")

    tool_dict = {}

    for wiki_title in wiki_titles:
        vector_index = VectorStoreIndex.from_documents(
            city_docs[wiki_title],
        )
        vector_query_engine = vector_index.as_query_engine(llm=llm)

        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            name=wiki_title,
            description=("Useful for questions related to" f" {wiki_title}"),
        )
        tool_dict[wiki_title] = vector_tool

    """
    ### Define Tool Retriever
    """
    logger.info("### Define Tool Retriever")

    tool_index = ObjectIndex.from_objects(
        list(tool_dict.values()),
        index_cls=VectorStoreIndex,
    )
    tool_retriever = tool_index.as_retriever(similarity_top_k=1)

    """
    ### Load Data
    
    Here we load wikipedia pages from different cities.
    
    ## Define Meta-Tools for GPT Builder
    """
    logger.info("### Load Data")

    GEN_SYS_PROMPT_STR = """\
    Task information is given below.
    
    Given the task, please generate a system prompt for an OllamaFunctionCalling-powered bot to solve this task:
    {task} \
    """

    gen_sys_prompt_messages = [
        ChatMessage(
            role="system",
            content="You are helping to build a system prompt for another bot.",
        ),
        ChatMessage(role="user", content=GEN_SYS_PROMPT_STR),
    ]

    GEN_SYS_PROMPT_TMPL = ChatPromptTemplate(gen_sys_prompt_messages)

    agent_cache = {}

    async def create_system_prompt(task: str):
        """Create system prompt for another agent given an input task."""
        llm = OllamaFunctionCalling(llm="gpt-4")
        fmt_messages = GEN_SYS_PROMPT_TMPL.format_messages(task=task)
        response = llm.chat(fmt_messages)
        logger.success(format_json(response))
        return response.message.content

    async def get_tools(task: str):
        """Get the set of relevant tools to use given an input task."""
        subset_tools = await tool_retriever.aretrieve(task)
        logger.success(format_json(subset_tools))
        return [t.metadata.name for t in subset_tools]

    def create_agent(system_prompt: str, tool_names: List[str]):
        """Create an agent given a system prompt and an input set of tools."""
        llm = OllamaFunctionCalling(model="llama3.2")
        try:
            input_tools = [tool_dict[tn] for tn in tool_names]

            agent = FunctionAgent(
                tools=input_tools, llm=llm, system_prompt=system_prompt
            )
            agent_cache["agent"] = agent
            return_msg = "Agent created successfully."
        except Exception as e:
            return_msg = f"An error occurred when building an agent. Here is the error: {repr(e)}"
        return return_msg

    system_prompt_tool = FunctionTool.from_defaults(fn=create_system_prompt)
    get_tools_tool = FunctionTool.from_defaults(fn=get_tools)
    create_agent_tool = FunctionTool.from_defaults(fn=create_agent)

    GPT_BUILDER_SYS_STR = """\
    You are helping to construct an agent given a user-specified task. You should generally use the tools in this order to build the agent.
    
    1) Create system prompt tool: to create the system prompt for the agent.
    2) Get tools tool: to fetch the candidate set of tools to use.
    3) Create agent tool: to create the final agent.
    """

    prefix_msgs = [ChatMessage(role="system", content=GPT_BUILDER_SYS_STR)]

    builder_agent = FunctionAgent(
        tools=[system_prompt_tool, get_tools_tool, create_agent_tool],
        prefix_messages=prefix_msgs,
        llm=OllamaFunctionCalling(model="llama3.2"),
        verbose=True,
    )

    handler = builder_agent.run(
        "Build an agent that can tell me about Toronto.")
    async for event in handler.stream_events():
        if isinstance(event, ToolCallResult):
            logger.debug(
                f"Called tool {event.tool_name} with input {event.tool_kwargs}\nGot output: {event.tool_output}"
            )

    result = await handler
    logger.success(format_json(result))
    logger.debug(f"Result: {result}")

    city_agent = agent_cache["agent"]

    response = await city_agent.run("Tell me about the parks in Toronto")
    logger.success(format_json(response))
    logger.debug(str(response))

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
