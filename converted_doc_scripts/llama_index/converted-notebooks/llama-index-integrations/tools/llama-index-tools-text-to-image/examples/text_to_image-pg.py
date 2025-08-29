async def main():
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index import SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.workflow import Context
    from llama_index.llms import OllamaFunctionCallingAdapter
    from llama_index.tools import QueryEngineTool, ToolMetadata
    from llama_index.tools.text_to_image.base import TextToImageToolSpec
    import os
    import requests
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    
    # os.environ["OPENAI_API_KEY"] = "sk-..."
    
    
    
    response = requests.get(
        "https://www.dropbox.com/s/f6bmb19xdg0xedm/paul_graham_essay.txt?dl=1"
    )
    essay_txt = response.text
    with open("pg_essay.txt", "w") as fp:
        fp.write(essay_txt)
    
    documents = SimpleDirectoryReader(input_files=["pg_essay.txt"]).load_data()
    
    index = VectorStoreIndex.from_documents(documents)
    
    query_engine = index.as_query_engine()
    
    query_engine_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="paul_graham",
            description=(
                "Provides a biography of Paul Graham, from childhood to college to adult"
                " life"
            ),
        ),
    )
    
    
    llm = OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096)
    
    text_to_image_spec = TextToImageToolSpec()
    tools = text_to_image_spec.to_tool_list()
    agent = FunctionAgent(
        tools=tools + [query_engine_tool], llm=llm
    )
    
    ctx = Context(agent)
    
    logger.debug(
        await agent.run(
            "generate an image of the car that Paul Graham bought after Yahoo bought his"
            " company",
            ctx=ctx
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