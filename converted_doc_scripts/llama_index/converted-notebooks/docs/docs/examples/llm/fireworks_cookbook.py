async def main():
    from jet.transformers.formatters import format_json
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter.utils import to_openai_tool
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.tools import BaseTool, FunctionTool
    from llama_index.llms.fireworks import Fireworks
    from llama_index.program.openai import OllamaFunctionCallingAdapterPydanticProgram
    from pydantic import BaseModel
    import os
    import shutil


    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    """
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/fireworks_cookbook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # Fireworks Function Calling Cookbook
    
    Fireworks.ai supports function calling for its LLMs, similar to OllamaFunctionCalling. This lets users directly describe the set of tools/functions available and have the model dynamically pick the right function calls to invoke, without complex prompting on the user's part.
    
    Since our Fireworks LLM directly subclasses OllamaFunctionCalling, we can use our existing abstractions with Fireworks.
    
    We show this on three levels: directly on the model API, as part of a Pydantic Program (structured output extraction), and as part of an agent.
    """
    logger.info("# Fireworks Function Calling Cookbook")

    # %pip install llama-index-llms-fireworks

    # %pip install llama-index


    os.environ["FIREWORKS_API_KEY"] = ""


    llm = Fireworks(
        model="accounts/fireworks/models/firefunction-v1", temperature=0
    )

    """
    ## Function Calling on the LLM Module
    
    You can directly input function calls on the LLM module.
    """
    logger.info("## Function Calling on the LLM Module")



    class Song(BaseModel):
        """A song with name and artist"""

        name: str
        artist: str


    song_fn = to_openai_tool(Song)


    response = llm.complete("Generate a song from Beyonce", tools=[song_fn])
    tool_calls = response.additional_kwargs["tool_calls"]
    logger.debug(tool_calls)

    """
    ## Using a Pydantic Program
    
    Our Pydantic programs allow structured output extraction into a Pydantic object. `OllamaFunctionCallingAdapterPydanticProgram` takes advantage of function calling for structured output extraction.
    """
    logger.info("## Using a Pydantic Program")


    prompt_template_str = "Generate a song about {artist_name}"
    program = OllamaFunctionCallingAdapterPydanticProgram.from_defaults(
        output_cls=Song, prompt_template_str=prompt_template_str, llm=llm
    )

    output = program(artist_name="Eminem")

    output

    """
    ## Using An OllamaFunctionCalling Agent
    """
    logger.info("## Using An OllamaFunctionCalling Agent")




    def multiply(a: int, b: int) -> int:
        """Multiple two integers and returns the result integer"""
        return a * b


    multiply_tool = FunctionTool.from_defaults(fn=multiply)


    def add(a: int, b: int) -> int:
        """Add two integers and returns the result integer"""
        return a + b


    add_tool = FunctionTool.from_defaults(fn=add)

    agent = FunctionAgent(
        tools=[multiply_tool, add_tool],
        llm=llm,
    )

    response = await agent.run("What is (121 * 3) + 42?")
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