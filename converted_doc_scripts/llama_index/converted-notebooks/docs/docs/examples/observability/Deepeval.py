async def main():
    from jet.transformers.formatters import format_json
    from deepeval.integrations.llama_index import FunctionAgent
    from deepeval.integrations.llama_index import instrument_llama_index
    from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent
    import asyncio
    import deepeval
    import llama_index.core.instrumentation as instrument
    import os
    import shutil
    import time


    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    """
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/observability/Deepeval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # DeepEval: Evaluation and Observability for LlamaIndex
    
    DeepEval (by [Confident AI](https://documentation.confident-ai.com/docs)) now integrates with LlamaIndex, giving you end-to-end visibility and evaluation tools for your LlamaIndex agents.
    
    ## Quickstart
    Install the following packages:
    """
    logger.info("# DeepEval: Evaluation and Observability for LlamaIndex")

    # !pip install -U deepeval llama-index

    """
    Login with your [Confident API key](https://confident-ai.com/) and configure DeepEval as instrument LlamaIndex:
    """
    logger.info("Login with your [Confident API key](https://confident-ai.com/) and configure DeepEval as instrument LlamaIndex:")



    deepeval.login("<your-confident-api-key>")

    instrument_llama_index(instrument.get_dispatcher())

    """
    ### Example Agent
    
    ⚠️ **Note**: DeepEval may not work reliably in Jupyter notebooks due to event loop conflicts. It is recommended to run examples in a standalone Python script instead.
    """
    logger.info("### Example Agent")




    deepeval.login("<your-confident-api-key>")

    instrument_llama_index(instrument.get_dispatcher())

    # os.environ["OPENAI_API_KEY"] = "<your-openai-api-key>"


    def multiply(a: float, b: float) -> float:
        """Useful for multiplying two numbers."""
        return a * b


    agent = FunctionAgent(
        tools=[multiply],
        llm=OllamaFunctionCalling(model="llama3.2"),
        system_prompt="You are a helpful assistant that can perform calculations.",
    )


    async def main():
        response = await agent.run("What's 7 * 8?")
        logger.success(format_json(response))
        logger.debug(response)


    if __name__ == "__main__":
        asyncio.run(main())

    """
    You can directly view the traces in the **Observatory** by clicking on the link in the output printed in the console.
    
    ### Online Evaluations
    
    You can use DeepEval to evaluate your LlamaIndex agents on Confident AI.
    
    1. Create a [metric collection](https://documentation.confident-ai.com/docs/llm-evaluation/metrics/create-on-the-cloud) on Confident AI.
    2. Pass the metric collection name on DeepEval's LlamaIndex agent wrapper.
    """
    logger.info("### Online Evaluations")




    deepeval.login("<your-confident-api-key>")

    instrument_llama_index(instrument.get_dispatcher())

    # os.environ["OPENAI_API_KEY"] = ""


    def multiply(a: float, b: float) -> float:
        """Useful for multiplying two numbers."""
        return a * b


    agent = FunctionAgent(
        tools=[multiply],
        llm=OllamaFunctionCalling(model="llama3.2"),
        system_prompt="You are a helpful assistant that can perform calculations.",
        metric_collection="test_collection_1",
    )


    async def main():
        response = await agent.run("What's 7 * 8?")
        logger.success(format_json(response))
        logger.debug(response)


    if __name__ == "__main__":
        asyncio.run(main())

    """
    ### References 
    
    - [Get started with DeepEval](https://deepeval.com/docs/getting-started)
    - [LlamaIndex integration](https://documentation.confident-ai.com/docs/llm-tracing/integrations/llamaindex)
    """
    logger.info("### References")

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