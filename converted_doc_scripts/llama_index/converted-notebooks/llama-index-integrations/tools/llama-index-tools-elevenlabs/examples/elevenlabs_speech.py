async def main():
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.tools.elevenlabs import ElevenLabsToolSpec
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    ### Prerequisites
    Make sure you have installed the following two packages
    ```
    llama-index-agent-openai
    llama-index-tools-elevenlabs
    ```
    """
    logger.info("### Prerequisites")
    
    
    # os.environ["OPENAI_API_KEY"] = "your-key"
    
    
    speech_tool = ElevenLabsToolSpec(api_key="your-key")
    
    agent = FunctionAgent(
        tools=[*speech_tool.to_tool_list()],
        llm=OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096),
    )
    logger.debug(
        await agent.run(
            'Get the list of available voices, select ONLY the first voice, and use it to create speech from the text "Hello world!" saving to "speech.wav"'
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