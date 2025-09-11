from jet.transformers.formatters import format_json
from ads4gpts_langchain import Ads4gptsInlineSponsoredResponseTool, Ads4gptsToolkit
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
import asyncio
import os
import shutil

async def main():
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger.basicConfig(filename=log_file)
    logger.info(f"Logs: {log_file}")
    
    PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
    os.makedirs(PERSIST_DIR, exist_ok=True)
    
    """
    # ADS4GPTs
    
    Integrate AI native advertising into your Agentic application.
    
    ## Overview
    
    This notebook outlines how to use the ADS4GPTs Tools and Toolkit in LangChain directly. In your LangGraph application though you will most likely use our prebuilt LangGraph agents.
    
    ## Setup
    
    ### Install ADS4GPTs Package
    Install the ADS4GPTs package using pip.
    """
    logger.info("# ADS4GPTs")
    
    # !pip install ads4gpts-langchain
    
    """
    Set up the environment variables for API authentication ([Obtain API Key](https://www.ads4gpts.com)).
    """
    logger.info("Set up the environment variables for API authentication ([Obtain API Key](https://www.ads4gpts.com)).")
    
    if not os.environ.get("ADS4GPTS_API_KEY"):
    #     os.environ["ADS4GPTS_API_KEY"] = getpass("Enter your ADS4GPTS API key: ")
    
    """
    ## Instantiation
    
    Import the necessary libraries, including ADS4GPTs tools and toolkit.
    
    Initialize the ADS4GPTs tools such as Ads4gptsInlineSponsoredResponseTool. We are going to work with one tool because the process is the same for every other tool we provide.
    """
    logger.info("## Instantiation")
    
    # from getpass import getpass
    
    
    inline_sponsored_response_tool = Ads4gptsInlineSponsoredResponseTool(
        ads4gpts_api_key=os.environ["ADS4GPTS_API_KEY"],
    )
    
    """
    ### Toolkit Instantiation
    Initialize the Ads4gptsToolkit with the required parameters.
    """
    logger.info("### Toolkit Instantiation")
    
    toolkit = Ads4gptsToolkit(
        ads4gpts_api_key=os.environ["ADS4GPTS_API_KEY"],
    )
    
    tools = toolkit.get_tools()
    
    for tool in tools:
        logger.debug(f"Initialized tool: {tool.__class__.__name__}")
    
    """
    ## Invocation
    
    Run the ADS4GPTs tools with sample inputs and display the results.
    """
    logger.info("## Invocation")
    
    sample_input = {
        "id": "test_id",
        "user_gender": "female",
        "user_age": "25-34",
        "user_persona": "test_persona",
        "ad_recommendation": "test_recommendation",
        "undesired_ads": "test_undesired_ads",
        "context": "test_context",
        "num_ads": 1,
        "style": "neutral",
    }
    
    inline_sponsored_response_result = inline_sponsored_response_tool._run(
        **sample_input, ad_format="INLINE_SPONSORED_RESPONSE"
    )
    logger.debug("Inline Sponsored Response Result:", inline_sponsored_response_result)
    
    """
    ### Async Run ADS4GPTs Tools
    Run the ADS4GPTs tools asynchronously with sample inputs and display the results.
    """
    logger.info("### Async Run ADS4GPTs Tools")
    
    
    
    async def run_ads4gpts_tools_async():
        inline_sponsored_response_result = await inline_sponsored_response_tool._arun(
                **sample_input, ad_format="INLINE_SPONSORED_RESPONSE"
            )
        logger.success(format_json(inline_sponsored_response_result))
        logger.debug("Async Inline Sponsored Response Result:", inline_sponsored_response_result)
    
    """
    ### Toolkit Invocation
    Use the Ads4gptsToolkit to get and run tools.
    """
    logger.info("### Toolkit Invocation")
    
    sample_input = {
        "id": "test_id",
        "user_gender": "female",
        "user_age": "25-34",
        "user_persona": "test_persona",
        "ad_recommendation": "test_recommendation",
        "undesired_ads": "test_undesired_ads",
        "context": "test_context",
        "num_ads": 1,
        "style": "neutral",
    }
    
    tool = tools[0]
    result = tool._run(**sample_input)
    logger.debug(f"Result from {tool.__class__.__name__}:", result)
    
    
    async def run_toolkit_tools_async():
        result = await tool._arun(**sample_input)
        logger.success(format_json(result))
        logger.debug(f"Async result from {tool.__class__.__name__}:", result)
    
    
    await run_toolkit_tools_async()
    
    """
    ## Chaining
    """
    logger.info("## Chaining")
    
    # if not os.environ.get("OPENAI_API_KEY"):
    #     os.environ["OPENAI_API_KEY"] = getpass("Enter your OPENAI_API_KEY API key: ")
    
    
    
    # ollama_model = ChatOllama(model="llama3.2")
    model = ollama_model.bind_tools(tools)
    model_response = model.invoke(
        "Get me an ad for clothing. I am a young man looking to go out with friends."
    )
    logger.debug("Tool call:", model_response)
    
    """
    ## API reference
    
    You can learn more about ADS4GPTs and the tools at our [GitHub](https://github.com/ADS4GPTs/ads4gpts/tree/main)
    """
    logger.info("## API reference")
    
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