from jet.logger import logger
from langchain_community.chat_models.dappier import ChatDappierAI
from langchain_core.messages import HumanMessage
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
    # Dappier AI
    
    **Dappier: Powering AI with Dynamic, Real-Time Data Models**
    
    Dappier offers a cutting-edge platform that grants developers immediate access to a wide array of real-time data models spanning news, entertainment, finance, market data, weather, and beyond. With our pre-trained data models, you can supercharge your AI applications, ensuring they deliver precise, up-to-date responses and minimize inaccuracies.
    
    Dappier data models help you build next-gen LLM apps with trusted, up-to-date content from the world's leading brands. Unleash your creativity and enhance any GPT App or AI workflow with actionable, proprietary, data through a simple API. Augment your AI with proprietary data from trusted sources is the best way to ensure factual, up-to-date, responses with fewer hallucinations no matter the question.
    
    For Developers, By Developers
    Designed with developers in mind, Dappier simplifies the journey from data integration to monetization, providing clear, straightforward paths to deploy and earn from your AI models. Experience the future of monetization infrastructure for the new internet at **https://dappier.com/**.
    
    This example goes over how to use LangChain to interact with Dappier AI models
    
    -----------------------------------------------------------------------------------
    
    To use one of our Dappier AI Data Models, you will need an API key. Please visit Dappier Platform (https://platform.dappier.com/) to log in and create an API key in your profile.
    
    
    You can find more details on the API reference : https://docs.dappier.com/introduction
    
    To work with our Dappier Chat Model you can pass the key directly through the parameter named dappier_api_key when initiating the class
    or set as an environment variable.
    
    ```bash
    export DAPPIER_API_KEY="..."
    ```
    """
    logger.info("# Dappier AI")
    
    
    chat = ChatDappierAI(
        dappier_endpoint="https://api.dappier.com/app/datamodelconversation",
        dappier_model="dm_01hpsxyfm2fwdt2zet9cg6fdxt",
        dappier_)
    
    messages = [HumanMessage(content="Who won the super bowl in 2024?")]
    chat.invoke(messages)
    
    await chat.ainvoke(messages)
    
    
    
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