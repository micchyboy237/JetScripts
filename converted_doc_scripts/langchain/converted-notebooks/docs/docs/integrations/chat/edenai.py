from jet.logger import logger
from langchain_community.chat_models.edenai import ChatEdenAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field
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
    # Eden AI
    
    Eden AI is revolutionizing the AI landscape by uniting the best AI providers, empowering users to unlock limitless possibilities and tap into the true potential of artificial intelligence. With an all-in-one comprehensive and hassle-free platform, it allows users to deploy AI features to production lightning fast, enabling effortless access to the full breadth of AI capabilities via a single API. (website: https://edenai.co/)
    
    This example goes over how to use LangChain to interact with Eden AI models
    
    -----------------------------------------------------------------------------------
    
    `EdenAI` goes beyond mere model invocation. It empowers you with advanced features, including:
    
    - **Multiple Providers**: Gain access to a diverse range of language models offered by various providers, giving you the freedom to choose the best-suited model for your use case.
    
    - **Fallback Mechanism**: Set a fallback mechanism to ensure seamless operations even if the primary provider is unavailable, you can easily switches to an alternative provider.
    
    - **Usage Tracking**: Track usage statistics on a per-project and per-API key basis. This feature allows you to monitor and manage resource consumption effectively.
    
    - **Monitoring and Observability**: `EdenAI` provides comprehensive monitoring and observability tools on the platform. Monitor the performance of your language models, analyze usage patterns, and gain valuable insights to optimize your applications.
    
    Accessing the EDENAI's API requires an API key, 
    
    which you can get by creating an account https://app.edenai.run/user/register  and heading here https://app.edenai.run/admin/iam/api-keys
    
    Once we have a key we'll want to set it as an environment variable by running:
    
    ```bash
    export EDENAI_API_KEY="..."
    ```
    
    You can find more details on the API reference : https://docs.edenai.co/reference
    
    If you'd prefer not to set an environment variable you can pass the key in directly via the edenai_api_key named parameter
    
     when initiating the EdenAI Chat Model class.
    """
    logger.info("# Eden AI")
    
    
    chat = ChatEdenAI(
        edenai_provider="ollama", temperature=0.2, max_tokens=250
    )
    
    messages = [HumanMessage(content="Hello !")]
    chat.invoke(messages)
    
    await chat.ainvoke(messages)
    
    """
    ## Streaming and Batching
    
    `ChatEdenAI` supports streaming and batching. Below is an example.
    """
    logger.info("## Streaming and Batching")
    
    for chunk in chat.stream(messages):
        logger.debug(chunk.content, end="", flush=True)
    
    chat.batch([messages])
    
    """
    ## Fallback mecanism
    
    With Eden AI you can set a fallback mechanism to ensure seamless operations even if the primary provider is unavailable, you can easily switches to an alternative provider.
    """
    logger.info("## Fallback mecanism")
    
    chat = ChatEdenAI(
        edenai_provider="ollama",
        temperature=0.2,
        max_tokens=250,
        fallback_providers="google",
    )
    
    """
    In this example, you can use Google as a backup provider if Ollama encounters any issues.
    
    For more information and details about Eden AI, check out this link: : https://docs.edenai.co/docs/additional-parameters
    
    ## Chaining Calls
    """
    logger.info("## Chaining Calls")
    
    
    prompt = ChatPromptTemplate.from_template(
        "What is a good name for a company that makes {product}?"
    )
    chain = prompt | chat
    
    chain.invoke({"product": "healthy snacks"})
    
    """
    ## Tools
    
    ### bind_tools()
    
    With `ChatEdenAI.bind_tools`, we can easily pass in Pydantic classes, dict schemas, LangChain tools, or even functions as tools to the model.
    """
    logger.info("## Tools")
    
    
    llm = ChatEdenAI(provider="ollama", temperature=0.2, max_tokens=500)
    
    
    class GetWeather(BaseModel):
        """Get the current weather in a given location"""
    
        location: str = Field(..., description="The city and state, e.g. San Francisco, CA")
    
    
    llm_with_tools = llm.bind_tools([GetWeather])
    
    ai_msg = llm_with_tools.invoke(
        "what is the weather like in San Francisco",
    )
    ai_msg
    
    ai_msg.tool_calls
    
    """
    ### with_structured_output()
    
    The BaseChatModel.with_structured_output interface makes it easy to get structured output from chat models. You can use ChatEdenAI.with_structured_output, which uses tool-calling under the hood), to get the model to more reliably return an output in a specific format:
    """
    logger.info("### with_structured_output()")
    
    structured_llm = llm.with_structured_output(GetWeather)
    structured_llm.invoke(
        "what is the weather like in San Francisco",
    )
    
    """
    ### Passing Tool Results to model
    
    Here is a full example of how to use a tool. Pass the tool output to the model, and get the result back from the model
    """
    logger.info("### Passing Tool Results to model")
    
    
    
    @tool
    def add(a: int, b: int) -> int:
        """Adds a and b.
    
        Args:
            a: first int
            b: second int
        """
        return a + b
    
    
    llm = ChatEdenAI(
        provider="ollama",
        max_tokens=1000,
        temperature=0.2,
    )
    
    llm_with_tools = llm.bind_tools([add], tool_choice="required")
    
    query = "What is 11 + 11?"
    
    messages = [HumanMessage(query)]
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    
    tool_call = ai_msg.tool_calls[0]
    tool_output = add.invoke(tool_call["args"])
    
    messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
    
    llm_with_tools.invoke(messages).content
    
    """
    ### Streaming
    
    Eden AI does not currently support streaming tool calls. Attempting to stream will yield a single final message.
    """
    logger.info("### Streaming")
    
    list(llm_with_tools.stream("What's 9 + 9"))
    
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