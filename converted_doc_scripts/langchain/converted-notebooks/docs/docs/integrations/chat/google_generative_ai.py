from jet.transformers.formatters import format_json
from IPython.display import Image, display
from google.ai.generativelanguage_v1beta.types import Tool as GenAITool
from jet.logger import logger
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langchain_google_genai import (
from langchain_google_genai import ChatGoogleGenerativeAI
import base64
import os
import shutil

async def main():
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
    )
    
    
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
    ---
    sidebar_label: Google Gemini
    ---
    
    # ChatGoogleGenerativeAI
    
    Access Google's Generative AI models, including the Gemini family, directly via the Gemini API or experiment rapidly using Google AI Studio. The `langchain-google-genai` package provides the LangChain integration for these models. This is often the best starting point for individual developers.
    
    For information on the latest models, their features, context windows, etc. head to the [Google AI docs](https://ai.google.dev/gemini-api/docs/models/gemini). All model ids can be found in the [Gemini API docs](https://ai.google.dev/gemini-api/docs/models).
    
    ### Integration details
    
    | Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/google_generativeai) | Package downloads | Package latest |
    | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
    | [ChatGoogleGenerativeAI](https://python.langchain.com/api_reference/google_genai/chat_models/langchain_google_genai.chat_models.ChatGoogleGenerativeAI.html) | [langchain-google-genai](https://python.langchain.com/api_reference/google_genai/index.html) | ❌ | beta | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-google-genai?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-google-genai?style=flat-square&label=%20) |
    
    ### Model features
    
    | [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
    | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
    | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
    
    ### Setup
    
    To access Google AI models you'll need to create a Google Account, get a Google AI API key, and install the `langchain-google-genai` integration package.
    
    **1. Installation:**
    """
    logger.info("# ChatGoogleGenerativeAI")
    
    # %pip install -U langchain-google-genai
    
    """
    **2. Credentials:**
    
    Head to [https://ai.google.dev/gemini-api/docs/api-key](https://ai.google.dev/gemini-api/docs/api-key) (or via Google AI Studio) to generate a Google AI API key.
    
    ### Chat Models
    
    Use the `ChatGoogleGenerativeAI` class to interact with Google's chat models. See the [API reference](https://python.langchain.com/api_reference/google_genai/chat_models/langchain_google_genai.chat_models.ChatGoogleGenerativeAI.html) for full details.
    """
    logger.info("### Chat Models")
    
    # import getpass
    
    if "GOOGLE_API_KEY" not in os.environ:
    #     os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
    
    """
    To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:
    """
    logger.info("To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:")
    
    
    
    """
    ## Instantiation
    
    Now we can instantiate our model object and generate chat completions:
    """
    logger.info("## Instantiation")
    
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    """
    ## Invocation
    """
    logger.info("## Invocation")
    
    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]
    ai_msg = llm.invoke(messages)
    ai_msg
    
    logger.debug(ai_msg.content)
    
    """
    ## Chaining
    
    We can [chain](/docs/how_to/sequence/) our model with a prompt template like so:
    """
    logger.info("## Chaining")
    
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that translates {input_language} to {output_language}.",
            ),
            ("human", "{input}"),
        ]
    )
    
    chain = prompt | llm
    chain.invoke(
        {
            "input_language": "English",
            "output_language": "German",
            "input": "I love programming.",
        }
    )
    
    """
    ## Multimodal Usage
    
    Gemini models can accept multimodal inputs (text, images, audio, video) and, for some models, generate multimodal outputs.
    
    ### Image Input
    
    Provide image inputs along with text using a `HumanMessage` with a list content format. Make sure to use a model that supports image input, such as `gemini-2.5-flash`.
    """
    logger.info("## Multimodal Usage")
    
    
    
    message_url = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Describe the image at the URL.",
            },
            {"type": "image_url", "image_url": "https://picsum.photos/seed/picsum/200/300"},
        ]
    )
    result_url = llm.invoke([message_url])
    logger.debug(f"Response for URL image: {result_url.content}")
    
    image_file_path = "/Users/philschmid/projects/google-gemini/langchain/docs/static/img/agents_vs_chains.png"
    
    with open(image_file_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    message_local = HumanMessage(
        content=[
            {"type": "text", "text": "Describe the local image."},
            {"type": "image_url", "image_url": f"data:image/png;base64,{encoded_image}"},
        ]
    )
    result_local = llm.invoke([message_local])
    logger.debug(f"Response for local image: {result_local.content}")
    
    """
    Other supported `image_url` formats:
    - A Google Cloud Storage URI (`gs://...`). Ensure the service account has access.
    - A PIL Image object (the library handles encoding).
    
    ### Audio Input
    
    Provide audio file inputs along with text.
    """
    logger.info("### Audio Input")
    
    
    
    audio_file_path = "example_audio.mp3"
    audio_mime_type = "audio/mpeg"
    
    
    with open(audio_file_path, "rb") as audio_file:
        encoded_audio = base64.b64encode(audio_file.read()).decode("utf-8")
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Transcribe the audio."},
            {
                "type": "media",
                "data": encoded_audio,  # Use base64 string directly
                "mime_type": audio_mime_type,
            },
        ]
    )
    response = llm.invoke([message])  # Uncomment to run
    logger.debug(f"Response for audio: {response.content}")
    
    """
    ### Video Input
    
    Provide video file inputs along with text.
    """
    logger.info("### Video Input")
    
    
    
    video_file_path = "example_video.mp4"
    video_mime_type = "video/mp4"
    
    
    with open(video_file_path, "rb") as video_file:
        encoded_video = base64.b64encode(video_file.read()).decode("utf-8")
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe the first few frames of the video."},
            {
                "type": "media",
                "data": encoded_video,  # Use base64 string directly
                "mime_type": video_mime_type,
            },
        ]
    )
    response = llm.invoke([message])  # Uncomment to run
    logger.debug(f"Response for video: {response.content}")
    
    """
    ### Image Generation (Multimodal Output)
    
    Certain models (such as `gemini-2.0-flash-preview-image-generation`) can generate text and images inline. You need to specify the desired `response_modalities`. See more information on the [Gemini API docs](https://ai.google.dev/gemini-api/docs/image-generation) for details.
    """
    logger.info("### Image Generation (Multimodal Output)")
    
    
    
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-preview-image-generation")
    
    message = {
        "role": "user",
        "content": "Generate a photorealistic image of a cuddly cat wearing a hat.",
    }
    
    response = llm.invoke(
        [message],
        generation_config=dict(response_modalities=["TEXT", "IMAGE"]),
    )
    
    
    def _get_image_base64(response: AIMessage) -> None:
        image_block = next(
            block
            for block in response.content
            if isinstance(block, dict) and block.get("image_url")
        )
        return image_block["image_url"].get("url").split(",")[-1]
    
    
    image_base64 = _get_image_base64(response)
    display(Image(data=base64.b64decode(image_base64), width=300))
    
    """
    ### Image and text to image
    
    You can iterate on an image in a multi-turn conversation, as shown below:
    """
    logger.info("### Image and text to image")
    
    next_message = {
        "role": "user",
        "content": "Can you take the same image and make the cat black?",
    }
    
    response = llm.invoke(
        [message, response, next_message],
        generation_config=dict(response_modalities=["TEXT", "IMAGE"]),
    )
    
    image_base64 = _get_image_base64(response)
    display(Image(data=base64.b64decode(image_base64), width=300))
    
    """
    You can also represent an input image and query in a single message by encoding the base64 data in the [data URI scheme](https://en.wikipedia.org/wiki/Data_URI_scheme):
    """
    logger.info("You can also represent an input image and query in a single message by encoding the base64 data in the [data URI scheme](https://en.wikipedia.org/wiki/Data_URI_scheme):")
    
    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Can you make this cat orange?",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"},
            },
        ],
    }
    
    response = llm.invoke(
        [message],
        generation_config=dict(response_modalities=["TEXT", "IMAGE"]),
    )
    
    image_base64 = _get_image_base64(response)
    display(Image(data=base64.b64decode(image_base64), width=300))
    
    """
    You can also use LangGraph to manage the conversation history for you as in [this tutorial](/docs/tutorials/chatbot/).
    
    ## Tool Calling
    
    You can equip the model with tools to call.
    """
    logger.info("## Tool Calling")
    
    
    
    @tool(description="Get the current weather in a given location")
    def get_weather(location: str) -> str:
        return "It's sunny."
    
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    llm_with_tools = llm.bind_tools([get_weather])
    
    query = "What's the weather in San Francisco?"
    ai_msg = llm_with_tools.invoke(query)
    
    logger.debug(ai_msg.tool_calls)
    
    
    tool_message = ToolMessage(
        content=get_weather(*ai_msg.tool_calls[0]["args"]),
        tool_call_id=ai_msg.tool_calls[0]["id"],
    )
    llm_with_tools.invoke([ai_msg, tool_message])  # Example of passing tool result back
    
    """
    ## Structured Output
    
    Force the model to respond with a specific structure using Pydantic models.
    """
    logger.info("## Structured Output")
    
    
    
    class Person(BaseModel):
        """Information about a person."""
    
        name: str = Field(..., description="The person's name")
        height_m: float = Field(..., description="The person's height in meters")
    
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    structured_llm = llm.with_structured_output(Person)
    
    result = structured_llm.invoke(
        "Who was the 16th president of the USA, and how tall was he in meters?"
    )
    logger.debug(result)
    
    """
    ## Token Usage Tracking
    
    Access token usage information from the response metadata.
    """
    logger.info("## Token Usage Tracking")
    
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    
    result = llm.invoke("Explain the concept of prompt engineering in one sentence.")
    
    logger.debug(result.content)
    logger.debug("\nUsage Metadata:")
    logger.debug(result.usage_metadata)
    
    """
    ## Built-in tools
    
    Google Gemini supports a variety of built-in tools ([google search](https://ai.google.dev/gemini-api/docs/grounding/search-suggestions), [code execution](https://ai.google.dev/gemini-api/docs/code-execution?lang=python)), which can be bound to the model in the usual way.
    """
    logger.info("## Built-in tools")
    
    
    resp = llm.invoke(
        "When is the next total solar eclipse in US?",
        tools=[GenAITool(google_search={})],
    )
    
    logger.debug(resp.content)
    
    
    resp = llm.invoke(
        "What is 2*2, use python",
        tools=[GenAITool(code_execution={})],
    )
    
    for c in resp.content:
        if isinstance(c, dict):
            if c["type"] == "code_execution_result":
                logger.debug(f"Code execution result: {c['code_execution_result']}")
            elif c["type"] == "executable_code":
                logger.debug(f"Executable code: {c['executable_code']}")
        else:
            logger.debug(c)
    
    """
    ## Native Async
    
    Use asynchronous methods for non-blocking calls.
    """
    logger.info("## Native Async")
    
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    
    async def run_async_calls():
        result_ainvoke = await llm.ainvoke("Why is the sky blue?")
        logger.success(format_json(result_ainvoke))
        logger.debug("Async Invoke Result:", result_ainvoke.content[:50] + "...")
    
        logger.debug("\nAsync Stream Result:")
        for chunk in llm.stream(
            "Write a short poem about asynchronous programming."
        ):
            logger.debug(chunk.content, end="", flush=True)
        logger.debug("\n")
    
        results_abatch = await llm.abatch(["What is 1+1?", "What is 2+2?"])
        logger.success(format_json(results_abatch))
        logger.debug("Async Batch Results:", [res.content for res in results_abatch])
    
    
    await run_async_calls()
    
    """
    ## Safety Settings
    
    Gemini models have default safety settings that can be overridden. If you are receiving lots of "Safety Warnings" from your models, you can try tweaking the `safety_settings` attribute of the model. For example, to turn off safety blocking for dangerous content, you can construct your LLM as follows:
    """
    logger.info("## Safety Settings")
    
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
    )
    
    """
    For an enumeration of the categories and thresholds available, see Google's [safety setting types](https://ai.google.dev/api/python/google/generativeai/types/SafetySettingDict).
    
    ## API reference
    
    For detailed documentation of all ChatGoogleGenerativeAI features and configurations head to the [API reference](https://python.langchain.com/api_reference/google_genai/chat_models/langchain_google_genai.chat_models.ChatGoogleGenerativeAI.html).
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