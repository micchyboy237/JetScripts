async def main():
    from jet.transformers.formatters import format_json
    from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapterResponses
    from jet.logger import CustomLogger
    from llama_index.core.llms import ChatMessage
    from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock
    from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
    from llama_index.core.prompts import PromptTemplate
    from llama_index.core.tools import FunctionTool
    from pydantic import BaseModel
    from typing import List
    import base64
    import os
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    """
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/openai_responses.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

    # OllamaFunctionCalling Responses API

    This notebook shows how to use the OllamaFunctionCalling Responses LLM.

    If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
    """
    logger.info("# OllamaFunctionCalling Responses API")

    # %pip install llama-index llama-index-llms-ollama

    """
    ## Basic Usage
    """
    logger.info("## Basic Usage")

    # os.environ["OPENAI_API_KEY"] = "..."

    llm = OllamaFunctionCallingAdapterResponses(
        model="llama3.2",
    )

    """
    #### Call `complete` with a prompt
    """
    logger.info("#### Call `complete` with a prompt")

    resp = llm.complete("Paul Graham is ")

    logger.debug(resp)

    """
    #### Call `chat` with a list of messages
    """
    logger.info("#### Call `chat` with a list of messages")

    messages = [
        ChatMessage(
            role="system", content="You are a pirate with a colorful personality"
        ),
        ChatMessage(role="user", content="What is your name"),
    ]
    resp = llm.chat(messages)

    logger.debug(resp)

    """
    ## Streaming

    Using `stream_complete` endpoint
    """
    logger.info("## Streaming")

    resp = llm.stream_complete("Paul Graham is ")

    for r in resp:
        logger.debug(r.delta, end="")

    """
    Using `stream_chat` endpoint
    """
    logger.info("Using `stream_chat` endpoint")

    messages = [
        ChatMessage(
            role="system", content="You are a pirate with a colorful personality"
        ),
        ChatMessage(role="user", content="What is your name"),
    ]
    resp = llm.stream_chat(messages)

    for r in resp:
        logger.debug(r.delta, end="")

    """
    ## Configure Parameters

    The Respones API supports many options:
    - Setting the model name
    - Generation parameters like temperature, top_p, max_output_tokens
    - enabling built-in tool calling
    - setting the resoning effort for O-series models
    - tracking previous responses for automatic conversation history
    - and more!

    ### Basic Parameters
    """
    logger.info("## Configure Parameters")

    llm = OllamaFunctionCallingAdapterResponses(
        model="llama3.2",
        temperature=0.5,  # default is 0.1
        max_output_tokens=100,  # default is None
        top_p=0.95,  # default is 1.0
    )

    """
    ### Built-in Tool Calling

    The responses API supports built-in tool calling, which you can read more about [here](https://platform.openai.com/docs/guides/tools?api-mode=responses).

    Configuring this means that the LLM will automatically call the tool and use it to augment the response.

    Tools are defined as a list of dictionaries, each containing settings for a tool.

    Below is an example of using the built-in web search tool.
    """
    logger.info("### Built-in Tool Calling")

    llm = OllamaFunctionCallingAdapterResponses(
        model="llama3.2",
        built_in_tools=[{"type": "web_search_preview"}],
    )

    resp = llm.chat(
        [ChatMessage(role="user", content="What is the weather in San Francisco?")]
    )
    logger.debug(resp)
    logger.debug("========" * 2)
    logger.debug(resp.additional_kwargs)

    """
    ## Reasoning Effort

    For O-series models, you can set the reasoning effort to control the amount of time the model will spend reasoning.

    See the [OllamaFunctionCalling API docs](https://platform.openai.com/docs/guides/reasoning?api-mode=responses) for more information.
    """
    logger.info("## Reasoning Effort")

    llm = OllamaFunctionCallingAdapterResponses(
        model="o3-mini",
        reasoning_options={"effort": "high"},
    )

    resp = llm.chat(
        [ChatMessage(role="user", content="What is the meaning of life?")]
    )
    logger.debug(resp)
    logger.debug("========" * 2)
    logger.debug(resp.additional_kwargs)

    """
    ## Image Support

    OllamaFunctionCalling has support for images in the input of chat messages for many models.

    Using the content blocks feature of chat messages, you can easily combone text and images in a single LLM prompt.
    """
    logger.info("## Image Support")

    # !wget https://cdn.pixabay.com/photo/2016/07/07/16/46/dice-1502706_640.jpg -O image.png

    llm = OllamaFunctionCallingAdapterResponses(model="llama3.2")

    messages = [
        ChatMessage(
            role="user",
            blocks=[
                ImageBlock(path="image.png"),
                TextBlock(text="Describe the image in a few sentences."),
            ],
        )
    ]

    resp = llm.chat(messages)
    logger.debug(resp.message.content)

    """
    ## Using Function/Tool Calling

    OllamaFunctionCalling models have native support for function calling. This conveniently integrates with LlamaIndex tool abstractions, letting you plug in any arbitrary Python function to the LLM.

    In the example below, we define a function to generate a Song object.
    """
    logger.info("## Using Function/Tool Calling")

    class Song(BaseModel):
        """A song with name and artist"""

        name: str
        artist: str

    def generate_song(name: str, artist: str) -> Song:
        """Generates a song with provided name and artist."""
        return Song(name=name, artist=artist)

    tool = FunctionTool.from_defaults(fn=generate_song)

    """
    The `strict` parameter tells OllamaFunctionCalling whether or not to use constrained sampling when generating tool calls/structured outputs. This means that the generated tool call schema will always contain the expected fields.

    Since this seems to increase latency, it defaults to false.
    """
    logger.info("The `strict` parameter tells OllamaFunctionCalling whether or not to use constrained sampling when generating tool calls/structured outputs. This means that the generated tool call schema will always contain the expected fields.")

    llm = OllamaFunctionCallingAdapterResponses(model="llama3.2", strict=True)
    response = llm.predict_and_call(
        [tool],
        "Write a random song for me",
    )
    logger.debug(str(response))

    """
    We can also do multiple function calling.
    """
    logger.info("We can also do multiple function calling.")

    llm = OllamaFunctionCallingAdapterResponses(model="llama3.2")
    response = llm.predict_and_call(
        [tool],
        "Generate five songs from the Beatles",
        allow_parallel_tool_calls=True,
    )
    for s in response.sources:
        logger.debug(
            f"Name: {s.tool_name}, Input: {s.raw_input}, Output: {str(s)}")

    """
    ### Manual Tool Calling

    If you want to control how a tool is called, you can also split the tool calling and tool selection into their own steps.

    First, lets select a tool.
    """
    logger.info("### Manual Tool Calling")

    llm = OllamaFunctionCallingAdapterResponses(model="llama3.2")

    chat_history = [ChatMessage(
        role="user", content="Write a random song for me")]

    resp = llm.chat_with_tools([tool], chat_history=chat_history)

    """
    Now, lets call the tool the LLM selected (if any).

    If there was a tool call, we should send the results to the LLM to generate the final response (or another tool call!).
    """
    logger.info("Now, lets call the tool the LLM selected (if any).")

    tools_by_name = {t.metadata.name: t for t in [tool]}
    tool_calls = llm.get_tool_calls_from_response(
        resp, error_on_no_tool_call=False
    )

    while tool_calls:
        chat_history.append(resp.message)

        for tool_call in tool_calls:
            tool_name = tool_call.tool_name
            tool_kwargs = tool_call.tool_kwargs

            logger.debug(f"Calling {tool_name} with {tool_kwargs}")
            tool_output = tool(**tool_kwargs)
            chat_history.append(
                ChatMessage(
                    role="tool",
                    content=str(tool_output),
                    additional_kwargs={"call_id": tool_call.tool_id},
                )
            )

            resp = llm.chat_with_tools([tool], chat_history=chat_history)
            tool_calls = llm.get_tool_calls_from_response(
                resp, error_on_no_tool_call=False
            )

    """
    Now, we should have a final response!
    """
    logger.info("Now, we should have a final response!")

    logger.debug(resp.message.content)

    """
    ## Structured Prediction

    An important use case for function calling is extracting structured objects. LlamaIndex provides an intuitive interface for converting any LLM into a structured LLM - simply define the target Pydantic class (can be nested), and given a prompt, we extract out the desired object.
    """
    logger.info("## Structured Prediction")

    class MenuItem(BaseModel):
        """A menu item in a restaurant."""

        course_name: str
        is_vegetarian: bool

    class Restaurant(BaseModel):
        """A restaurant with name, city, and cuisine."""

        name: str
        city: str
        cuisine: str
        menu_items: List[MenuItem]

    llm = OllamaFunctionCallingAdapterResponses(model="llama3.2")
    prompt_tmpl = PromptTemplate(
        "Generate a restaurant in a given city {city_name}"
    )
    restaurant_obj = (
        llm.as_structured_llm(Restaurant)
        .complete(prompt_tmpl.format(city_name="Dallas"))
        .raw
    )

    restaurant_obj

    """
    ## Async
    """
    logger.info("## Async")

    llm = OllamaFunctionCallingAdapterResponses(model="llama3.2")

    resp = llm.complete("Paul Graham is ")
    logger.success(format_json(resp))

    logger.debug(resp)

    resp = llm.stream_complete("Paul Graham is ")
    logger.success(format_json(resp))

    async for delta in resp:
        logger.debug(delta.delta, end="")

    """
    Async function calling is also supported.
    """
    logger.info("Async function calling is also supported.")

    llm = OllamaFunctionCallingAdapterResponses(model="llama3.2")
    response = llm.predict_and_call([tool], "Generate a random song")
    logger.success(format_json(response))
    logger.debug(str(response))

    """
    ## Additional kwargs

    If there are additional kwargs not present in the constructor, you can set them at a per-instance level with `additional_kwargs`.

    These will be passed into every call to the LLM.
    """
    logger.info("## Additional kwargs")

    llm = OllamaFunctionCallingAdapterResponses(
        model="llama3.2", additional_kwargs={"user": "your_user_id"}
    )
    resp = llm.complete("Paul Graham is ")
    logger.debug(resp)

    """
    ## Image generation

    You can use [image generation](https://platform.openai.com/docs/guides/image-generation?image-generation-model=gpt-image-1#generate-images) by passing, as a built-in-tool, `{'type': 'image_generation'}` or, if you want to enable streaming, `{'type': 'image_generation', 'partial_images': 2}`:
    """
    logger.info("## Image generation")

    llm = OllamaFunctionCallingAdapterResponses(
        model="llama3.2", request_timeout=300.0, context_window=4096, built_in_tools=[{"type": "image_generation"}]
    )
    messages = [
        ChatMessage.from_str(
            content="A llama dancing with a cat in a meadow", role="user"
        )
    ]
    response = llm.chat(
        messages
    )  # response = llm.chat(messages) for an implementation
    logger.success(format_json()  # response))
    for block in response.message.blocks:
        if isinstance(block, ImageBlock):
            with open("llama_and_cat_dancing.png", "wb") as f:
                f.write(bas64.b64decode(block.image))
        elif isinstance(block, TextBlock):
            logger.debug(block.text)

    llm_stream=OllamaFunctionCallingAdapterResponses(
        model="llama3.2", request_timeout=300.0, context_window=4096,
        built_in_tools=[{"type": "image_generation", "partial_images": 2}],
    )
    response=llm_stream.stream_chat(
        messages
    # response = await llm_stream.asteam_chat(messages) for an async implementation
    )
    logger.success(format_json()  # response))
    for event in response:
        for block in event.message.blocks:
            if isinstance(block, ImageBlock):
                with open(f"llama_and_cat_dancing_{block.detail}.png", "wb") as f:
                    f.write(bas64.b64decode(block.image))
            elif isinstance(block, TextBlock):
                logger.debug(block.text)

    """
    ## MCP Remote calls

    You can call any [remote MCP](https://platform.openai.com/docs/guides/tools-remote-mcp) through the OllamaFunctionCalling Responses API just by passing the MCP specifics as a built-in tool to the LLM
    """
    logger.info("## MCP Remote calls")


    llm=OllamaFunctionCallingAdapterResponses(
        model="llama3.2", request_timeout=300.0, context_window=4096,
        built_in_tools=[
            {
                "type": "mcp",
                "server_label": "deepwiki",
                "server_url": "https://mcp.deepwiki.com/mcp",
                "require_approval": "never",
            }
        ],
    )
    messages=[
        ChatMessage.from_str(
            content="What transport protocols are supported in the 2025-03-26 version of the MCP spec?",
            role="user",
        )
    ]
    response=llm.chat(messages)
    logger.debug(response.message.content)
    logger.debug(response.raw.output[0])

    """
    ## Code interpreter

    You can use the [Code Interpreter](https://platform.openai.com/docs/guides/tools-code-interpreter) just by setting, as a built-in tool, `"type": "code_interpreter", "container": { "type": "auto" }`.
    """
    logger.info("## Code interpreter")


    llm=OllamaFunctionCallingAdapterResponses(
        model="llama3.2", request_timeout=300.0, context_window=4096,
        built_in_tools=[
            {
                "type": "code_interpreter",
                "container": {"type": "auto"},
            }
        ],
    )
    messages=messages=[
        ChatMessage.from_str(
            content="I need to solve the equation 3x + 11 = 14. Can you help me?",
            role="user",
        )
    ]
    response=llm.chat(messages)
    logger.debug(response.message.content)
    logger.debug(response.raw.output[0])

    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    try:
        loop=asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())
