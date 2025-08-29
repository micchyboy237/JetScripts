async def main():
    from jet.transformers.formatters import format_json
    from IPython.display import clear_output
    from datetime import datetime
    from jet.logger import CustomLogger
    from llama_index.core import Document
    from llama_index.core import Settings
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.bridge.pydantic import BaseModel
    from llama_index.core.llms import ChatMessage
    from llama_index.core.llms import ChatMessage, CitationBlock
    from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
    from llama_index.core.llms import CitableBlock, TextBlock
    from llama_index.core.llms import CitationBlock
    from llama_index.core.prompts import PromptTemplate
    from llama_index.core.tools import FunctionTool
    from llama_index.llms.anthropic import Anthropic
    from pprint import pprint
    from typing import List
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/anthropic.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # Anthropic
    
    Anthropic offers many state-of-the-art models from the haiku, sonnet, and opus families.
    
    Read on to learn how to use these models with LlamaIndex!
    
    If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
    """
    logger.info("# Anthropic")
    
    # %pip install llama-index-llms-anthropic
    
    """
    #### Set Tokenizer
    
    First we want to set the tokenizer, which is slightly different than TikToken. This ensures that token counting is accurate throughout the library.
    
    **NOTE**: Anthropic recently updated their token counting API. Older models like claude-2.1 are no longer supported for token counting in the latest versions of the Anthropic python client.
    """
    logger.info("#### Set Tokenizer")
    
    
    tokenizer = Anthropic().tokenizer
    Settings.tokenizer = tokenizer
    
    """
    ## Basic Usage
    """
    logger.info("## Basic Usage")
    
    
    # os.environ["ANTHROPIC_API_KEY"] = "sk-..."
    
    """
    You can call `complete` with a prompt:
    """
    logger.info("You can call `complete` with a prompt:")
    
    
    llm = Anthropic(model="claude-sonnet-4-0")
    
    resp = llm.complete("Who is Paul Graham?")
    
    logger.debug(resp)
    
    """
    You can also call `chat` with a list of chat messages:
    """
    logger.info("You can also call `chat` with a list of chat messages:")
    
    
    messages = [
        ChatMessage(
            role="system", content="You are a pirate with a colorful personality"
        ),
        ChatMessage(role="user", content="Tell me a story"),
    ]
    llm = Anthropic(model="claude-sonnet-4-0")
    resp = llm.chat(messages)
    
    logger.debug(resp)
    
    """
    ## Streaming Support
    
    Every method supports streaming through the `stream_` prefix.
    """
    logger.info("## Streaming Support")
    
    
    llm = Anthropic(model="claude-sonnet-4-0")
    
    resp = llm.stream_complete("Who is Paul Graham?")
    for r in resp:
        logger.debug(r.delta, end="")
    
    
    messages = [
        ChatMessage(role="user", content="Who is Paul Graham?"),
    ]
    
    resp = llm.stream_chat(messages)
    for r in resp:
        logger.debug(r.delta, end="")
    
    """
    ## Async Usage
    
    Every synchronous method has an async counterpart.
    """
    logger.info("## Async Usage")
    
    
    llm = Anthropic(model="claude-sonnet-4-0")
    
    resp = llm.stream_complete("Who is Paul Graham?")
    logger.success(format_json(resp))
    logger.success(format_json(resp))
    async for r in resp:
        logger.debug(r.delta, end="")
    
    messages = [
        ChatMessage(role="user", content="Who is Paul Graham?"),
    ]
    
    resp = llm.chat(messages)
    logger.success(format_json(resp))
    logger.success(format_json(resp))
    logger.debug(resp)
    
    """
    ## Vertex AI Support
    
    By providing the `region` and `project_id` parameters (either through environment variables or directly), you can use an Anthropic model through Vertex AI.
    """
    logger.info("## Vertex AI Support")
    
    
    os.environ["ANTHROPIC_PROJECT_ID"] = "YOUR PROJECT ID HERE"
    os.environ["ANTHROPIC_REGION"] = "YOUR PROJECT REGION HERE"
    
    """
    Do keep in mind that setting region and project_id here will make Anthropic use the Vertex AI client
    
    ## Bedrock Support
    
    LlamaIndex also supports Anthropic models through AWS Bedrock.
    """
    logger.info("## Bedrock Support")
    
    
    llm = Anthropic(
        model="anthropic.claude-3-7-sonnet-20250219-v1:0",
        aws_region="us-east-1",
    )
    
    resp = llm.complete("Who is Paul Graham?")
    
    """
    ## Multi-Modal Support
    
    Using `ChatMessage` objects, you can pass in images and text to the LLM.
    """
    logger.info("## Multi-Modal Support")
    
    # !wget https://cdn.pixabay.com/photo/2021/12/12/20/00/play-6865967_640.jpg -O image.jpg
    
    
    llm = Anthropic(model="claude-sonnet-4-0")
    
    messages = [
        ChatMessage(
            role="user",
            blocks=[
                ImageBlock(path="image.jpg"),
                TextBlock(text="What is in this image?"),
            ],
        )
    ]
    
    resp = llm.chat(messages)
    logger.debug(resp)
    
    """
    ## Prompt Caching
    
    Anthropic models support the idea of prompt cahcing -- wherein if a prompt is repeated multiple times, or the start of a prompt is repeated, the LLM can reuse pre-calculated attention results to speed up the response and lower costs.
    
    To enable prompt caching, you can set `cache_control` on your `ChatMessage` objects, or set `cache_idx` on the LLM to always cache the first X messages (with -1 being all messages).
    """
    logger.info("## Prompt Caching")
    
    
    llm = Anthropic(model="claude-sonnet-4-0")
    
    messages = [
        ChatMessage(
            role="user",
            content="<some very long prompt>",
            additional_kwargs={"cache_control": {"type": "ephemeral"}},
        ),
    ]
    
    resp = llm.chat(messages)
    
    llm = Anthropic(model="claude-sonnet-4-0", cache_idx=-1)
    
    resp = llm.chat(messages)
    
    """
    ## Structured Prediction
    
    LlamaIndex provides an intuitive interface for converting any Anthropic LLMs into a structured LLM through `structured_predict` - simply define the target Pydantic class (can be nested), and given a prompt, we extract out the desired object.
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
    
    
    llm = Anthropic(model="claude-sonnet-4-0")
    prompt_tmpl = PromptTemplate(
        "Generate a restaurant in a given city {city_name}"
    )
    
    restaurant_obj = (
        llm.as_structured_llm(Restaurant)
        .complete(prompt_tmpl.format(city_name="Miami"))
        .raw
    )
    
    restaurant_obj
    
    """
    #### Structured Prediction with Streaming
    
    Any LLM wrapped with `as_structured_llm` supports streaming through `stream_chat`.
    """
    logger.info("#### Structured Prediction with Streaming")
    
    
    input_msg = ChatMessage.from_str("Generate a restaurant in San Francisco")
    
    sllm = llm.as_structured_llm(Restaurant)
    stream_output = sllm.stream_chat([input_msg])
    for partial_output in stream_output:
        clear_output(wait=True)
        plogger.debug(partial_output.raw.dict())
        restaurant_obj = partial_output.raw
    
    restaurant_obj
    
    """
    ## Model Thinking
    
    With `claude-3.7 Sonnet`, you can enable the model to "think" harder about a task, generating a chain-of-thought response before writing out the final answer.
    
    You can enable this by passing in the `thinking_dict` parameter to the constructor, specififying the amount of tokens to reserve for the thinking process.
    """
    logger.info("## Model Thinking")
    
    
    llm = Anthropic(
        model="claude-sonnet-4-0",
        max_tokens=64000,
        temperature=1.0,
        thinking_dict={"type": "enabled", "budget_tokens": 1600},
    )
    
    messages = [
        ChatMessage(role="user", content="(1234 * 3421) / (231 + 2341) = ?")
    ]
    
    resp_gen = llm.stream_chat(messages)
    
    for r in resp_gen:
        logger.debug(r.delta, end="")
    
    logger.debug()
    logger.debug(r.message.content)
    
    logger.debug(r.message.additional_kwargs["thinking"]["signature"])
    
    """
    We can also expose the exact thinking process:
    """
    logger.info("We can also expose the exact thinking process:")
    
    logger.debug(r.message.additional_kwargs["thinking"]["thinking"])
    
    """
    ## Tool/Function Calling
    
    Anthropic supports direct tool/function calling through the API. Using LlamaIndex, we can implement some core agentic tool calling patterns.
    """
    logger.info("## Tool/Function Calling")
    
    
    llm = Anthropic(model="claude-sonnet-4-0")
    
    
    def get_current_time() -> dict:
        """Get the current time"""
        return {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    
    tool = FunctionTool.from_defaults(fn=get_current_time)
    
    """
    We can simply do a single pass to call the tool and get the result:
    """
    logger.info("We can simply do a single pass to call the tool and get the result:")
    
    resp = llm.predict_and_call([tool], "What is the current time?")
    logger.debug(resp)
    
    """
    We can also use lower-level APIs to implement an agentic tool-calling loop!
    """
    logger.info("We can also use lower-level APIs to implement an agentic tool-calling loop!")
    
    chat_history = [ChatMessage(role="user", content="What is the current time?")]
    tools_by_name = {t.metadata.name: t for t in [tool]}
    
    resp = llm.chat_with_tools([tool], chat_history=chat_history)
    tool_calls = llm.get_tool_calls_from_response(
        resp, error_on_no_tool_call=False
    )
    
    if not tool_calls:
        logger.debug(resp)
    else:
        while tool_calls:
            chat_history.append(resp.message)
    
            for tool_call in tool_calls:
                tool_name = tool_call.tool_name
                tool_kwargs = tool_call.tool_kwargs
    
                logger.debug(f"Calling {tool_name} with {tool_kwargs}")
                tool_output = tool.call(**tool_kwargs)
                logger.debug("Tool output: ", tool_output)
                chat_history.append(
                    ChatMessage(
                        role="tool",
                        content=str(tool_output),
                        additional_kwargs={"tool_call_id": tool_call.tool_id},
                    )
                )
    
                resp = llm.chat_with_tools([tool], chat_history=chat_history)
                tool_calls = llm.get_tool_calls_from_response(
                    resp, error_on_no_tool_call=False
                )
        logger.debug("Final response: ", resp.message.content)
    
    """
    ## Server-Side Tool Calling
    
    Anthropic now also supports server-side tool calling in latest versions. 
    
    Here's an example of how to use it:
    """
    logger.info("## Server-Side Tool Calling")
    
    
    llm = Anthropic(
        model="claude-sonnet-4-0",
        max_tokens=1024,
        tools=[
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 3,  # Limit to 3 searches
            }
        ],
    )
    
    response = llm.complete("What are the latest AI research trends?")
    
    logger.debug(response.text)
    
    for citation in response.citations:
        logger.debug(f"Source: {citation.get('url')} - {citation.get('cited_text')}")
    
    """
    ## Tool Calling + Citations
    
    In `llama-index-core>=0.12.46` + `llama-index-llms-anthropic>=0.7.6`, we've added support for outputting citable tool results!
    
    Using Anthropic, you can now utilize server-side citations to cite specific parts of your tool results.
    
    If the LLM cites a tool result, the citation will appear in the output as a `CitationBlock`, containing the source, title, and cited content.
    
    Let's cover a few ways to do this in practice.
    
    First, let's define a dummy tool/function that returns a citable block.
    """
    logger.info("## Tool Calling + Citations")
    
    
    dummy_text = Document.example().text
    
    
    async def search_fn(query: str):
        """Useful for searching the web to answer questions."""
        return CitableBlock(
            content=[TextBlock(text=dummy_text)],
            title="Facts about LLMs and LlamaIndex",
            source="https://docs.llamaindex.ai",
        )
    
    
    search_tool = FunctionTool.from_defaults(search_fn)
    
    
    llm = Anthropic(
        model="claude-sonnet-4-0",
    )
    
    """
    ### Agents + Citable Tools
    
    You can also use these tools directly in pre-built agents, like the `FunctionAgent`, to get the same citations in the output.
    """
    logger.info("### Agents + Citable Tools")
    
    
    agent = FunctionAgent(
        tools=[search_tool],
        llm=llm,
        system_prompt="Only make one search query per user message.",
        timeout=None,
    )
    
    output = await agent.run("How do LlamaIndex and LLMs work together?")
    logger.success(format_json(output))
    logger.success(format_json(output))
    
    
    logger.debug(output.response.content)
    logger.debug("----" * 20)
    for block in output.response.blocks:
        if isinstance(block, CitationBlock):
            logger.debug("Source: ", block.source)
            logger.debug("Title: ", block.title)
            logger.debug("Cited Content:\n", block.cited_content.text)
            logger.debug("----" * 20)
    
    """
    ### Manual Tool Calling + Citations
    
    Using our tool that returns a citable block, we can manually call the LLM with the given tool in a manual agent loop.
    
    Once the LLM stops making tool calls, we can return the final response and parse the citations from the response.
    """
    logger.info("### Manual Tool Calling + Citations")
    
    
    chat_history = [
        ChatMessage(
            role="system",
            content="Only make one search query per user message.",
        ),
        ChatMessage(
            role="user", content="How do LlamaIndex and LLMs work together?"
        ),
    ]
    resp = llm.chat_with_tools([search_tool], chat_history=chat_history)
    chat_history.append(resp.message)
    
    tool_calls = llm.get_tool_calls_from_response(
        resp, error_on_no_tool_call=False
    )
    while tool_calls:
        for tool_call in tool_calls:
            if tool_call.tool_name == "search_fn":
                tool_result = search_tool.call(tool_call.tool_kwargs)
                chat_history.append(
                    ChatMessage(
                        role="tool",
                        blocks=tool_result.blocks,
                        additional_kwargs={"tool_call_id": tool_call.tool_id},
                    )
                )
    
        resp = llm.chat_with_tools([search_tool], chat_history=chat_history)
        chat_history.append(resp.message)
        tool_calls = llm.get_tool_calls_from_response(
            resp, error_on_no_tool_call=False
        )
    
    logger.debug(resp.message.content)
    logger.debug("----" * 20)
    for block in resp.message.blocks:
        if isinstance(block, CitationBlock):
            logger.debug("Source: ", block.source)
            logger.debug("Title: ", block.title)
            logger.debug("Cited Content:\n", block.cited_content.text)
            logger.debug("----" * 20)
    
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