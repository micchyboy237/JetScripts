async def main():
    from jet.transformers.formatters import format_json
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core import PromptTemplate
    from llama_index.core.agent.workflow import AgentStream, ToolCallResult
    from llama_index.core.agent.workflow import ReActAgent
    from llama_index.core.workflow import Context
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/react_agent.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # ReActAgent - A Simple Intro with Calculator Tools
    
    This is a notebook that showcases the ReAct agent over very simple calculator tools (no fancy RAG pipelines or API calls).
    
    We show how it can reason step-by-step over different tools to achieve the end goal.
    
    The main advantage of the ReAct agent over a Function Calling agent is that it can work with any LLM regardless of whether it supports function calling.
    
    If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
    """
    logger.info("# ReActAgent - A Simple Intro with Calculator Tools")
    
    # %pip install llama-index
    
    
    # os.environ["OPENAI_API_KEY"] = "sk-..."
    
    """
    ## Define Function Tools
    
    We setup some trivial `multiply` and `add` tools. Note that you can define arbitrary functions and pass it to the `FunctionTool` (which will process the docstring and parameter signature).
    """
    logger.info("## Define Function Tools")
    
    def multiply(a: int, b: int) -> int:
        """Multiply two integers and returns the result integer"""
        return a * b
    
    
    def add(a: int, b: int) -> int:
        """Add two integers and returns the result integer"""
        return a + b
    
    """
    ## Run Some Queries
    """
    logger.info("## Run Some Queries")
    
    
    llm = OllamaFunctionCallingAdapter(model="llama3.2")
    agent = ReActAgent(tools=[multiply, add], llm=llm)
    
    ctx = Context(agent)
    
    """
    ## Run Some Example Queries
    
    By streaming the result, we can see the full response, including the thought process and tool calls.
    
    If we wanted to stream only the result, we can buffer the stream and start streaming once `Answer:` is in the response.
    """
    logger.info("## Run Some Example Queries")
    
    
    handler = agent.run("What is 20+(2*4)?", ctx=ctx)
    
    async for ev in handler.stream_events():
        if isinstance(ev, AgentStream):
            logger.debug(f"{ev.delta}", end="", flush=True)
    
    response = await handler
    logger.success(format_json(response))
    logger.success(format_json(response))
    
    logger.debug(str(response))
    
    logger.debug(response.tool_calls)
    
    """
    ## View Prompts
    
    Let's take a look at the core system prompt powering the ReAct agent! 
    
    Within the agent, the current conversation history is dumped below this line.
    """
    logger.info("## View Prompts")
    
    prompt_dict = agent.get_prompts()
    for k, v in prompt_dict.items():
        logger.debug(f"Prompt: {k}\n\nValue: {v.template}")
    
    """
    ### Customizing the Prompt
    
    For fun, let's try instructing the agent to output the answer along with reasoning in bullet points. See "## Additional Rules" section.
    """
    logger.info("### Customizing the Prompt")
    
    
    react_system_header_str = """\
    
    You are designed to help with a variety of tasks, from answering questions \
        to providing summaries to other types of analyses.
    
    You have access to a wide variety of tools. You are responsible for using
    the tools in any sequence you deem appropriate to complete the task at hand.
    This may require breaking the task into subtasks and using different tools
    to complete each subtask.
    
    You have access to the following tools:
    {tool_desc}
    
    To answer the question, please use the following format.
    
    ```
    Thought: I need to use a tool to help me answer the question.
    Action: tool name (one of {tool_names}) if using a tool.
    Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
    ```
    
    Please ALWAYS start with a Thought.
    
    Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.
    
    If this format is used, the user will respond in the following format:
    
    ```
    Observation: tool response
    ```
    
    You should keep repeating the above format until you have enough information
    to answer the question without using any more tools. At that point, you MUST respond
    in the one of the following two formats:
    
    ```
    Thought: I can answer without using any more tools.
    Answer: [your answer here]
    ```
    
    ```
    Thought: I cannot answer the question with the provided tools.
    Answer: Sorry, I cannot answer your query.
    ```
    
    - The answer MUST contain a sequence of bullet points that explain how you arrived at the answer. This can include aspects of the previous conversation history.
    - You MUST obey the function signature of each tool. Do NOT pass in no arguments if the function expects arguments.
    
    Below is the current conversation consisting of interleaving human and assistant messages.
    
    """
    react_system_prompt = PromptTemplate(react_system_header_str)
    
    agent.get_prompts()
    
    agent.update_prompts({"react_header": react_system_prompt})
    
    handler = agent.run("What is 5+3+2")
    
    async for ev in handler.stream_events():
        if isinstance(ev, AgentStream):
            logger.debug(f"{ev.delta}", end="", flush=True)
    
    response = await handler
    logger.success(format_json(response))
    logger.success(format_json(response))
    
    logger.debug(response)
    
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